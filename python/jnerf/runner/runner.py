import os
import jittor as jt
from PIL import Image
import numpy as np
from tqdm import tqdm
from jnerf.ops.code_ops import *
from jnerf.dataset.dataset import jt_srgb_to_linear, jt_linear_to_srgb
from jnerf.utils.config import get_cfg, save_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES
from jnerf.models.losses.mse_loss import img2mse, mse2psnr
from jnerf.dataset import camera_path
import cv2

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=100, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=128, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    # parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    # parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()
    return opt

class Runner():
    def __init__(self):

        # from jnerf.dataset.provider import NeRFDataset
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # NeRFDataset(get_opt(), device, 'train')

        self.cfg = get_cfg()
        if self.cfg.fp16 and jt.flags.cuda_archs[0] < 70:
            print("Warning: Sm arch is lower than sm_70, fp16 is not supported. Automatically use fp32 instead.")
            self.cfg.fp16 = False
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        self.exp_name           = self.cfg.exp_name
        self.dataset            = {}
        self.dataset["train"]   = build_from_cfg(self.cfg.dataset.train, DATASETS)
        self.cfg.dataset_obj    = self.dataset["train"]
        if self.cfg.dataset.val:
            self.dataset["val"] = build_from_cfg(self.cfg.dataset.val, DATASETS)
        else:
            self.dataset["val"] = self.dataset["train"]
        self.dataset["test"]    = None
        self.model              = build_from_cfg(self.cfg.model, NETWORKS)
        self.cfg.model_obj      = self.model
        self.sampler            = build_from_cfg(self.cfg.sampler, SAMPLERS)
        self.cfg.sampler_obj    = self.sampler
        self.optimizer          = build_from_cfg(self.cfg.optim, OPTIMS, params=self.model.parameters())
        self.optimizer          = build_from_cfg(self.cfg.expdecay, OPTIMS, nested_optimizer=self.optimizer)
        self.ema_optimizer      = build_from_cfg(self.cfg.ema, OPTIMS, params=self.model.parameters())
        self.loss_func          = build_from_cfg(self.cfg.loss, LOSSES)
        self.background_color   = self.cfg.background_color
        self.tot_train_steps    = self.cfg.tot_train_steps
        self.n_rays_per_batch   = self.cfg.n_rays_per_batch
        self.using_fp16         = self.cfg.fp16
        self.save_path          = os.path.join(self.cfg.log_dir, self.exp_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.cfg.ckpt_path and self.cfg.ckpt_path is not None:
            self.ckpt_path = self.cfg.ckpt_path
        else:
            self.ckpt_path = os.path.join(self.save_path, "params.pkl")
        if self.cfg.load_ckpt:
            self.load_ckpt(self.ckpt_path)
        else:
            self.start=0

        self.cfg.m_training_step = 0
        self.val_freq = 4096
        self.image_resolutions = self.dataset["train"].resolution
        self.W = self.image_resolutions[0]
        self.H = self.image_resolutions[1]

    def train(self):
        for i in tqdm(range(self.start, self.tot_train_steps)):
            self.cfg.m_training_step = i
            img_ids, rays_o, rays_d, rgb_target, time = next(self.dataset["train"].__next__())
            training_background_color = jt.random([rgb_target.shape[0],3]).stop_grad()

            rgb_target = (rgb_target[..., :3] * rgb_target[..., 3:] + training_background_color * (1 - rgb_target[..., 3:])).detach()

            pos, dir = self.sampler.sample(img_ids, rays_o, rays_d, time, is_training=True)
            if pos is None:
                continue
            network_outputs = self.model(pos, dir, time)
            rgb = self.sampler.rays2rgb(network_outputs, training_background_color)

            loss = self.loss_func(rgb, rgb_target)
            self.optimizer.step(loss)
            self.ema_optimizer.ema_step()
            if self.using_fp16:
                self.model.set_fp16()

            if i>0 and i%self.val_freq==0:
                psnr=mse2psnr(self.val_img(i))
                print("STEP={} | LOSS={} | VAL PSNR={}".format(i,loss.mean().item(), psnr))
        self.save_ckpt(os.path.join(self.save_path, "params.pkl"))
        self.test()



    #
    # def train(self):
    #     for i in tqdm(range(self.start, self.tot_train_steps)):
    #         print("i",i)
    #         self.cfg.m_training_step = i
    #         img_ids, rays_o, rays_d, rgb_target = next(self.dataset["train"])
    #         training_background_color = jt.random([rgb_target.shape[0],3]).stop_grad()
    #
    #         rgb_target = (rgb_target[..., :3] * rgb_target[..., 3:] + training_background_color * (1 - rgb_target[..., 3:])).detach()
    #
    #         pos, dir = self.sampler.sample(img_ids, rays_o, rays_d, is_training=True)
    #         network_outputs = self.model(pos, dir, t)
    #         rgb = self.sampler.rays2rgb(network_outputs, training_background_color)
    #
    #         loss = self.loss_func(rgb, rgb_target)
    #         self.optimizer.step(loss)
    #         self.ema_optimizer.ema_step()
    #         if self.using_fp16:
    #             self.model.set_fp16()
    #
    #         if i>0 and i%self.val_freq==0:
    #             psnr=mse2psnr(self.val_img(i))
    #             print("STEP={} | LOSS={} | VAL PSNR={}".format(i,loss.mean().item(), psnr))
    #     self.save_ckpt(os.path.join(self.save_path, "params.pkl"))
    #     self.test()
    #




    def test(self, load_ckpt=False):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if self.dataset["test"] is None:
            self.dataset["test"] = build_from_cfg(self.cfg.dataset.test, DATASETS)
        if not os.path.exists(os.path.join(self.save_path, "test")):
            os.makedirs(os.path.join(self.save_path, "test"))
        mse_list=self.render_test(save_path=os.path.join(self.save_path, "test"))
        if self.dataset["test"].have_img:
            tot_psnr=0
            for mse in mse_list:
                tot_psnr += mse2psnr(mse)
            print("TOTAL TEST PSNR===={}".format(tot_psnr/len(mse_list)))

    def render(self, load_ckpt=True, save_path=None):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if save_path is None or save_path=="":
            save_path = os.path.join(self.save_path, "demo.mp4")
        else:
            assert save_path.endswith(".mp4"), "suffix of save_path need to be .mp4"
        print("rendering video with specified camera path")
        fps = 28
        W, H = self.image_resolutions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
        cam_path = camera_path.path_spherical()

        for pose in tqdm(cam_path):
            img = self.render_img_with_pose(pose)
            img = (img*255+0.5).clip(0, 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            videowriter.write(img)
        videowriter.release()
        
    def save_ckpt(self, path):
        jt.save({
            'global_step': self.cfg.m_training_step,
            'model': self.model.state_dict(),
            'sampler': self.sampler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'nested_optimizer': self.optimizer._nested_optimizer.state_dict(),
            'ema_optimizer': self.ema_optimizer.state_dict(),
        }, path)

    def load_ckpt(self, path):
        print("Loading ckpt from:",path)
        ckpt = jt.load(path)
        self.start = ckpt['global_step']
        self.model.load_state_dict(ckpt['model'])
        if self.using_fp16:
            self.model.set_fp16()
        self.sampler.load_state_dict(ckpt['sampler'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        nested=ckpt['nested_optimizer']['defaults']['param_groups'][0]
        for pg in self.optimizer._nested_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i]=jt.array(nested["values"][i])
                pg["m"][i]=jt.array(nested["m"][i])
        ema=ckpt['ema_optimizer']['defaults']['param_groups'][0]
        for pg in self.ema_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i]=jt.array(ema["values"][i])
        self.ema_optimizer.steps=ckpt['ema_optimizer']['defaults']['steps']
        
    def val_img(self, iter):
        with jt.no_grad():
            img, img_tar = self.render_img(dataset_mode="val")
            self.save_img(self.save_path+f"/img{iter}.png", img)
            self.save_img(self.save_path+f"/target{iter}.png", img_tar)
            return img2mse(
                jt.array(img), 
                jt.array(img_tar)).item()
    
    def render_test(self, save_img=True, save_path=None):
        if save_path is None:
            save_path = self.save_path
        mse_list = []
        print("rendering testset...")
        for img_i in tqdm(range(0,self.dataset["test"].n_images,1)):
            with jt.no_grad():
                imgs=[]
                for i in range(1):
                    simg, img_tar = self.render_img(dataset_mode="test", img_id=img_i)
                    imgs.append(simg)
                img = np.stack(imgs, axis=0).mean(0)
                if save_img:
                    self.save_img(save_path+f"/{self.exp_name}_r_{img_i}.png", img)
                    if self.dataset["test"].have_img:
                        self.save_img(save_path+f"/{self.exp_name}_gt_{img_i}.png", img_tar)
                mse_list.append(img2mse(
                jt.array(img), 
                jt.array(img_tar)).item())
        return mse_list

    def save_img(self, path, img):
        if isinstance(img, np.ndarray):
            ndarr = (img*255+0.5).clip(0, 255).astype('uint8')
        elif isinstance(img, jt.Var):
            ndarr = (img*255+0.5).clamp(0, 255).uint8().numpy()
        im = Image.fromarray(ndarr)
        im.save(path)

    def render_img(self, dataset_mode="train", img_id=None):
        W, H = self.image_resolutions
        H = int(H)
        W = int(W)
        if img_id is None:
            img_id = np.random.randint(0, self.dataset[dataset_mode].n_images, [1])[0]
            img_ids = jt.zeros([H*W], 'int32')+img_id
        else:
            img_ids = jt.zeros([H*W], 'int32')+img_id
        rays_o_total, rays_d_total, rays_pix_total, time = self.dataset[dataset_mode].generate_rays_total_test(
            img_ids, W, H)
        rays_pix_total = rays_pix_total.unsqueeze(-1)
        pixel = 0
        imgs = np.empty([H*W+self.n_rays_per_batch, 3])
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)

            pos, dir = self.sampler.sample(img_ids, rays_o, rays_d, time)
            if pos is None:
                continue
            network_outputs = self.model(pos, dir, time)
            rgb = self.sampler.rays2rgb(network_outputs, inference=True)
            imgs[pixel:end] = rgb.numpy()
        imgs = imgs[:H*W].reshape(H, W, 3)
        imgs_tar=jt.array(self.dataset[dataset_mode].image_data[img_id]).reshape(H, W, 4)
        imgs_tar = imgs_tar[..., :3] * imgs_tar[..., 3:] + jt.array(self.background_color) * (1 - imgs_tar[..., 3:])
        imgs_tar = imgs_tar.detach().numpy()
        jt.gc()
        return imgs, imgs_tar

    def render_img_with_pose(self, pose):
        W, H = self.image_resolutions
        H = int(H)
        W = int(W)
        fake_img_ids = jt.zeros([H*W], 'int32')
        rays_o_total, rays_d_total = self.dataset["train"].generate_rays_with_pose(pose, W, H)
        img = np.empty([H*W+self.n_rays_per_batch, 3])
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)
            pos, dir = self.sampler.sample(fake_img_ids, rays_o, rays_d)
            network_outputs = self.model(pos, dir)
            rgb = self.sampler.rays2rgb(network_outputs, inference=True)
            img[pixel:end] = rgb.numpy()
        img = img[:H*W].reshape(H, W, 3)
        return img
