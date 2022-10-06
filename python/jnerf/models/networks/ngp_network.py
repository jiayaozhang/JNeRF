from turtle import pos, position
import jittor as jt
from jittor import nn, init
import os
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
from jnerf.ops.code_ops.fully_fused_mlp import FullyFusedMlp_weight
from jnerf.models.position_encoders.hash_encoder.grid_encode import GridEncode
from jnerf.models.position_encoders.hash_encoder.hash_encoder import HashEncoder
from jnerf.models.position_encoders.freq_encoder import FrequencyEncoder
from jnerf.models.position_encoders.sh_encoder.sh_encoder import SHEncoder

class FMLP(nn.Module):
    def __init__(self, weight_shapes, weights=None):
        super(FMLP, self).__init__()
        if weights == None:                   
            assert len(weight_shapes) > 2
            self.output_shape1 = weight_shapes[-1]
            dweights = []
            for i in range(len(weight_shapes) - 1):
                dweights.append(init.invariant_uniform((weight_shapes[i], weight_shapes[i+1]), "float16").float16())
        else:
            assert len(weights) >= 2
            self.output_shape1 = weights[-1].shape[-1]
            dweights = weights
        self.func = FullyFusedMlp_weight(dweights)
        con_weights = []
        for i in range(len(dweights)):
            if i == len(dweights) - 1:
                if dweights[i].shape[1] < 16: 
                    dweights[i] = jt.concat([dweights[i], jt.zeros((dweights[i].shape[0], 16 - dweights[i].shape[1]))], -1).float16()
            con_weights.append(dweights[i].transpose(1,0).reshape(-1))
        jt_con_weights = jt.concat(con_weights, -1)
        self.con_weights = jt_con_weights

    def execute(self, x):
        if x.shape[0] == 0:
            return jt.empty([0, self.output_shape1]).float16()
        ret = self.func(x, self.con_weights)
        if self.output_shape1 != ret.shape[1]:
            ret = ret[:,:self.output_shape1]
        return ret


@NETWORKS.register_module()
class NGPNetworks_new(nn.Module):
    def __init__(self, use_fully=True, density_hidden_layer=1, density_n_neurons=64, rgb_hidden_layer=2, rgb_n_neurons=64):
        super(NGPNetworks_new, self).__init__()
        self.use_fully = use_fully
        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.bg_radius = -1

        # self.pos_encoder = GridEncode(self.hash_func_header, n_pos_dims=3, n_features_per_level=2,
        #                           n_levels=16, base_resolution=16, log2_hashmap_size=19)

        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        self.deform_encoder = FrequencyEncoder(input_dims=3, multires=10, log_sampling=True)

        self.time_encoder = FrequencyEncoder(input_dims=1, multires=6, log_sampling=True)

        self.sigma_encoder = HashEncoder(n_pos_dims=3, n_features_per_level=2,
                                  n_levels=16, base_resolution=16, log2_hashmap_size=19)

        # self.dir_encoder = SHEncoder()

        # much smaller hash grid
        self.encoder_bg = GridEncode(hash_func_header = "\"", n_pos_dims=2, n_levels=4, base_resolution=16, log2_hashmap_size=19)

        if self.use_fully and jt.flags.cuda_archs[0] >= 75 and self.using_fp16:
            # assert self.pos_encoder.out_dim%16==0
            # assert self.dir_encoder.out_dim%16==0
            self.density_mlp = FMLP([self.deform_encoder.out_dim, density_n_neurons, 16])
            self.rgb_mlp = FMLP([self.dir_encoder.out_dim+16, rgb_n_neurons, rgb_n_neurons, 3])
        else:
            if self.use_fully and not (jt.flags.cuda_archs[0] >= 75):
                print("Warning: Sm arch is lower than sm_75, FFMLPs is not supported. Automatically use original MLPs instead.")
            elif self.use_fully and not self.using_fp16:
                print("Warning: FFMLPs only support float16. Automatically use original MLPs instead.")
            self.density_mlp = nn.Sequential(
                nn.Linear(self.deform_encoder.out_dim, density_n_neurons, bias=False),
                nn.ReLU(), 
                nn.Linear(density_n_neurons, 16, bias=False))
            self.rgb_mlp = nn.Sequential(nn.Linear(self.dir_encoder.out_dim+16, rgb_n_neurons, bias=False),
                            nn.ReLU(),
                            nn.Linear(rgb_n_neurons, rgb_n_neurons, bias=False),
                            nn.ReLU(),
                            nn.Linear(rgb_n_neurons, 3, bias=False))

        ### deform network
        self.num_layers_deform = num_layers_deform = 5
        hidden_dim_deform = 128
        in_dim_time = 1
        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.deform_encoder.out_dim + self.time_encoder.out_dim # grid dim + time
            else:
                in_dim = hidden_dim_deform
            
            if l == num_layers_deform - 1:
                out_dim = 3 # deformation for xyz
            else:
                out_dim = hidden_dim_deform
            
            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)        

        #sigma network
        self.num_layers = num_layers = 2
        self.hidden_dim = hidden_dim = 64
        self.geo_feat_dim = geo_feat_dim = 15

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.sigma_encoder.out_dim + self.time_encoder.out_dim + self.deform_encoder.out_dim  # concat everything
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)



        #color network
        self.num_layers_color = num_layers_color = 3
        self.hidden_dim_color = hidden_dim_color = 64

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.dir_encoder.out_dim + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)


        #background network
        num_layers_bg = 2
        hidden_dim_bg = 64
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.encoder_bg.output_mask + self.dir_encoder.out_dim
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def execute(self, pos_input, dir_input, t):
        # t = jt.ones([1, 1])
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(pos_input, dir_input, t)
        else:
            return self.execute_(pos_input, dir_input, t)

    def execute_(self, pos_input, dir_input, t):
        # t = jt.ones([1, 1])
        # pos_input: [N, 3], in [-bound, bound]
        # dir_input: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]
        # print("pos_input",pos_input.shape)
        # print("t",t.shape,t)
        t = t[0]
        enc_dir_input = self.dir_encoder(dir_input)
        enc_pos_input = self.deform_encoder(pos_input)

        enc_t = self.time_encoder(t) # [1, 1] --> [1, C']

        enc_t = enc_t.repeat(pos_input.shape[0], 1) # [1, C'] --> [N, C']
        # print("enc_pos_input",enc_pos_input.shape)
        # print("enc_t",enc_t.shape)
        deform = jt.concat([enc_pos_input, enc_t], dim=1) # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = jt.nn.relu(deform)
        
        pos_input = pos_input + deform

        pos_input = self.sigma_encoder(pos_input)
        h = jt.concat([pos_input, enc_pos_input, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = jt.nn.relu(h)

        sigma = jt.nn.relu(h[..., 0])
        #sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        dir_input = self.dir_encoder(dir_input)
        h = jt.concat([dir_input, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = jt.nn.relu(h)

        # sigmoid activation for rgb
        rgbs = jt.sigmoid(h)

        # return sigma, rgbs, deform
        outputs = jt.concat([rgbs, sigma.reshape(sigma.shape[0], 1)], -1)  # batchsize 4: rgbd
        # return outputs

        # density = self.density_mlp(enc_pos_input)
        # rgb = jt.concat([density, dir_input], -1)
        # rgb = self.rgb_mlp(rgb)
        # outputs = jt.concat([rgb, density[..., :1]], -1)  # batchsize 4: rgbd
        return outputs

    def density(self, pos_input):  # batchsize,3
        # density = self.deform_encoder(pos_input)
        # density = self.density_mlp(density)[:,:1]
        # return density
        t = jt.ones([1, 1])
        results = {}
        # deformation
        enc_pos_input = self.deform_encoder(pos_input)  # [N, C]
        enc_t = self.time_encoder(t)  # [1, 1] --> [1, C']

        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(pos_input.shape[0], 1)  # [1, C'] --> [N, C']

        deform = jt.concat([enc_pos_input, enc_t], dim=1)  # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = jt.nn.relu(deform)

        pos_input = pos_input + deform

        # sigma
        pos_input = self.sigma_encoder(pos_input)
        h = jt.concat([pos_input, enc_pos_input, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = jt.nn.relu(h)

        sigma = jt.nn.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # results['sigma'] = sigma
        # results['geo_feat'] = geo_feat

        # questions about how to concate the sigma, geo_feat and deform
        # results = jt.concat([deform, sigma.reshape(sigma.shape[0], 1)], -1)
        results = sigma
        return results

    def background(self, pos_input, dir_input):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(pos_input)  # [N, C]
        dir_input = self.dir_encoder(dir_input)

        h = jt.concat([dir_input, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = jt.nn.relu(h)

        # sigmoid activation for rgb
        rgbs = jt.sigmoid(h)

        return rgbs

        # allow masked inference

    def color(self, pos_input, dir_input, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = jt.zeros(mask.shape[0], 3)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            pos_input = pos_input[mask]
            dir_input = dir_input[mask]
            geo_feat = geo_feat[mask]

        dir_input = self.dir_encoder(dir_input)
        h = jt.concat([dir_input, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = jt.nn.relu(h)

        # sigmoid activation for rgb
        h = jt.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs




    def set_fp16(self):
        if self.using_fp16:
            self.density_mlp.float16()
            self.rgb_mlp.float16()
            self.pos_encoder.float16()
            self.dir_encoder.float16()
            self.deform_encoder.float16()
#
#
@NETWORKS.register_module()
class NGPNetworks(nn.Module):
    def __init__(self, use_fully=True, density_hidden_layer=1, density_n_neurons=64, rgb_hidden_layer=2, rgb_n_neurons=64):
        super(NGPNetworks, self).__init__()
        self.use_fully = use_fully
        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.pos_encoder = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS)
        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        if self.use_fully and jt.flags.cuda_archs[0] >= 75 and self.using_fp16:
            assert self.pos_encoder.out_dim%16==0
            assert self.dir_encoder.out_dim%16==0
            self.density_mlp = FMLP([self.pos_encoder.out_dim, density_n_neurons, 16])
            self.rgb_mlp = FMLP([self.dir_encoder.out_dim+16, rgb_n_neurons, rgb_n_neurons, 3])
        else:
            if self.use_fully and not (jt.flags.cuda_archs[0] >= 75):
                print("Warning: Sm arch is lower than sm_75, FFMLPs is not supported. Automatically use original MLPs instead.")
            elif self.use_fully and not self.using_fp16:
                print("Warning: FFMLPs only support float16. Automatically use original MLPs instead.")
            self.density_mlp = nn.Sequential(
                nn.Linear(self.pos_encoder.out_dim, density_n_neurons, bias=False),
                nn.ReLU(),
                nn.Linear(density_n_neurons, 16, bias=False))
            self.rgb_mlp = nn.Sequential(nn.Linear(self.dir_encoder.out_dim+16, rgb_n_neurons, bias=False),
                            nn.ReLU(),
                            nn.Linear(rgb_n_neurons, rgb_n_neurons, bias=False),
                            nn.ReLU(),
                            nn.Linear(rgb_n_neurons, 3, bias=False))
        self.set_fp16()

    def execute(self, pos_input, dir_input, time):
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(pos_input, dir_input)
        else:
            return self.execute_(pos_input, dir_input)

    def execute_(self, pos_input, dir_input):
        dir_input = self.dir_encoder(dir_input)
        pos_input = self.pos_encoder(pos_input)
        density = self.density_mlp(pos_input)
        rgb = jt.concat([density, dir_input], -1)
        rgb = self.rgb_mlp(rgb)
        outputs = jt.concat([rgb, density[..., :1]], -1)  # batchsize 4: rgbd
        return outputs

    def density(self, pos_input):  # batchsize,3
        density = self.pos_encoder(pos_input)
        density = self.density_mlp(density)[:,:1]
        return density

    def set_fp16(self):
        if self.using_fp16:
            self.density_mlp.float16()
            self.rgb_mlp.float16()
            self.pos_encoder.float16()
            self.dir_encoder.float16()