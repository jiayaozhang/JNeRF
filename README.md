<!-- # JNeRF -->
<div align="center">
<img src="docs/logo.png" height="200"/>
</div>

## Introduction
JNeRF is an NeRF benchmark based on [Jittor](https://github.com/Jittor/jittor). JNeRF supports Instant-NGP capable of training NeRF in 5 seconds and achieves similar performance and speed to the paper.

5s training demo of Instant-NGP implemented by JNeRF:

<img src="docs/demo_5s.gif" width="300"/>

**Step 1: Install the requirements**
```shell
sudo apt-get install tcl-dev tk-dev python3-tk
git clone https://github.com/Jittor/JNeRF
cd JNeRF
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JNeRF**

JNeRF is a benchmark toolkit and can be updated frequently, so installing in editable mode is recommended.
Thus any modifications made to JNeRF will take effect without reinstallation.

```shell
cd python
python -m pip install -e .
```

After installation, you can ```import jnerf``` in python interpreter to check if it is successful or not.

## Getting Started

### Datasets

We use fox datasets and blender lego datasets for training demonstrations. 

#### Fox Dataset
We provided fox dataset (from [Instant-NGP](https://github.com/NVlabs/instant-ngp)) in this repository at `./data/fox`.

#### Lego Dataset
You can download the lego dataset in nerf_example_data.zip at https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1. And move `lego` folder to `./data/lego`.

#### Customized Datasets

If you want to train JNerf with your own dataset, then you should follow the format of our datasets. You should split your datasets into training, validation and testing sets. Each set should be paired with a json file that describes the camera parameters of each images.

### Config

We organize our configs of JNeRF in projects/. You are referred to `./projects/ngp/configs/ngp_base.py` to learn how it works.

### Train from scratch

You can train from scratch on the `lego` scene with the following command. It should be noted that since jittor is a just-in-time compilation framework, it will take some time to compile on the first run.
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py
```
NOTE: Competitors participating in the Jittor AI Challenge can use `./projects/ngp/configs/ngp_comp.py` as config.

### Test with pre-trained model

After training, the ckpt file `params.pkl` will be automatically saved in `./logs/lego/`. And you can modify the ckpt file path by setting the `ckpt_path` in the config file. 

Set the `--task` of the command to `test` to test with pre-trained model:
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py --task test
```

### Render demo video

Set the `--task` of the command to `render` to render demo video `demo.mp4` with specified camera path based on pre-trained model:
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py --task render
```

## Config Files

* Everytime when I encounter some errors I do the following in order, it will always work!

in pycharm, remember to LD_LIBRARY_PATH=0 otherwise it would has errors

1.   unset LD_LIBRARY_PATH

2.   python -m jittor

3.   python -m jittor_utils.install_cuda


* <img src="docs/config.png" width="2400"/>
 

 ## How to add the deformation module


 In NGP_Network, add these modules

```
        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        self.deform_encoder = FrequencyEncoder(input_dims=3, multires=10, log_sampling=True)

        self.time_encoder = FrequencyEncoder(input_dims=1, multires=6, log_sampling=True)

        self.sigma_encoder = HashEncoder(n_pos_dims=3, n_features_per_level=2,
                                  n_levels=16, base_resolution=16, log2_hashmap_size=19)

        self.dir_encoder = SHEncoder()
```


### New features

        ## deform network
```    
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
```

        ## sigma network
```
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
```


        #color network
```        
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
```

        #background network
```       
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
```

##Reference

* I try to move the Dnerf from torch_ngp to Jnerf

Which I would use for future use in the simualtion domain

## Reference

* https://github.com/ashawkey/torch-ngp