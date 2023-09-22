#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

##Hui Note: the hyperparameters in this file will be used in train.py

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):  ##Hui: used in train.py
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3  ##Order of spherical harmonics to be used (no larger than 3). 3 by default.
        self._source_path = "" ##Path to the source directory containing a COLMAP or Synthetic NeRF data set.
        self._model_path = "" ## Path where the trained model should be stored (output/<random> by default).
        self._images = "images" ##Alternative subdirectory for COLMAP images (images by default).
        self._resolution = -1 ## Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively.
        self._white_background = False ##Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
        self.data_device = "cuda" 
        self.eval = False ##Add this flag to use a MipNeRF360-style training/test split for evaluation.
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False ##Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
        self.compute_cov3D_python = False ##Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
        self.debug = False ##Enables debug mode if you experience erros.
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000  ##Number of total iterations to train for, 30_000 by default.
        self.position_lr_init = 0.00016 ## lr==learning rate in deep learning; below are the same
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01 ##Position learning rate multiplier (cf. Plenoxels)
        self.position_lr_max_steps = 30_000 ##Number of steps (from 0) where position learning rate goes from initial to final
        self.feature_lr = 0.0025 ##Spherical harmonics features learning rate.
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01 ##Percentage of scene extent (0--1) a point must exceed to be forcibly densified
        self.lambda_dssim = 0.2 ##Influence of SSIM on total loss from 0 to 1
        self.densification_interval = 100 ##How frequently to densify
        self.opacity_reset_interval = 3000 ##How frequently to reset opacity
        self.densify_from_iter = 500 ##Iteration where densification starts
        self.densify_until_iter = 15_000 ##Iteration where densification stops
        self.densify_grad_threshold = 0.0002 ##Limit that decides if points should be densified based on 2D position gradient
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser): ###HuiNote: used in render.py
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args") ###HuiNote: file in /output/out_coral/cfg_args
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string) ###HuiNote: eval("sum([8, 16, 32])") >>> 56

    merged_dict = vars(args_cfgfile).copy() ###HuiNote: present initial dictornay
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict) ###HuiNote:命名空间提供了一个在大型项目下避免名字冲突的方法
