#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


##Hui: the use of ArgumentParse refer to 
## https://docs.python.org/zh-cn/dev/library/argparse.html

"""
We provide a converter script convert.py, to extract undistorted images and SfM information from input images. 

For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. 
Optionally, you can use ImageMagick to resize the undistorted images. 
This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution 
in corresponding folders. 

To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. 
Put the images you want to use in a directory <location>/input.

Hui: the only needed files in the initilization stage are: 
    input, images, sparse/0/cameras.bin,images.bin,points3D.bin, points3D.ply
"""


import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true') ##Hui: --name represents this parameter will be used, elsewise without --, then not be used.
parser.add_argument("--skip_matching", action='store_true') ##Flag to indicate that COLMAP info is available for images.
parser.add_argument("--source_path", "-s", required=True, type=str) ##Location of the inputs
parser.add_argument("--camera", default="OPENCV", type=str) ##Which camera model to use for the early matching steps, OPENCV by default.
parser.add_argument("--colmap_executable", default="", type=str) ##Path to the COLMAP executable (.bat on Windows).
parser.add_argument("--resize", action="store_true") ## Flag for creating resized versions of input images.
parser.add_argument("--magick_executable", default="", type=str) ##Path to the ImageMagick executable.
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True) ##Huinote: files in "/distorted/sparse" later are copied into sparse/0

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP") ###HuiNote: read images in /input folder, create /distorted/sparse/0
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse") ###HuiNote: folder /sparse
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)  ###HuiNote: useful folder /sparse/0
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':  ###HuiNote: if folder name == 0
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize): ###HuiNote:True
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True) ###HuiNote: no such folder, resize 1/2
    os.makedirs(args.source_path + "/images_4", exist_ok=True) ###HuiNote: no such folder, resize 1/4
    os.makedirs(args.source_path + "/images_8", exist_ok=True) ###HuiNote: no such folder, resize 1/8
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file) ###Huinote: save in ./images

        destination_file = os.path.join(args.source_path, "images_2", file) ###Huinote: no such files
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file) ###Huinote: no such files
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file) ###Huinote: no such files
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
