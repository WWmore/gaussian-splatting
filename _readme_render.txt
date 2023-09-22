Command Line Arguments for render.py
--model_path / -m
Path to the trained model directory you want to create renderings for.

--skip_train
Flag to skip rendering the training set.

--skip_test
Flag to skip rendering the test set.

--quiet
Flag to omit any text written to standard out pipe.

The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.

--source_path / -s
Path to the source directory containing a COLMAP or Synthetic NeRF data set.

--images / -i
Alternative subdirectory for COLMAP images (images by default).

--eval
Add this flag to use a MipNeRF360-style training/test split for evaluation.

--resolution / -r
Changes the resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. 1 by default.

--white_background / -w
Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.

--convert_SHs_python
Flag to make pipeline render with computed SHs from PyTorch instead of ours.

--convert_cov3D_python
Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.