


convert.py: to extract undistorted images and SfM information from input images. 



To run the optimizer, simply use
python train.py -s <path to COLMAP or NeRF Synthetic dataset




Evaluation:

python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
