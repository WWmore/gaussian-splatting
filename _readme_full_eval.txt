We further provide the full_eval.py script. This script specifies the routine used in our evaluation and demonstrates the use of some additional parameters, e.g., --images (-i) to define alternative image directories within COLMAP data sets. If you have downloaded and extracted all the training data, you can run it like this:


In the current version, this process takes about 7h on our reference machine containing an A6000. If you want to do the full evaluation on our pre-trained models, you can specify their download location and skip training.


If you want to compute the metrics on our paper's evaluation images, you can also skip rendering. In this case it is not necessary to provide the source datasets. You can compute metrics for multiple image sets at a time.




Command Line Arguments for full_eval.py
--skip_training
Flag to skip training stage.

--skip_rendering
Flag to skip rendering stage.

--skip_metrics
Flag to skip metrics calculation stage.

--output_path
Directory to put renderings and results in, ./eval by default, set to pre-trained model location if evaluating them.

--mipnerf360 / -m360
Path to MipNeRF360 source datasets, required if training or rendering.

--tanksandtemples / -tat
Path to Tanks&Temples source datasets, required if training or rendering.

--deepblending / -db
Path to Deep Blending source datasets, required if training or rendering.