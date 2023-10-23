This repository is forked from the [repository](https://github.com/graphdeco-inria/gaussian-splatting), which is the source code for the best paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) in SIGGRAPH 2023.

Followed up papers are updated [here](https://github.com/MrNeRF/awesome-3D-gaussian-splatting).

The installation can refer to a step-by-step [Youtube tutorial](https://www.youtube.com/watch?v=UXtuigy_wYc).

<details>
<summary><span style="font-weight: bold;">Installation steps in Anaconda</span></summary>

  - open Anaconda Prompt
  - cd C:User/<username>
  - git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
  - SET DISTUTILS_USE_SDK=1
  - conda env create --file environment.yml
  - conda activate gaussian_splatting
  #### Below packages may be needed
  - conda install -c conda-forge pillow
  - python3 -m pip install -U pip
  - python3 -m pip install pillow-heif
  - conda install -c conda-forge pcl
  - conda install -c open3d-admin open3d
  - conda install -c anaconda numpy
  #### if error: Importing the numpy C-extensions failed.
  - pip install setuptools
  - pip install numpy
  - pip install cupy
  - pip install probreg
  - pip install pillow-heif

</details>

<details>
<summary><span style="font-weight: bold;">Running in Anaconda Prompt</span></summary>

  - open Anaconda Prompt
  - conda activate gaussian_splatting
  - cd C:\User\<username>\gaussian-splatting
  - put images in the folder data/input
  - python convert.py -s data (wait within 5 mins)
  - python train.py -s data (wait around 1h)
  - output GaussianSplatting ply appears in the folder output/<name>
  - rename <name> to out
  - cd viewers/bin
  - SIBR_gaussianViewer_app.exe -m C:\Users\<username>\gaussian-splatting\output\out 
  - the GUI pops up, if not, may the CUDA support problem.

</details>
<br>

The default viewer is by SIBR. Another viewer is by Unity (free to use) as seen [here](https://github.com/aras-p/UnityGaussianSplatting). Based on my experience, the comparison between them are

Viewers
|---SIBR
|   |---Pros.: focused view of the object when opening
|   |---Cons.: need keyboard to navigate, mouse control is so bad even freezes the GUI
|---Unity
|   |---Pros.: can directly trim the Gaussian Splatting in the scene and export the ply; parameters are interactivly set
|   |---Cons.: global view of the whole scene; tilt basement; need mouse to zoom in-out; hard to control


https://github.com/WWmore/gaussian-splatting/assets/28695253/d1d8aaeb-a890-434e-95b8-acf526bea44b
