
from plyfile import PlyData
import numpy as np

from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary

path = r'C:\Users\WANGH0M\gaussian-splatting\data_Dji_L2\sparse\0'

camera_path = path + r'\cameras.bin'
image_path = path + r'\images.bin'
point_path = path + r'\points3D.bin'



cam_intrinsics = read_intrinsics_binary(camera_path)
print('---caremas.bin: ', cam_intrinsics)
## which shows below:   
# {1: Camera(id=1, model='PINHOLE', width=5648, height=4088, params=array([3935.34649034, 3935.67303032, 2824., 2044.]))}


cam_extrinsics = read_extrinsics_binary(image_path)
print('---images.bin: ', cam_extrinsics)
## which shows below
# .....
# 295: Image(id=295, qvec=array([ 0.9398035 , -0.05777745,  0.016862  ,  0.33637304]), 
# tvec=array([ 1.9230831 ,  5.32993502, -0.16734355]), 
# camera_id=1, name='DJI_20231110105754_0297_D.JPG', 
# xys=array([[2849.5637214 ,    9.02935227],
    #    [  -8.10663443, -170.27998594],
    #    [1406.91058329,  -24.09725147],
    #    ...,
    #    [3815.54805337, 1490.00851419],
    #    [2300.20869683, 1660.60338   ],
    #    [2300.20869683, 1660.60338   ]]), 
# point3D_ids=array([   -1,    -1,    -1, ..., 96051,    -1,    -1]))}


xyz, rgb, err = read_points3D_binary(point_path)
# print('---points3D.bin: ', xyz, rgb, err)


pcd = PlyData.read(path + r'\points3D.ply')
print('---points3D.ply: ',pcd) 
##Hui: which shows property [x,y,z], [nx,ny,nz], [red, green, blue]

