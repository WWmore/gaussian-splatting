
import json 
from plyfile import PlyData


path = r'C:\Users\WANGH0M\gaussian-splatting\output'

f = r'\out_bonsai_new'

carema_path = path + f + r'\cameras.json'
carema = json.load(open(carema_path))
print('---carema.json: ', carema)
##Hui: which shows id, img_name, width, height, positions[x3],rotation:[x3],[x3],[x3],[x3],fy,fx

input = PlyData.read(path + f + r'\input.ply')
print('---input.ply: ', input) 
##Hui: which shows property [x,y,z], [nx,ny,nz], [red, green, blue]


point_path = path + f + r'\point_cloud\iteration_30000\point_cloud.ply'
point_cloud = PlyData.read(point_path)
print('---point_cloud.ply: ', point_cloud) 
##Hui: which shows property [x,y,z], [nx,ny,nz], [f_dc_0,1,2], [f_rest_0,...,44], 
##                          opacity, [scale_0,1,2], [rot_0,1,2,3]


