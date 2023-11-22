
"""
pip install laspy
pip install mpl-tools
pip install matplotlib

conda install -c conda-forge laspy
conda install -c conda-forge mpld3
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
import open3d as o3d

# reading las file and copy points
path1 = r'C:\Users\WANGH0M\Desktop\LiDAR\Zenmuse L1 202107 Rio Brazos.las' ##(40824215, 3)

input_las = laspy.read(path1)
point_records = input_las.points.copy()

red = np.array(input_las.red)
green = np.array(input_las.green)
blue = np.array(input_las.blue)

red = np.round(red/np.max(red), 2) ##range [0,1]
green = np.round(green/np.max(green), 2)
blue = np.round(blue/np.max(blue), 2)
rgb = np.c_[red, green, blue]

# getting scaling and offset parameters
las_scaleX = input_las.header.scale[0]
las_offsetX = input_las.header.offset[0]
las_scaleY = input_las.header.scale[1]
las_offsetY = input_las.header.offset[1]
las_scaleZ = input_las.header.scale[2]
las_offsetZ = input_las.header.offset[2]

# calculating coordinates
p_X = np.array((point_records['X'] * las_scaleX) + las_offsetX)
p_Y = np.array((point_records['Y'] * las_scaleY) + las_offsetY)
p_Z = np.array((point_records['Z'] * las_scaleZ) + las_offsetZ)

def get_colors(inp, colormap, vmin=None, vmax=None):
    if vmin == None:
        vmin=np.nanmin(inp)
    if vmax == None:
        vmax=np.nanmax(inp)
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

xyz = np.vstack((p_X, p_Y, p_Z)).transpose()
print(xyz.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
print('pcd:', pcd)

pcd_1m = pcd.voxel_down_sample(voxel_size=1)
print('pcd_1m: ', pcd_1m)

if 0:
    o3d.visualization.draw_geometries([pcd])
elif 1:
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])
elif 0:
    rgb = get_colors(np.asarray(pcd_1m.points)[:,2], plt.cm.terrain, 
                    vmin=np.percentile(np.asarray(pcd_1m.points)[:,2],2), 
                    vmax=np.percentile(np.asarray(pcd_1m.points)[:,2],98))
    pcd_1m.colors = o3d.utility.Vector3dVector(rgb[:,0:3]) 
    #print(rgb[:,0:3]) ##show the range is [0,1]
    o3d.visualization.draw_geometries([pcd_1m])
elif 0:
    from scipy.spatial import cKDTree
    A = np.asarray(pcd_1m.points)

    k=24
    dist, indices = cKDTree(A).query(A, k=k, workers=-1)
    max_distances = np.max(dist, axis=1)
    circle_volume = max_distances**3 * (4./3.) * np.pi
    pt_density = k / circle_volume #nr of points / area

    rgb = get_colors(pt_density, plt.cm.viridis, 
                    vmin=np.percentile(pt_density,2), 
                    vmax=np.percentile(pt_density,98))
    pcd_1m.colors = o3d.utility.Vector3dVector(rgb[:,0:3])
    o3d.visualization.draw_geometries([pcd_1m])