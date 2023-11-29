"""Pipeline to use LiDAR data (pointcloud) to calibrate the 3D GS
Way1: 3DGS --> Cleaned PCs --> Colored PCs -----> Calibrated PCs  --> focused + calibrated 3DGS
                                      / \
                                       |
                                       |
                                    LiDAR PCs

Way2: 3DGS --> Colored PCs -----> Calibrated PCs --> focused + calibrated 3DGS --> Cleaned PCs
                                    
Steps:                     
* python 3dgsconverter.py -i input_3dgs.ply -o output_drcc.ply -f cc --density_filter --remove_flyers --rgb

1. read colored_pc (i.e. output_drcc.ply)
2. read lidar_pc
3. register colored_pc to lidar_pc, find the closest_index 
4. replace colored_pc = lidar_pc[closest_index]
5. save colored_pc (as output_drccc.ply)

* python 3dgsconverter.py -i input_drccc.ply -o output_drccc_3dgs.ply -f 3dgs
"""


def read_coloredPC(path):
    pass

def read_lidarPC(path):
    pass

def closest_index(PC1, PC2):
    """ find closest_index in PC2 such that PC1 ~~ PC2[closest_index]
    PC1: colored_pc
    PC2: lidar_pc
    return: closest_index
    """

def replacement(PC1, PC2, index):

    PC1 = PC2[index]

    return PC1


path = r'C:\Users\WANGH0M\gaussian-splatting\output\out_Dji\point_cloud\iteration_30000'
path1 = path + r'\point_cloud.ply'
path2 = path + r'\Dji_cc.ply'

# PC1 = read_coloredPC(path1)
# PC2 = read_lidarPC(path2)
# index = closest_index(PC1, PC2)
# coloredPC = replacement(PC1, PC2, index)

# "save coloredPC into the path"





