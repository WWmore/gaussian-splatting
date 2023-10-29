


import open3d as o3d
import torch
from pytorch3d.ops import points_alignment ## failed to install in Windows

path = r'C:\Users\WANGH0M\gaussian-splatting\output'
if 1:
    old_path = path + r'\bonsai_old_cut.ply'
    new_path = path + r'\bonsai_new_cut.ply'
    source = o3d.io.read_point_cloud(old_path)
    target = o3d.io.read_point_cloud(new_path)
else:
    full_path = path + r'\out_bonsai_new\point_cloud\iteration_30000\point_cloud.ply'
    new_path = path + r'\bonsai_new_cut.ply'
    source = o3d.io.read_point_cloud(full_path)
    target = o3d.io.read_point_cloud(new_path)


source_points = torch.tensor(source.points, dtype=torch.float32)
target_points = torch.tensor(source.points, dtype=torch.float32)

# Apply point cloud alignment
aligned_points = points_alignment(source_points, target_points)

# Print the aligned points
print(aligned_points)