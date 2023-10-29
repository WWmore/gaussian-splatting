


import open3d as o3d
import numpy as np
import copy


path = r'C:\Users\WANGH0M\gaussian-splatting\output'
full_path = path + r'\out_bonsai_new\point_cloud\iteration_30000\point_cloud.ply'
new_path = path + r'\bonsai_new_cut.ply'
source = o3d.io.read_point_cloud(full_path)
target = o3d.io.read_point_cloud(new_path)



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


print("Visualization of two point clouds before manual alignment")
draw_registration_result(source, target, np.identity(4))

# pick points from two point clouds and builds correspondences
picked_id_source = pick_points(source)
picked_id_target = pick_points(target)

assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
assert (len(picked_id_source) == len(picked_id_target))
corr = np.zeros((len(picked_id_source), 2))
corr[:, 0] = picked_id_source
corr[:, 1] = picked_id_target

# estimate rough transformation using correspondences
print("Compute a rough transform using the correspondences given by user")
p2p = o3d.registration.TransformationEstimationPointToPoint()
trans_init = p2p.compute_transformation(source, target,
                                        o3d.utility.Vector2iVector(corr))

# point-to-point ICP for refinement
print("Perform point-to-point ICP refinement")
threshold = 0.03  # 3cm distance threshold
reg_p2p = o3d.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.registration.TransformationEstimationPointToPoint())
draw_registration_result(source, target, reg_p2p.transformation)
print("")


o3d.registration.TransformationEstimation(source, target)