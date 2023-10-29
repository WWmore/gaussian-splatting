
"""
http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html?highlight=registration
Colored point cloud registration
"""


import open3d as o3d
import numpy as np
import copy
#-------------------------------------------


print("1. Load two point clouds and show initial pose")

if 0:
    demo_colored_icp_pcds = o3d.data.DemoColoredICPPointClouds() ## AttributeError: module 'open3d' has no attribute 'data'
    source = o3d.io.read_point_cloud(demo_colored_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_colored_icp_pcds.paths[1])
else:
    path = r'C:\Users\WANGH0M\gaussian-splatting\output'
    old_path = path + r'\bonsai_old_cut.ply'
    new_path = path + r'\bonsai_new_cut.ply'
    source = o3d.io.read_point_cloud(old_path)
    target = o3d.io.read_point_cloud(new_path)

def unitscale(source, target):
    print('input')
    # fit to unit cube
    unit_scale = 1 / np.max(source.get_max_bound() - source.get_min_bound())
    print('----- unitscale:=', unit_scale) ## for bonsai_old, it's 0.227
    source.scale(unit_scale, center=source.get_center())
    source.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(2000, 3)))

    target.scale(unit_scale, center=target.get_center())
    target.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(2000, 3)))
    o3d.visualization.draw_geometries([source])  ## plotting point cloud
    o3d.visualization.draw_geometries([target])

    print('voxelization')
    voxel_size=0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])
    return source, target, voxel_size

#source, target, voxel_size = unitscale(source, target)
voxel_size = 0.05

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def global_fast_registration(source, target,voxel_size=0.05): # means 5cm for the dataset

    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(voxel_size, source, target):
        print(":: Load two point clouds and disturb initial pose.")

        # demo_icp_pcds = o3d.data.DemoICPPointClouds()
        # source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        # target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

        trans_init = np.identity(4) # np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                #[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)
    
    def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    #start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    #print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)

global_fast_registration(source, target, voxel_size)






#----------------------------------------------------------------

# draw initial alignment
current_transformation = np.identity(4)
# draw_registration_result_original_color(source, target, current_transformation)


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])
    
def point_pln_ICP(source, target): ## no use
    # point to plane ICP
    current_transformation = np.identity(4)
    print("2. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(result_icp)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)


def local_registration(source, target): ## no use
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp, current_transformation)


    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)
# local_registration(source, target) ## no result