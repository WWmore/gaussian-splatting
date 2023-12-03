
from plyfile import PlyData
import numpy as np
import laspy
import open3d as o3d
import igl

from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary

path = r'C:\Users\WANGH0M\gaussian-splatting\data_Dji_L2\sparse\0'

camera_path = path + r'\cameras.bin'
image_path = path + r'\images.bin'
point_path = path + r'\points3D.bin'

def read_colmap_output(camera_path, image_path, point_path):
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
    ##Hui: which shows property 278817 points with [x,y,z], [nx,ny,nz], [red, green, blue]

#------------------------------------------------------------------------------------------------

def read_lidar(path):
    #path1 = r'C:\Users\WANGH0M\Desktop\LiDAR\Zenmuse L1 202107 Rio Brazos.las' ##(40824215, 3)
    input_las = laspy.read(path)
    point_records = input_las.points.copy()

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

    xyz = np.vstack((p_X, p_Y, p_Z)).transpose()
    print('Lidar points shape = ',xyz.shape)

    return xyz

def read_lidar2(path):
    input_las = laspy.read(path)
    point_records = input_las.points.copy()

    # calculating coordinates
    p_X = np.array(point_records['X'])
    p_Y = np.array(point_records['Y'])
    p_Z = np.array(point_records['Z'])

    xyz = np.vstack((p_X, p_Y, p_Z)).transpose()
    return xyz

def read_ply(path):
    pcd = o3d.io.read_point_cloud(path) # Read the point cloud

    # Visualize the point cloud within open3d
    #o3d.visualization.draw_geometries([pcd]) 

    # Convert open3d format to numpy array
    point_cloud_in_numpy = np.asarray(pcd.points) 
    return point_cloud_in_numpy

def write_ply(point, name, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    o3d.io.write_point_cloud(path+name, pcd)

def plot_pointcloud(point):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    o3d.visualization.draw_geometries([pcd])

def plot_multi_pointclouds(point1, point2):
    if True:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(point2)
        o3d.visualization.draw_geometries([pcd1, pcd2])
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # geometry is the point cloud used in your animaiton
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)

        if 0:
            for i in range(len(pcd_list)):
                # now modify the points of your geometry
                # you can use whatever method suits you best, this is just an example
                geometry.points = pcd_list[i].points
                vis.update_geometry(geometry)
                vis.poll_events()
                vis.update_renderer()
        else:
            geometry.points = pcd1
            vis.update_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()

            geometry.points = pcd2
            vis.update_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()

def get_pointcloud_boundingbox_open3d(point1, point2): ##no use, since there is an error which cannot be solved
    "scale by bounding box; translation by barycenters" 
    "point1 is from denser lidar pointcloud"
    #point1 *= 0.01
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point1)
    box1 = pcd1.get_oriented_bounding_box() ##show RuntimeError: QH6235 qhull error (qh_memalloc): negative request size (-2097038872)
    box1.color = (0,0,1)
    print(box1)
    o3d.visualization.draw_geometries([pcd1, box1])

    centroid1 = np.array(box1.center)
    x1_max, y1_max, z1_max = box1.extent
    d1 = np.sqrt(x1_max**2+y1_max**2+z1_max**2)

    "point2 is from pointcloud by colmap"
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point2)
    box2 = pcd2.get_oriented_bounding_box() ## show boundingbox result
    box2.color = (1,0,0)
    print(box2)
    o3d.visualization.draw_geometries([pcd2, box2])

    centroid2 = np.array(box2.center)
    x2_max, y2_max, z2_max = box2.extent
    d2 = np.sqrt(x2_max**2+y2_max**2+z2_max**2)
    return [centroid1, d1], [centroid2, d2]

def get_pointcloud_boundingbox(point):
    corner = igl.bounding_box(point)[0]
    centroid = np.mean(corner, axis=0)
    d1 = np.linalg.norm(corner[0]-corner[6])
    # d2 = np.linalg.norm(corner[1]-corner[7]) ##same as d1
    # d3 = np.linalg.norm(corner[2]-corner[4])
    # d4 = np.linalg.norm(corner[3]-corner[5])
    return centroid, d1

def rescale_two_pointclouds(point1, point2):
    _, d1 = get_pointcloud_boundingbox(point1)
    centroid2, d2 = get_pointcloud_boundingbox(point2)
    point1 *= d2 / d1

    corner1 = igl.bounding_box(point1)[0]
    centroid1 = np.mean(corner1, axis=0)
    point1 += centroid2 - centroid1

    plot_multi_pointclouds(point1, point2)

    return point1, point2

def write_lidar_point(point):
    pcd = o3d.geometry.PointCloud()

    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    pcd.points = o3d.utility.Vector3dVector(point)

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=True)

    # read ply file
    # pcd = o3d.io.read_point_cloud('my_pts.ply')


# def register_colmappoint_with_lidarpoint(lidar_path, colmap_pcd):
#     """ find closest_index in PC2 such that PC1 ~~ PC2[closest_index]
#     PC1: colored_pc
#     PC2: lidar_pc
#     return: closest_index
#     """
#     #lidar = PlyData.read(lidar_path)

#     input_las = laspy.read(lidar_path)
#     point_records = input_las.points.copy()
#     # calculating coordinates
#     p_X = np.array((point_records['X'] * las_scaleX) + las_offsetX)
#     p_Y = np.array((point_records['Y'] * las_scaleY) + las_offsetY)
#     p_Z = np.array((point_records['Z'] * las_scaleZ) + las_offsetZ)

#     xyz = np.vstack((p_X, p_Y, p_Z)).transpose()
#     print(xyz.shape)


#     PC1 = PC2[index]

#     return PC1


def draw_registration_result(source, target, transformation):
    import copy
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


if __name__ == "__main__":

    lidar_path = r'C:\Users\WANGH0M\Documents\DJI\DJITerra\hui.wang.1@kaust.edu.sa\L2_Hui\lidars\terra_las\cloud_merged.las'
    lidar_ply_path = r'C:\Users\WANGH0M\Documents\DJI\DJITerra\hui.wang.1@kaust.edu.sa\L2_Hui\lidars\terra_point_ply\cloud_merged.ply'
    colmap_ply_path = r'C:\Users\WANGH0M\gaussian-splatting\data_Dji_L2\sparse\0\points3D.ply'

    if 0:
        point1 = read_lidar(lidar_path) ## or point = read_lidar2(lidar_path)
        ## print(point1) ##PointCloud with 274741051 points
        #plot_pointcloud(point1)
    else:
        "above way is much slower and still has the RuntimeError: QH6235 qhull error (qh_memalloc)"
        point1 = read_ply(lidar_ply_path)

    point2 = read_ply(colmap_ply_path)

    "Step0: plot"
    #plot_multi_pointclouds(point1, point2)

    "Step1: rescale lidar_points to be compatible with points3D"
    #point1,_ = rescale_two_pointclouds(point1, point2) ## comment for global_fast_registration

    "Step2: save the rescaled lidar_points"
    path = r'C:\Users\WANGH0M\gaussian-splatting\data_Dji_L2\sparse\0'
    name = '\lidar3D.ply'
    #write_ply(point1, name, path) ## comment for global_fast_registration

    "Step3: registration the rescaled lidar_point and points3D"
    full_path = path + r'\points3D.ply'
    new_path = path + r'\lidar3D.ply'
    source = o3d.io.read_point_cloud(full_path)
    target = o3d.io.read_point_cloud(new_path)
    global_fast_registration(source, target, voxel_size=0.05)