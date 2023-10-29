import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor

def point_exists(point, cropped_points, threshold=1e-6):
    """Check if a point exists in the cropped_points within a threshold."""
    distances = np.linalg.norm(cropped_points - point, axis=1)
    return np.any(distances < threshold)

def process_chunk(chunk, cropped_points):
    "extract the points in chunk (full points)"
    return [point for point in chunk if point_exists(point[:3], cropped_points)]

def main(full_path, cut_name):
    # Load the PLY files using plyfile
    new_path = cut_name + r'.ply'
    full_ply = PlyData.read(full_path)
    full_points = np.array([list(point) for point in full_ply['vertex'].data])

    cropped_pcd = o3d.io.read_point_cloud(new_path)
    cropped_points = np.asarray(cropped_pcd.points)
    

    # Split full_points into chunks for parallel processing
    num_workers = 60  # Adjust this based on your CPU's cores
    chunk_size = len(full_points) // num_workers
    chunks = [full_points[i:i + chunk_size] for i in range(0, len(full_points), chunk_size)]

    # Process each chunk concurrently
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_chunk, chunks, [cropped_points] * len(chunks)))

    # Flatten the results
    filtered_data = [tuple(point) for sublist in results for point in sublist]

    # Convert the filtered data to a structured numpy array for saving
    filtered_data_array = np.array(filtered_data, dtype=full_ply['vertex'].dtype())

    # Save intermediate results to a temporary numpy binary file
    temp_filename = "temp_filtered_data.npy"
    np.save(temp_filename, filtered_data_array)
    print(f"Intermediate results saved to {temp_filename}")

    # Create a new PLY data structure with the filtered points
    filtered_ply = PlyData([PlyElement.describe(filtered_data_array, 'vertex')])

    # Save the final point cloud
    filtered_ply.write(cut_name + "_filtered.ply") ##Hui: slow need time, saved in gaussian-splattin folder

    # Optionally, delete the temporary file after successful completion
    # os.remove(temp_filename)

    print("Processing complete!")

if __name__ == "__main__":

    import time

    start = time.time()

    "produced extracted base Gaussian-Splatting data of the full_ply"
    path = r'C:\Users\WANGH0M\gaussian-splatting\output'
    full_path = path + r'\out_bonsai_new\point_cloud\iteration_30000\point_cloud.ply'
    cut_name = path + r'\bonsai_new_cut'
    main(full_path, cut_name)

    "read the filtered.ply"
    filtered_ply = PlyData.read(cut_name + "_filtered.ply")
    print(filtered_ply) 
    ##Hui: which shows property [x,y,z], [nx,ny,nz], [f_dc_0,1,2], [f_rest_0,...,44], 
    ##                          opacity, [scale_0,1,2], [rot_0,1,2,3]

    end = time.time()
    print('Total running time: = ', end - start)