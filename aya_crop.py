import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor

def point_exists(point, cropped_points, threshold=1e-6):
    """Check if a point exists in the cropped_points within a threshold."""
    distances = np.linalg.norm(cropped_points - point, axis=1)
    return np.any(distances < threshold)

def process_chunk(chunk, cropped_points):
    return [point for point in chunk if point_exists(point[:3], cropped_points)]

def main():
    # Load the PLY files using plyfile
    full_ply = PlyData.read(r"c:\Users\aya\git\gaussian-splatting\output\coral_colorized_cropped\point_cloud\iteration_30000\point_cloud_full.ply")
    cropped_pcd = o3d.io.read_point_cloud(r"c:\Users\aya\git\gaussian-splatting\output\coral_colorized_cropped\point_cloud\iteration_30000\point_cloud.ply")

    cropped_points = np.asarray(cropped_pcd.points)
    full_points = np.array([list(point) for point in full_ply['vertex'].data])

    # Split full_points into chunks for parallel processing
    num_workers = 32  # Adjust this based on your CPU's cores
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
    filtered_ply.write("filtered.ply")

    # Optionally, delete the temporary file after successful completion
    # os.remove(temp_filename)

    print("Processing complete!")

if __name__ == "__main__":
    main()
