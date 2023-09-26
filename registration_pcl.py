
import pcl

# Load point clouds
source_cloud = pcl.load("source_cloud.pcd")
target_cloud = pcl.load("target_cloud.pcd")

# Create the ICP object
icp = source_cloud.make_IterativeClosestPoint()

# Set the parameters
icp.setTransformationEpsilon(1e-6)
icp.setMaxCorrespondenceDistance(0.1)
icp.setMaximumIterations(100)

# Register the source cloud to the target cloud
icp.align(source_cloud, target_cloud)

# Get the transformation
transformation = icp.getFinalTransformation()
