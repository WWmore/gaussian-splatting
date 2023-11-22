
#----------------------------------------------------------------------------------------------
"""
https://geoai.gishub.org/examples/dataviz/lidar_viz/

pip install "leafmap[lidar]" open3d
pip install leafmap ##problem, anaconda prompt shows successfully satisfied, but ModuleNotFoundError: No module named 'leafmap'
"""

import leafmap
url = 'https://open.gishub.org/data/lidar/madison.zip'
filename = 'madison.las'
leafmap.download_file(url, 'madison.zip', unzip=True)
las = leafmap.read_lidar(filename)
las.header
las.header.point_count
list(las.point_format.dimension_names)
print(las.X, las.Y, las.Z, las.intensity)

"Visualize LiDAR data using the pyvista backend."
leafmap.view_lidar(filename, cmap='terrain', backend='pyvista')

"Visualize LiDAR data using the ipygany backend."
leafmap.view_lidar(filename, backend='ipygany', background='white')

"Visualize LiDAR data using the panel backend."
leafmap.view_lidar(filename, cmap='terrain', backend='panel', background='white')

"Visualize LiDAR data using the open3d backend."
leafmap.view_lidar(filename, backend='open3d')