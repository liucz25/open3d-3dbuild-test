import open3d as o3d
print("Load a ply point cloud, print it, and render it")
file="./t1.ply"
file="./pointcloud/pointcloud_1.ply"
# sample_ply_data = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(file)
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([sample_ply_data])
# file2="./demo_rock/depth/depth_18.png"
file2="./cv_depth.png"

# sample_ply_data = o3d.data.PLYPointCloud()
pcd = o3d.io.read_image(file2)
o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([sample_ply_data])