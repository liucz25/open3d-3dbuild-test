import open3d as o3d
import numpy as np
import cv2

# 1. 下载并加载Open3D官方示例数据
print("下载示例数据...")
dataset = o3d.data.SampleRedwoodRGBDImages()
color_paths = dataset.color_paths
depth_paths = dataset.depth_paths

print(f"找到 {len(color_paths)} 张颜色图，{len(depth_paths)} 张深度图")

# 2. 验证第一对图像格式
def validate_image_formats(color_path, depth_path):
    # 使用OpenCV加载以检查格式
    color_cv = cv2.imread(color_path)
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    print(f"颜色图: 形状={color_cv.shape}, 类型={color_cv.dtype}")
    print(f"深度图: 形状={depth_cv.shape}, 类型={depth_cv.dtype}")
    
    # 确保格式正确
    if color_cv.dtype != np.uint8:
        print("警告: 颜色图不是uint8格式，需要转换")
        color_cv = color_cv.astype(np.uint8)
    
    if depth_cv.dtype not in [np.uint16, np.float32]:
        print("警告: 深度图格式不正确，转换为uint16")
        depth_cv = (depth_cv * 1000).astype(np.uint16)  # 假设深度单位为米
    
    return color_cv, depth_cv

# 3. 创建TSDF体积
voxel_size = 0.02
sdf_trunc = 0.08

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# 4. 相机内参（使用PrimeSense默认值）
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

# 5. 处理每对图像
for i in range(min(10, len(color_paths))):  # 限制前10帧测试
    print(f"\n处理第 {i+1} 帧...")
    
    try:
        # 验证格式
        color_cv, depth_cv = validate_image_formats(color_paths[i], depth_paths[i])
        
        # 转换为Open3D图像格式
        color_o3d = o3d.geometry.Image(color_cv)
        depth_o3d = o3d.geometry.Image(depth_cv)
        
        # 创建RGBD图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,    # 深度单位为毫米
            depth_trunc=3.0,       # 截断距离3米
            convert_rgb_to_intensity=False
        )
        
        # 简单位姿估计（实际应用中需要更准确的位姿）
        pose = np.eye(4)
        if i > 0:
            # 简单平移
            pose[0, 3] = i * 0.05
        
        # 融合到TSDF体积
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        print(f"第 {i+1} 帧融合成功")
        
    except Exception as e:
        print(f"处理第 {i+1} 帧时出错: {e}")
        print("尝试使用备用方法...")
        
        # 备用方法：直接使用Open3D读取
        try:
            color = o3d.io.read_image(color_paths[i])
            depth = o3d.io.read_image(depth_paths[i])
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth,
                depth_scale=1000.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )
            
            volume.integrate(rgbd, intrinsic, np.eye(4))
            print(f"第 {i+1} 帧备用方法成功")
        except Exception as e2:
            print(f"备用方法也失败: {e2}")

# 6. 提取并保存结果
print("\n提取网格...")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

# 保存结果
output_path = "reconstructed_model.ply"
o3d.io.write_triangle_mesh(output_path, mesh)
print(f"重建完成！模型已保存到: {output_path}")

# 显示结果
o3d.visualization.draw_geometries([mesh], window_name="3D重建结果")