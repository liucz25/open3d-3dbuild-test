import open3d as o3d
import numpy as np
import glob
import cv2

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

# 使用Open3D自带的RGBD数据集测试
def test_with_sample_dataset():
    # 下载示例数据
    dataset = o3d.data.SampleRedwoodRGBDImages()
    
    # 读取彩色和深度图像
    color_paths = dataset.color_paths
    depth_paths = dataset.depth_paths
    
    # 相机内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    
    # 创建TSDF体积
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for i in range(len(color_paths)):
        print(f"处理第 {i+1}/{len(color_paths)} 帧")
        color_cv, depth_cv = validate_image_formats(color_paths[i], depth_paths[i])
        
        # 使用Open3D直接读取
        color = o3d.geometry.Image(color_cv)
        depth = o3d.geometry.Image(depth_cv)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=3.0,convert_rgb_to_intensity=False
        )
        print(rgbd)
        # 简单位姿
        pose = np.eye(4)
        if i > 0:
            pose[0, 3] = i * 0.05
        
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    
    # 提取并显示结果
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("reconstructed.ply", mesh)

# test_with_sample_dataset()


# dataset = o3d.data.SampleRedwoodRGBDImages()
    
#     # 读取彩色和深度图像
# color_paths = dataset.color_paths
# depth_paths = dataset.depth_paths
# 读取多对RGB-D图像
# color_paths = sorted(glob.glob("./dr/rgb1-1/*.png"))
# depth_paths = sorted(glob.glob("./dr/depth1-1/*.png"))
color_paths = sorted(glob.glob("./box/rgb2/*.png"))
depth_paths = sorted(glob.glob("./box/depth2/*.png"))
# print(color_paths)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)
# use_width =1280
# use_height=720
# # 创建正确的相机内参
# # 方法1: 使用PrimeSense相机参数，但调整到实际尺寸
# # 需要知道实际相机的内参，以下是示例值
# fx = 525.0  # 焦距x
# fy = 525.0  # 焦距y
# cx = use_width / 2.0  # 主点x
# cy = use_height / 2.0  # 主点y

# # RealSense D435/D455在1280×720分辨率下的典型内参
# use_width = 1280
# use_height = 720
# fx = 912.16  # X方向焦距
# fy = 911.27  # Y方向焦距
# cx = 634.52  # 主点X坐标
# cy = 365.33  # 主点Y坐标


# # 创建相机内参对象
# intrinsic = o3d.camera.PinholeCameraIntrinsic(
#     use_width,  # 图像宽度
#     use_height,  # 图像高度
#     fx,  # 焦距x
#     fy,  # 焦距y
#     cx,  # 主点x
#     cy   # 主点y
# )

# 创建TSDF体积进行融合
voxel_size = 0.02  # 体素大小（米）
sdf_trunc = 0.08   # SDF截断距离

tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# 逐帧处理
T_global = np.eye(4)  # 初始位姿

for i in range(int(len(color_paths)/1)):

    validate_image_formats(color_paths[i], depth_paths[i])
    # 创建RGBD图像
    color = o3d.io.read_image(color_paths[i])
    depth = o3d.io.read_image(depth_paths[i])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000.0, depth_trunc=3.0,convert_rgb_to_intensity=False
    )
    
    # 计算里程计（第一帧除外）
    if i > 0:
        success, T_step, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_prev, rgbd, intrinsic, np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        )
        if success:
            T_global = T_global @ T_step

    print(rgbd, intrinsic)#, np.linalg.inv(T_global))
    # 融合到TSDF体积
    tsdf.integrate(rgbd, intrinsic, np.linalg.inv(T_global))
    rgbd_prev = rgbd

# 提取网格
mesh = tsdf.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
# 保存模型
o3d.io.write_triangle_mesh("reconstructed_model.ply", mesh)