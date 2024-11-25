# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# https://gaussian37.github.io/autodrive-lidar-intro/#ransac%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%8F%84%EB%A1%9C%EC%99%80-%EA%B0%9D%EC%B2%B4-%EA%B5%AC%EB%B6%84-1

# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "test_data/1727320101-665925967.pcd"
# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.1  # 필요에 따라 voxel 크기를 조정
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)
print(f"Point cloud size after ROR: {len(ror_pcd.points)}")

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출 - 3번 코드까지의 부분
final_point = ror_pcd.select_by_index(inliers, invert=True)


# HDBSCAN 클러스터링 적용 - (같은 물체끼리 묶기)
def apply_hdbscan_clustering(point_cloud, min_cluster_size=11, min_samples=None):
    # 포인트 클라우드의 좌표 추출
    points = np.asarray(point_cloud.points)

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(points)

    return labels


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(apply_hdbscan_clustering(final_point, min_cluster_size=11))

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])


# 포인트 클라우드 시각화 함수
def visualize_point_cloud_with_point_size(pcd, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_point_cloud_with_point_size(final_point,
                                      window_name="HDBSCAN Clustered Points", point_size=2.0)
