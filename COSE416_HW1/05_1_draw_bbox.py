# practice code
# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "test_data/1727320101-665925967.pcd"
# file_path = "../data/01_straight_walk\pcd\pcd_000001.pcd"
# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=10, radius=1.0)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링 적용
def apply_hdbscan_clustering(point_cloud, min_cluster_size=8, min_samples=10, cluster_selection_epsilon=0.3):
    # 포인트 클라우드의 좌표 추출
    points = np.asarray(point_cloud.points)

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
    labels = clusterer.fit_predict(points)

    return labels


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(apply_hdbscan_clustering(final_point, min_cluster_size=8, min_samples=5))

##########################################################################################두가지 선택지 (색칠 O, X)
##1
# 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
# colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
# colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정
#
# final_point.colors = o3d.utility.Vector3dVector(colors)

##2
# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
#########################################################################################################

# 필터링 기준 설정
min_points_in_cluster = 1   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 100  # 클러스터 내 최대 포인트 수
min_z_value = -1.5          # 클러스터 내 최소 Z값의 범위
max_z_value = 3.0           # 클러스터 내 최대 Z값
width_x = 2.0               # 클러스터 내 X 값의 범위 (너비)
width_y = 2.0               # 클러스터 내 Y 값의 범위 (너비)
min_height = 1.0            # Z값 차이의 최소값
max_height = 2.5            # Z값 차이의 최대값
max_distance = 200.0         # 원점으로부터의 최대 거리

# 주어진 필터링 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_1234 = []
for i in range(labels.max() + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]
        z_min = z_values.min()
        z_max = z_values.max()
        x_diff = max(points[:, 0]) - min(points[:, 0])
        y_diff = max(points[:, 1]) - min(points[:, 1])
        if min_z_value <= z_min <= 1.5 and 0 <= z_max <= max_z_value:
            height_diff = z_max - z_min
            if min_height <= height_diff <= max_height and 0 <= x_diff <= width_x and 0 <= y_diff <= width_y and 0 <= height_diff/x_diff <= 10:
                distances = np.linalg.norm(points, axis=1)
                if distances.max() <= max_distance:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0)
                    bboxes_1234.append(bbox)

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
print(f"Number of detected pedestrian clusters: {len(bboxes_1234)}")
visualize_with_bounding_boxes(final_point, bboxes_1234, point_size=2.0)

# for feature in features:
#     print(feature)
