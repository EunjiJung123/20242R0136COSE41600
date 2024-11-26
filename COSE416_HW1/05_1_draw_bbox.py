# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "test_data/1727320101-665925967.pcd"
# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=15, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링 적용
def apply_hdbscan_clustering(point_cloud, min_cluster_size=11, min_samples=None):
    # 포인트 클라우드의 좌표 추출
    points = np.asarray(point_cloud.points)

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(points)

    return labels


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(apply_hdbscan_clustering(final_point, min_cluster_size=11))
##################################

##########################################################################################두가지 선택지 (색칠 O, X)
##1
# # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
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

# 필터링 기준 설정 (x나 y 값 차이의 최소, 최댓값에 대한 설정도 넣기, 밀도에 대한 설정도)
features = []
for cluster_id in range(max(labels) + 1):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_pcd = final_point.select_by_index(cluster_indices)
    points = np.asarray(cluster_pcd.points)

    points_in_cluster = len(cluster_indices)
    z_min = min(points[:, 2])
    z_max = max(points[:, 2])

    bbox = cluster_pcd.get_axis_aligned_bounding_box()
    width, height, depth = bbox.get_extent()
    density = len(points) / (width * height * depth)
    # centroid = bbox.get_center()
    max_distance = max(np.linalg.norm(points, axis=1))

    features.append({
        "cluster_id": cluster_id,
        "width": width,
        "height": height,
        "depth": depth,
        "density": density,
        "z_min": z_min,
        "z_max": z_max,
        "max_distance": max_distance,
        "points_in_cluster": points_in_cluster
        # "centroid": centroid
    })

min_points_in_cluster = 1   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 150  # 클러스터 내 최대 포인트 수
min_z_value = -1.5          # 클러스터 내 최소 Z값
max_z_value = 2.5           # 클러스터 내 최대 Z값
min_height = 0.5            # Z값 차이의 최소값
max_height = 2.0            # Z값 차이의 최대값
max_distance = 30.0         # 원점으로부터의 최대 거리

# 보행자 후보 클러스터 필터링 및 바운딩 박스 생성
pedestrian_clusters = []
for feature in features:
    # 거리 기반 크기 조정
    min_height = max(0.5, 1.0 - max_distance * 0.01)
    max_height = min(2.5, 2.0 + max_distance * 0.005)

    min_width = max(0.2, 0.5 - max_distance * 0.05)
    max_width = min(1.2, 0.8 + max_distance * 0.05)

    # 거리 기반 밀도 조정
    min_density = max(0.05, 0.1 + 0.01 * feature["max_distance"])  # 거리와 비례
    max_density = min(100.0, 10.0 + 0.05 * feature["max_distance"])  # 최대 밀도 확장

    if min_height <= feature["width"] <= max_height and \
       + min_width <= feature["height"] <= max_width and \
       + min_density <= feature["density"] <= max_density and \
       + min_z_value <= feature["z_min"] and \
       + max_z_value >= feature["z_max"] and \
       + min_points_in_cluster <= feature["points_in_cluster"] <= max_points_in_cluster:
        pedestrian_clusters.append(feature["cluster_id"])

pedestrian_bboxes = []
for cluster_id in pedestrian_clusters:
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_pcd = final_point.select_by_index(cluster_indices)
    bbox = cluster_pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)
    pedestrian_bboxes.append(bbox)

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
print(f"Number of detected pedestrian clusters: {len(pedestrian_clusters)}")
visualize_with_bounding_boxes(final_point, pedestrian_bboxes, point_size=2.0)

# for feature in features:
#     print(feature)
print()
