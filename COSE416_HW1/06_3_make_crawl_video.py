# video로 나타낸 data 중 '03_straight_crawl'은 이 코드로 수행
# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import hdbscan

file_names = ['01_straight_walk','02_straight_duck_walk','03_straight_crawl','04_zigzag_walk','05_straight_duck_walk','06_straight_crawl','07_straight_walk']

# 사용할 folder에 따라 pcd_folder 값 바꾸기
pcd_folder = '../data/03_straight_crawl/pcd/'
pcd_files = sorted(glob.glob(pcd_folder + "*.pcd"))
# pcd_files = pcd_files[-30:] # test code

# 비디오 저장 설정
video_filename = "../videos/03_straight_crawl.avi"
frame_width, frame_height = 1920, 1080  # 영상 해상도
fps = 10  # 초당 프레임
video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

# Open3D 시각화 객체 초기화
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Video", width=frame_width, height=frame_height, visible=True)

# HDBSCAN 클러스터링 함수
def apply_hdbscan_clustering(point_cloud, min_cluster_size=11, min_samples=None):
    # 포인트 클라우드의 좌표 추출
    points = np.asarray(point_cloud.points)

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(points)

    return labels

# 카메라 위치 및 뷰포인트 설정 함수
def set_camera_view(vis, zoom=0.1, front=[0, -1, 0.1], lookat=[0, 20, 0], up=[0, 0, 1]):
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)  # 줌 인/아웃 설정
    ctr.set_front(front)  # 카메라가 바라보는 방향
    ctr.set_lookat(lookat)  # 카메라의 중심점
    ctr.set_up(up)  # 카메라 상단 방향

# 클러스터 중심 좌표 저장용 딕셔너리
cluster_centroids_history = {}

# 이동 거리 기준 (클러스터가 사람인지 판단할 때 사용)
movement_threshold = 0.3  # 클러스터가 움직였다고 판단하는 최소 이동 거리

# 루프를 통해 각 PCD 파일 처리
for frame_idx, pcd_file in enumerate(pcd_files):
    # PCD 파일 읽기
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 기존 코드 활용: 다운샘플링, ROR, RANSAC, 클러스터링 및 필터링 수행
    voxel_size = 0.1
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=5, radius=1.5)
    ror_pcd = downsample_pcd.select_by_index(ind)
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # HDBSCAN을 사용해 클러스터링 진행
    labels = np.array(apply_hdbscan_clustering(final_point, min_cluster_size=5, min_samples=3))

    # 각 클러스터를 색으로 표시
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # 노이즈를 제거하고 각 클러스터에 색상 지정
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 클러스터 중심 좌표 계산
    current_centroids = {}
    for cluster_id in range(labels.max() + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            centroid = np.mean(points, axis=0)  # 클러스터의 중심 좌표 계산
            current_centroids[cluster_id] = centroid

    # 클러스터 병합 로직
    cluster_distances = np.linalg.norm(
        np.array(list(current_centroids.values()))[:, None, :] -
        np.array(list(current_centroids.values()))[None, :, :],
        axis=-1
    )
    merge_threshold = 2.0  # 병합 거리 임계값
    merged_clusters = {id: id for id in current_centroids.keys()}

    # 병합 매핑 생성
    for i, id1 in enumerate(current_centroids.keys()):
        for j, id2 in enumerate(current_centroids.keys()):
            if cluster_distances[i, j] < merge_threshold and id1 != id2:
                merged_clusters[id2] = merged_clusters[id1]

    # 병합된 중심 좌표 갱신
    merged_centroids = {}
    for original_id, merged_id in merged_clusters.items():
        if merged_id not in merged_centroids:
            merged_centroids[merged_id] = []
        merged_centroids[merged_id].append(current_centroids[original_id])

    # 병합된 중심 좌표 계산 (평균값)
    final_centroids = {id: np.mean(points, axis=0) for id, points in merged_centroids.items()}
    current_centroids = final_centroids

    # 이전 프레임과 비교하여 움직이는 클러스터 탐지 및 필터링
    moving_clusters = []
    for cluster_id, centroid in current_centroids.items():
        if cluster_id in cluster_centroids_history:
            previous_centroid = cluster_centroids_history[cluster_id]
            movement = np.linalg.norm(centroid - previous_centroid)  # 이동 거리 계산

            # 조건 1: 이동 거리 기준 초과
            if movement > movement_threshold:
                # 클러스터 포인트 및 바운딩 박스 가져오기
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_pcd = final_point.select_by_index(cluster_indices)
                bbox = cluster_pcd.get_axis_aligned_bounding_box()

                points = np.asarray(cluster_pcd.points)
                distances = np.linalg.norm(points, axis=1)
                z_max = max(points[:,2])
                z_min = min(points[:, 2])

                # 클러스터 포인트 수 제한
                if not (5 <= len(cluster_indices) <= 300): #or not (distances.max() <= 100.0):
                    continue

                # 조건 2: 사람의 위치, 크기 필터링
                bbox_extent = bbox.get_extent()  # (너비, 높이, 깊이)
                width, height, depth = bbox_extent
                if not (-1.5 <= z_min < 1.0) :
                    continue
                if not (z_max < 3.0):
                    continue
                z_diff = z_max - z_min
                if not (0.4 < width < 1.5) or not (0.0 < z_diff < 2.5):
                    continue  # 크기가 사람 범위를 벗어남

                # 조건을 만족한 클러스터만 추가
                moving_clusters.append(cluster_id)

    # 클러스터 중심 좌표 업데이트
    cluster_centroids_history = current_centroids

    # 움직이는 클러스터만 바운딩 박스 생성
    bboxes_1234 = []
    for cluster_id in moving_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_pcd = final_point.select_by_index(cluster_indices)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # 빨간색 박스
        bboxes_1234.append(bbox)

    # 시각화 및 영상 저장
    vis.clear_geometries()
    vis.add_geometry(final_point)
    for bbox in bboxes_1234:
        vis.add_geometry(bbox)

    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.asarray([0, 0, 0])  # 검은색 배경 설정 (선택 사항)

    # 카메라 뷰 설정
    set_camera_view(vis)

    vis.poll_events()
    vis.update_renderer()

    # 화면 캡처
    image = vis.capture_screen_float_buffer(False)
    frame = np.asarray(image) * 255.0
    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

# 종료 작업
video_writer.release()
vis.destroy_window()

print("비디오 생성이 완료되었습니다:", video_filename)