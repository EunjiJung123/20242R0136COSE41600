# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

pcd_folder = '../data/01_straight_walk/pcd/'
pcd_files = sorted(glob.glob(pcd_folder + "*.pcd"))
pcd_files = pcd_files[:10]

# 비디오 저장 설정
video_filename = "../videos/01_straight_walk.avi"
frame_width, frame_height = 1920, 1080  # 영상 해상도
fps = 10  # 초당 프레임
video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

# Open3D 시각화 객체 초기화
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Video", width=frame_width, height=frame_height, visible=True)

# 적절한 카메라 위치 설정
pcd_file1 = pcd_files[0]
pcd = o3d.io.read_point_cloud(pcd_file1)
points = np.asarray(pcd.points)
x_avg = (np.min(points[:, 0]) + np.max(points[:, 0]))/2
y_avg = (np.min(points[:, 1]) + np.max(points[:, 1]))/2
z_avg = (np.min(points[:, 2]) + np.max(points[:, 2]))/2
print(x_avg, y_avg, z_avg)

# 카메라 위치 및 뷰포인트 설정 함수
def set_camera_view(vis, zoom=0.05, front=[0, -1, 0], lookat=[x_avg, y_avg, z_avg], up=[0, 0, 1]):
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)  # 줌 인/아웃 설정
    ctr.set_front(front)  # 카메라가 바라보는 방향
    ctr.set_lookat(lookat)  # 카메라의 중심점
    ctr.set_up(up)  # 카메라 상단 방향

# 루프를 통해 각 PCD 파일 처리
for pcd_file in pcd_files:
    # PCD 파일 읽기
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 기존 코드 활용: 다운샘플링, ROR, RANSAC, 클러스터링 및 필터링 수행
    voxel_size = 0.2
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 클러스터링 후 색상 제거 (검은색 설정)
    final_point.paint_uniform_color([0, 0, 0])  # 모든 점을 검은색으로 설정

    # 바운딩 박스 필터링 (기존 코드 활용)
    # 클러스터 내 최대 최소 포인트 수
    min_points_in_cluster = 5
    max_points_in_cluster = 40

    # 클러스터 내 최소 최대 Z값
    min_z_value = -1.5
    max_z_value = 2.5

    # 클러스터 내 최소 최대 Z값 차이
    min_height = 0.5
    max_height = 2.0

    max_distance = 30.0  # 원점으로부터의 최대 거리

    bboxes_1234 = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]  # Z값 추출
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0)
                        bboxes_1234.append(bbox)

    # 시각화 설정 및 프레임 캡처
    vis.clear_geometries()
    vis.add_geometry(final_point)
    for bbox in bboxes_1234:
        vis.add_geometry(bbox)

    # 카메라 뷰 설정
    set_camera_view(
        vis,
        zoom=0.1,
        front=[0, -1, 0.2],
        lookat=[-0.5, 0, 0.5],
        up=[0, 0, 1]
    )

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

# https://oh2279.tistory.com/204
# https://gaussian37.github.io/autodrive-lidar-open3d/