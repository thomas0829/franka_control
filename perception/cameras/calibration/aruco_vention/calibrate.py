import json
from datetime import datetime
from perception.cameras.multi_camera_wrapper import MultiCameraWrapper

multi_camera_wrapper = MultiCameraWrapper(type="realsense", use_threads=False)

num_cameras = multi_camera_wrapper.num_cameras

print(f"Number of cameras: {num_cameras}")

from utils.pointclouds import *

# calibrate using aruco tag
pcds = []
calib_dict = []
for camera in multi_camera_wrapper._all_cameras:
    marker_size = 0.15
    tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
        camera, marker_size=marker_size, verbose=True
    )

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotmat
    extrinsics[:3, 3:] = tvec

    frames = camera.read_camera()
    rgb = frames[0]["array"]
    depth = frames[1]["array"]

    sn = camera.read_camera()[0]["serial_number"]
    intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]

    # camera frame -> aruco frame
    extrinsics_inv = np.linalg.inv(extrinsics)
    # aruco frame to center robot frame
    
    # right to the robot
    # x: marker size / 2 + white space on paper
    # y: table size / 2 - marker size / 2
    # aruco_offset = np.array([- (marker_size / 2), - 0.675 + (marker_size / 2) + 0.05, 0.0])
    margin = 0.0  # 0.033 + 0.02
    marker_radius = marker_size / 2
    table_length = 1.350
    table_width = 0.809
    franka_center = 0.17
    aruco_offset = np.array(
        [
            -(franka_center - (marker_radius + margin)),
            -(table_length / 2 - (marker_radius + margin)),
            0.0,
        ]
    )

    # middle front of the table
    aruco_offset = np.array(
        [
            0.55,
            0.0,
            0.0,
        ]
    )

    extrinsics_inv[:3, 3] += aruco_offset

    calib_dict.append(
        {
            "camera_serial_number": sn,
            "intrinsics": {
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "ppx": intrinsics[0, 2],
                "ppy": intrinsics[1, 2],
                "height": rgb.shape[0],
                "width": rgb.shape[1],
                "fovy": camera._fovy,
                "coeffs": camera._intrinsics[sn.replace("/", "_")][
                    "distCoeffs"
                ].tolist(),
            },
            "camera_base_ori": extrinsics_inv[:3, :3].tolist(),
            "camera_base_pos": extrinsics_inv[:3, 3:].tolist(),
        }
    )

    depth[depth == np.inf] = 1.0  # depth[~(depth==np.inf)].max()
    depth[depth == -np.inf] = 1.0  # depth[~(depth==np.inf)].max()
    depth[np.isnan(depth)] = 1.0
    points = depth_to_points(depth, intrinsics, extrinsics_inv, depth_scale=1000.0)
    points[np.isnan(points)] = 1.0
    points[points == np.inf] = 1.0
    points[points == -np.inf] = 1.0

    colors = rgb.reshape(-1, 3) / 255.0
    points, colors = crop_points(points, colors=colors)
    pcds.append(points_to_pcd(points, colors=colors))

current_time_date = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
json.dump(
    calib_dict,
    open(f"perception/cameras/calibration/logs/aruco/{current_time_date}.json", "w"),
)
json.dump(
    calib_dict,
    open(f"perception/cameras/calibration/logs/aruco/most_recent_calib.json", "w"),
)

x = np.zeros((1, 3))
for d in np.arange(0, 1, 0.1):
    x[:, 0] = d
    pcds.append(points_to_pcd(x, colors=[[255.0, 0.0, 0.0]]))
    y = np.zeros((1, 3))
    y[:, 1] = d
    pcds.append(points_to_pcd(y, colors=[[0.0, 255.0, 0.0]]))
    z = np.zeros((1, 3))
    z[:, 2] = d
    pcds.append(points_to_pcd(z, colors=[[0.0, 0.0, 255.0]]))


visualize_pcds(pcds)
