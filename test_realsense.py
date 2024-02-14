import time
import cv2

from cameras.multi_camera_wrapper import MultiCameraWrapper

multi_camera_wrapper = MultiCameraWrapper(type="realsense", use_threads=False)

num_cameras = multi_camera_wrapper.num_cameras

multi_camera_wrapper.get_aruco_poses()

print(f"Number of cameras: {num_cameras}")

from helpers.pointclouds import *

# calibrate using aruco tag
pcds = []
calib_dict = []
for camera in multi_camera_wrapper._all_cameras:
    marker_size = 0.15
    tvec, rotmat = multi_camera_wrapper._get_aruco_pose(camera, marker_size=marker_size)

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotmat
    extrinsics[:3, 3:] = tvec

    frames = camera.read_camera()
    rgb = frames[0]["array"]
    depth = frames[1]["array"]

    sn = camera.read_camera()[0]["serial_number"]
    intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]

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
                "coeffs": camera._intrinsics[sn.replace("/", "_")]["distCoeffs"],
            },
            "camera_base_ori": np.linalg.inv(extrinsics)[:3, :3].tolist(),
            "camera_base_pos": np.linalg.inv(extrinsics)[:3, 3:].tolist(),
        }
    )

    depth[depth == np.inf] = 1.0  # depth[~(depth==np.inf)].max()
    points = depth_to_points(
        depth, intrinsics, np.linalg.inv(extrinsics), depth_scale=1000.0
    )
    points[np.isnan(points)] = 1.0
    points[points == np.inf] = 1.0

    colors = rgb.reshape(-1, 3) / 255.0
    points, colors = crop_points(
        points, colors=colors, crop_min=-np.ones(3), crop_max=np.ones(3)
    )
    pcds.append(points_to_pcd(points, colors=colors))


import json
json.dump(calib_dict, open("cameras/calibration/calibration_aruco_real.json", "w"))

# fixed translatation frame from aruco to robot on vention table
# - table width / 2 + marker size / 2
tvec = np.array([[-0.675 + marker_size / 2, 0.0, 0.0]])
zero = tvec @ np.eye(3)
pcds.append(points_to_pcd(zero, colors=[[0.0, 0.0, 255.0]]))

visualize_pcds(pcds)


while True:
    frames = multi_camera_wrapper.read_cameras()
    for i in range(0, len(frames), 2):
        rgb = frames[i]["array"]
        depth = frames[i + 1]["array"]
        cv2.imshow(f"RGB_cam_{i//2}", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow(
            f"Depth_cam_{i//2}",
            cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET),
        )
        k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):
        break
    else:
        pass
