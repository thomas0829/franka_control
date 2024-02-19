import time

import cv2

from perception.cameras.multi_camera_wrapper import MultiCameraWrapper

multi_camera_wrapper = MultiCameraWrapper(type="realsense", use_threads=False)

num_cameras = multi_camera_wrapper.num_cameras

print(f"Number of cameras: {num_cameras}")

# from helpers.pointclouds import *

# # calibrate using aruco tag
# pcds = []
# for camera in multi_camera_wrapper._all_cameras:
#     tvec, rotmat = multi_camera_wrapper._get_aruco_pose(camera)

#     extrinsics = np.eye(4)
#     extrinsics[:3, :3] = rotmat
#     extrinsics[:3, 3:] = tvec

#     frames = camera.read_camera()
#     rgb = frames[0]["array"]
#     depth = frames[1]["array"]

#     sn = camera.read_camera()[0]["serial_number"]
#     intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]

#     depth[depth==np.inf] = 1. # depth[~(depth==np.inf)].max()
#     points = depth_to_points(depth, intrinsics, np.linalg.inv(extrinsics), depth_scale=1000.)
#     points[np.isnan(points)] = 1.
#     points[points==np.inf] = 1.

#     colors = rgb.reshape(-1,3)/255.
#     points, colors = crop_points(points, colors=colors, crop_min=-np.ones(3), crop_max=np.ones(3))
#     pcds.append(points_to_pcd(points, colors=colors))

# pcds.append(points_to_pcd(np.zeros((1,3)), colors=[[0., 0., 255.]]))

# visualize_pcds(pcds)

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
