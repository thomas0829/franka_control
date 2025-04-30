import json
from datetime import datetime
from perception.cameras.multi_camera_wrapper import MultiCameraWrapper
import pickle

multi_camera_wrapper = MultiCameraWrapper(type="realsense", use_threads=False)

num_cameras = multi_camera_wrapper.num_cameras

print(f"Number of cameras: {num_cameras}")

from utils.pointclouds import *
import cv2

aruco_dict = cv2.aruco.DICT_APRILTAG_36h11

# calibrate using aruco tag
pcds = []
calib_dict = []
for camera in multi_camera_wrapper._all_cameras:
    # marker_size = 0.15
    marker_size = 0.118  # 118mm

    tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
        camera, aruco_dict=aruco_dict, marker_size=marker_size, verbose=True
    )

    # april tag with respect to camera
    april_tag_in_cam = np.eye(4)
    april_tag_in_cam[:3, :3] = rotmat
    april_tag_in_cam[:3, 3:] = tvec

    frames = camera.read_camera()
    rgb = frames[0]["array"]
    depth = frames[1]["array"]

    sn = camera.read_camera()[0]["serial_number"]
    intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]
    dist_coeffs = camera._intrinsics[sn.replace("/", "_")]["distCoeffs"]

    # # sanity check
    undistorted_image = cv2.undistort(rgb, intrinsics, dist_coeffs)
    # cv2.imwrite("undistorted.png", undistorted_image[..., ::-1])
    pose_image = cv2.drawFrameAxes(undistorted_image, intrinsics, dist_coeffs, rotmat, tvec, length=0.1, thickness=3)
    cv2.imwrite('pose_image.png', pose_image)

    with open("d435_cam_info.pkl", "rb") as f:
        d435_cam_info = pickle.load(f)

    april_tag_in_world = d435_cam_info[0]["april_tag_in_world"]
    
    # cam extrinsic in april tag frame
    cam_in_april_tag = np.linalg.inv(april_tag_in_cam)
    cam_extrinsic_in_world = np.dot(april_tag_in_world, cam_in_april_tag)

    print("cam_extrinsic_in_world: ", cam_extrinsic_in_world)

    calib_dict.append(
        {
            "camera_serial_number": sn,
            "intrinsic_info": {
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
            "raw_intrinsics": intrinsics,
            "raw_extrinsics": cam_extrinsic_in_world
        }
    )
    with open("calib_dict.pkl", "wb") as f:
        pickle.dump(calib_dict, f)
    print("saved calib_dict.pkl")   


# sanity check
pcds = []
calib_dict = []
for camera in multi_camera_wrapper._all_cameras:

    frames = camera.read_camera()
    rgb = frames[0]["array"]
    depth = frames[1]["array"]

    sn = camera.read_camera()[0]["serial_number"]
    intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]

    depth[depth == np.inf] = 1.0  # depth[~(depth==np.inf)].max()
    depth[depth == -np.inf] = 1.0  # depth[~(depth==np.inf)].max()
    depth[np.isnan(depth)] = 1.0
    points = depth_to_points(depth, intrinsics, cam_extrinsic_in_world, depth_scale=1000.0)
    points[np.isnan(points)] = 1.0
    points[points == np.inf] = 1.0
    points[points == -np.inf] = 1.0
    colors = rgb.reshape(-1, 3) / 255.0

    pcds.append(points_to_pcd(points, colors=colors))

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
visualize_pcds(pcds+[mesh_frame])