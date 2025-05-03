import json
from datetime import datetime
from perception.cameras.multi_camera_wrapper import MultiCameraWrapper
import logging
import cv2
import pickle   
multi_camera_wrapper = MultiCameraWrapper(type="realsense", use_threads=False)
num_cameras = multi_camera_wrapper.num_cameras
all_cameras = multi_camera_wrapper._all_cameras
logging.info(f"Number of cameras: {num_cameras}")
assert len(all_cameras) != 0, "No cameras found"
from utils.pointclouds import *
from matplotlib import pyplot as plt    


def get_pose_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 6D pose (translation + rotation) into a 4x4 transformation matrix.

    Assumes:
        - pose[:3]: (x, y, z) translation
        - pose[3:6]: (roll, pitch, yaw) rotation angles in radians
        - Rotation order: applied as 'xyz' intrinsic rotations (roll → pitch → yaw)
    
    Args:
        pose (np.ndarray): Array of shape (6,) representing translation and rotation.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    curr_pose_matrix = np.eye(4)
    curr_pose_matrix[:3, 3] = pose[:3]
    curr_pose_matrix[:3, :3] = R.from_euler('xyz', pose[3:6], degrees=False).as_matrix()
    return curr_pose_matrix

def get_xyzrpy_from_matrix(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 homogeneous transformation matrix into a 6D pose vector (x, y, z, roll, pitch, yaw).

    Assumes:
        - T[:3, 3]: translation (x, y, z)
        - T[:3, :3]: rotation matrix
        - Rotation order: 'xyz' intrinsic (roll → pitch → yaw)

    Args:
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        np.ndarray: Array of shape (6,) representing [x, y, z, roll, pitch, yaw].
    """
    translation = T[:3, 3]
    rotation = R.from_matrix(T[:3, :3])
    rpy = rotation.as_euler('xyz', degrees=False)
    return np.concatenate([translation, rpy])


def calibrate_cameras(args):
    # configs
    marker_size = args.marker_size  # 137mm
    aruco_dict = args.aruco_dict
    cam_info_file = args.cam_info_file

    # STEP 1: calibrate camera 1 using april tag 25h9
    pcds = []
    calib_dict = []
    for camera in multi_camera_wrapper._all_cameras:
        tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
            camera, aruco_dict=aruco_dict, marker_size=marker_size, verbose=True
        )

        # apriltag in camera frame
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotmat
        extrinsics[:3, 3:] = tvec

        frames = camera.read_camera()
        rgb = frames[0]["array"]
        depth = frames[1]["array"]
        sn = camera.read_camera()[0]["serial_number"]
        intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]
        dist_coeffs = camera._intrinsics[sn.replace("/", "_")]["distCoeffs"]


        undistorted_image = cv2.undistort(rgb, intrinsics, dist_coeffs)
        pose_image = cv2.drawFrameAxes(undistorted_image, intrinsics, dist_coeffs, rotmat, tvec, length=0.1, thickness=3)
        
        # plt.imshow(pose_image)
        # plt.show()
        # cv2.imwrite('pose_image.png', pose_image)
        
        # camera in apriltag frame -> world frame
        extrinsics_inv = np.linalg.inv(extrinsics)

        # unit: m
        # aruco_offset = np.array(
        #     [
        #         -0.000875,
        #         -0.141125,
        #         0.0,
        #     ]
        # )
        aruco_offset = np.array(
            [
                # -0.00875,
                0.00875,
                -0.141125,
                # -0.11,
                0
            ]
        )
        extrinsics_inv[:3, 3] += aruco_offset
        
        
        # # TODO: 
        extrinsics_inv[0, 3] -= 0.034 # x forward
        extrinsics_inv[1, 3] -= 0.03
        # extrinsics_inv[1, 3] -= 0.01
        extrinsics_inv[2, 3] -= 0.045
        
        
        # ex_pose = get_xyzrpy_from_matrix(extrinsics_inv)
        # ex_pose[-1] -= 0.07
        # extrinsics_inv = get_pose_matrix(ex_pose)
        

        # save camera info
        calib_dict.append(
            {
                "camera_serial_number": sn,
                "intrinsics_dict": {
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
                "camera_base_ori": extrinsics_inv[:3, :3],
                "camera_base_pos": extrinsics_inv[:3, 3:],
                "intrinsic": intrinsics,
                "extrinsic_in_world": extrinsics_inv,
            }
        )

        # with open(cam_info_file, "wb") as f:
        #     pickle.dump(calib_dict, f)

        # visualize point cloud
        points = depth_to_points(depth, intrinsics, extrinsics_inv, depth_scale=1000.0)
        colors = rgb.reshape(-1, 3) / 255.0
        points, colors = crop_points(points, colors=colors)
        pcds.append(points_to_pcd(points, colors=colors))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # visualize_pcds(pcds+[mesh_frame])
    visualize_pcds(pcds)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker_size", type=float, default=0.118)
    parser.add_argument("--aruco_dict", type=str, default=cv2.aruco.DICT_APRILTAG_25h9)
    parser.add_argument("--cam_info_file", type=str, default="d435_cam_info.pkl")
    args = parser.parse_args()
    calibrate_cameras(args)

# # STEP 2: calculate the transformation between camera 1 and apriltag DICT_APRILTAG_36h11
# pcds = []
# calib_dict = []
# for camera in multi_camera_wrapper._all_cameras:
#     # marker_size = 0.15
#     marker_size = 0.118  # 118mm

#     tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
#         camera, aruco_dict=cv2.aruco.DICT_APRILTAG_36h11, marker_size=marker_size, verbose=True
#     )

#     # april tag with respect to camera
#     extrinsics = np.eye(4)
#     extrinsics[:3, :3] = rotmat
#     extrinsics[:3, 3:] = tvec

#     frames = camera.read_camera()
#     rgb = frames[0]["array"]
#     depth = frames[1]["array"]

#     sn = camera.read_camera()[0]["serial_number"]
#     intrinsics = camera._intrinsics[sn.replace("/", "_")]["cameraMatrix"]
#     dist_coeffs = camera._intrinsics[sn.replace("/", "_")]["distCoeffs"]

#     # sanity check
#     undistorted_image = cv2.undistort(rgb, intrinsics, dist_coeffs)
#     # cv2.imwrite("undistorted.png", undistorted_image[..., ::-1])
#     # cv2.imshow("undistorted", undistorted_image)
#     pose_image = cv2.drawFrameAxes(undistorted_image, intrinsics, dist_coeffs, rotmat, tvec, length=0.1, thickness=3)
#     cv2.imwrite('pose_image2.png', pose_image)



#     with open(cam_info_file, "rb") as f:
#         calib_dict = pickle.load(f)
    

#     # cam extrinsic in world frame
#     cam_extrinsic_in_world = np.eye(4)
#     cam_extrinsic_in_world[:3, :3] = calib_dict[0]["camera_base_ori"]
#     cam_extrinsic_in_world[:3, 3] = calib_dict[0]["camera_base_pos"].reshape(3)


#     # TODO: april tag with respect to world, check if that is right
#     april_tag_in_world = np.dot(cam_extrinsic_in_world, extrinsics)

#     calib_dict[0]["april_tag_in_world"] = april_tag_in_world
#     with open(cam_info_file, "wb") as f:
#         pickle.dump(calib_dict, f)
    
#     print("april_tag_in_world: ", april_tag_in_world)

