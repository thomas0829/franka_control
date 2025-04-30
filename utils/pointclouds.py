import json

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# TYPING
from typing import List


def read_calibration_file(filename):

    with open(filename) as file:
        calib_file = json.load(file)

    calib_dict = {}
    for calib in calib_file:
        sn = calib["camera_serial_number"]
        calib_dict[sn] = {"intrinsic": {}, "extrinsic": {}}
        calib_dict[sn]["intrinsic"] = calib["intrinsics"]
        calib_dict[sn]["extrinsic"]["pos"] = calib["camera_base_pos"]
        calib_dict[sn]["extrinsic"]["ori"] = calib["camera_base_ori"]

    return calib_dict


def depth_to_points(depth, intrinsic, extrinsic, depth_scale=1000.0):
    height, width = depth.shape[:2]
    depth = depth.squeeze() / depth_scale
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
    py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
    points = np.stack((px, py, depth, np.ones(depth.shape)), axis=-1)
    points = (extrinsic @ points.reshape(-1, 4).T).T
    points = points[:, :3]
    return points


def compute_camera_intrinsic(fx, fy, ppx, ppy):
    return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]])


def compute_camera_extrinsic(pos, ori):
    cam_mat = np.eye(4)
    cam_mat[:3, :3] = ori
    cam_mat[:3, 3] = pos
    return cam_mat


def points_to_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_pcds(pcds):
    o3d.visualization.draw_geometries(pcds)


def crop_points(
    points, colors=None, crop_min=[-1.0, -1.0, -1.0], crop_max=[1.0, 1.0, 1.0]
    # points, colors=None, crop_min=[-0.2, -1.0, -0.2], crop_max=[1.0, 1.0, 1.0]
):

    idx_max = np.all((points < crop_max), axis=1)
    points = points[idx_max]
    if colors is not None:
        colors = colors[idx_max]

    idx_min = np.all((points > crop_min), axis=1)
    points = points[idx_min]
    if colors is not None:
        colors = colors[idx_min]

    if colors is not None:
        return points, colors
    return points

def pixel_to_world(clicked_2d_point, depth, cam_K, cam_pose, depth_scale=1000.0):
    """
    Project a 2D pixel point back to 3D world coordinates.
    
    Args:
        clicked_2d_point: (u, v) pixel coordinates.
        depth: 2D depth image.
        cam_K: 3x3 intrinsic matrix.
        cam_pose: 4x4 extrinsic matrix (camera-to-world).
        depth_scale: Scale factor for depth (e.g., 1000 if in mm).
        
    Returns:
        3D world coordinates as a (3,) numpy array.
    """
    u, v = clicked_2d_point
    z = depth[v, u] / depth_scale  # note: (v, u) indexing

    if z == 0:
        raise ValueError("Depth at the selected pixel is 0 (invalid)")

    # Back-project to camera coordinates
    x = (u - cam_K[0, 2]) * z / cam_K[0, 0]
    y = (v - cam_K[1, 2]) * z / cam_K[1, 1]
    point_cam = np.array([x, y, z, 1.0])  # homogeneous

    # Transform to world frame
    point_world = cam_pose @ point_cam
    return point_world[:3]

def create_point_sphere(center, radius=0.02, color=[1.0, 0.0, 0.0]):
    """
    Create a colored sphere centered at a 3D point.

    Args:
        center: (3,) numpy array of world coordinates.
        radius: radius of the sphere.
        color: RGB list, e.g., [1.0, 0.0, 0.0] for red.

    Returns:
        Open3D TriangleMesh sphere.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere

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


def interpolate_6d_poses(pose1: np.ndarray, pose2: np.ndarray, distance_threshold: float = 0.1) -> List[np.ndarray]:
    """Interpolate between two 6D poses (translation + rotation) into intermediate poses."""
    
    def interpolate_translation(t1: np.ndarray, t2: np.ndarray, alpha: float) -> np.ndarray:
        return (1 - alpha) * t1 + alpha * t2

    def interpolate_rotation(r1: np.ndarray, r2: np.ndarray, alpha: float) -> np.ndarray:
        # Convert rotation matrices to quaternions and perform SLERP interpolation
        q1 = R.from_matrix(r1).as_quat()
        q2 = R.from_matrix(r2).as_quat()
        slerp = Slerp([0, 1], R.from_quat([q1, q2]))
        return slerp([alpha]).as_matrix()[0]

    # Extract translation and rotation components from input poses
    t1, r1 = pose1[:3, 3], pose1[:3, :3]
    t2, r2 = pose2[:3, 3], pose2[:3, :3]

    # Calculate the Euclidean distance between the translations
    translation_distance = np.linalg.norm(t2 - t1)
    print(translation_distance, t2, t1)

    # Determine the number of interpolation points based on distance
    num_points = max(2, int(np.ceil(translation_distance / distance_threshold)))

    # Perform interpolation
    poses = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        interpolated_translation = interpolate_translation(t1, t2, alpha)
        interpolated_rotation = interpolate_rotation(r1, r2, alpha)
        interpolated_pose = np.zeros(6)
        interpolated_pose[:3] = interpolated_translation
        interpolated_pose[3:] = R.from_matrix(interpolated_rotation).as_euler('xyz', degrees=False)
        poses.append(interpolated_pose)

    return poses

def depth_to_pc(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    depth_mask = (depth > 0)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[depth_mask]
    xyz = np.linalg.solve(cam_K, uvd.T).T
    return xyz

def rotate_pose_z_axis_counterclockwise(pose, angle_deg):
    """Rotate a pose counter-clockwise around its own Z-axis by a given angle."""
    theta = np.deg2rad(angle_deg)  # positive for counter-clockwise
    R_delta = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    rotated_pose = pose.copy()
    rotated_pose[:3, :3] = pose[:3, :3] @ R_delta  # post-multiply to rotate around local Z-axis
    return rotated_pose