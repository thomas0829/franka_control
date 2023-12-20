import numpy as np
import open3d as o3d

def depth_to_points(depth, intrinsic, extrinsic, depth_scale=1000.):
    height, width = depth.shape[:2]
    depth = depth / depth_scale
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

def crop_points(points, colors=None, crop_min=[-0.2, -1., -0.2], crop_max=[1., 1., 1.]):

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