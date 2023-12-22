import numpy as np
import open3d as o3d
from cameras.multi_camera_wrapper import MultiCameraWrapper

from cameras.calibration.utils import read_calibration_file
from helpers.pointclouds import depth_to_points, compute_camera_intrinsic, compute_camera_extrinsic, points_to_pcd, crop_points, visualize_pcds

if __name__ == "__main__":

    # read calibration file
    calib_dict = read_calibration_file("cameras/calibration/calibration.json")

    # gather cameras
    camera_model = "realsense"
    if camera_model == "realsense":
        from cameras.realsense_camera import gather_realsense_cameras
        cameras = gather_realsense_cameras()
    elif camera_model == "zed":
        from cameras.zed_camera import gather_zed_cameras
        cameras = gather_zed_cameras()

    from cameras.multi_camera_wrapper import MultiCameraWrapper
    camera_reader = MultiCameraWrapper(cameras)
    
    # read cameras
    imgs = camera_reader.read_cameras()

    # prepare rgb and depth
    depth_dict = {}
    rgb_dict = {}
    for img in imgs:
        sn = img["serial_number"].split("/")[0]
        if img["type"] == "depth":
            depth_dict[sn] = img["array"]
        elif img["type"] == "rgb":
            rgb_dict[sn] = img["array"]

    assert all(key in calib_dict.keys() for key in depth_dict.keys()), "Missing calibration for some cameras connected!"

    # convert to (colored) pointclouds
    pcds = []
    for sn, depth in depth_dict.items():

        intrinsic = compute_camera_intrinsic(
            calib_dict[sn]["intrinsic"]["fx"],
            calib_dict[sn]["intrinsic"]["fy"],
            calib_dict[sn]["intrinsic"]["ppx"],
            calib_dict[sn]["intrinsic"]["ppy"],
        )
        extrinsic = compute_camera_extrinsic(
            pos=calib_dict[sn]["extrinsic"]["pos"], ori=calib_dict[sn]["extrinsic"]["ori"]
        )

        points = depth_to_points(depth, intrinsic, extrinsic)

        points, colors = crop_points(points, colors=rgb_dict[sn].reshape(-1,3))
        pcds.append(points_to_pcd(points, colors=colors.astype(np.float32) / 255.))

    # add origin (0,0,0) for visualization
    pcds.append(points_to_pcd(np.zeros((1,3)), colors=np.array([[255., 0., 0.]])))
    # add ground plane for visualization
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1e-3)
    mesh_box = mesh_box.translate(np.array([-0.75, -0.75, 5e-4]))
    pcds.append(mesh_box)

    visualize_pcds(pcds)