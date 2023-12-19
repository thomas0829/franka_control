TODO
- camera intr. extr, pts functionality
- sim camera from calibration file

def get_camera_intrinsic(self, camera_name):
    """
    Obtains camera intrinsic matrix.

    Args:
        camera_name (str): name of camera
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = self.model.cam_fovy[cam_id]

    # Compute intrinsic parameters
    fy = self.img_height / (2 * np.tan(np.radians(fovy / 2)))
    fx = fy
    cx = self.img_width / 2
    cy = self.img_height / 2

    # Camera intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K

def get_camera_extrinsic(self, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    https://github.com/ARISE-Initiative/robosuite/blob/de64fa5935f9f30ce01b36a3ef1a3242060b9cdb/robosuite/utils/camera_utils.py#L39

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    camera_pos = self.data.cam_xpos[cam_id]
    camera_rot = self.data.cam_xmat[cam_id].reshape(3, 3)

    R = np.eye(4)
    R[:3, :3] = camera_rot
    R[:3, 3] = camera_pos

    camera_axis_correction = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    R = R @ camera_axis_correction

    return R
