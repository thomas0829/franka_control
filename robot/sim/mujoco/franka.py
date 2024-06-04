# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import mujoco
import mujoco_viewer
import numpy as np
import omegaconf
import torch
import yaml

from utils.transformations import (euler_to_quat, quat_to_euler, rmat_to_euler,
                                   rmat_to_quat)
from utils.transformations_mujoco import mat_to_quat_mujoco, mat_to_euler_mujoco, euler_to_mat_mujoco

log = logging.getLogger(__name__)

from robot.franka_base import FrankaBase


class MujocoManipulatorEnv(FrankaBase):
    """A manipulator environment using MuJoCo.
    """

    def __init__(
        self,
        # robot
        robot_type="panda",
        control_hz=15,
        gripper=True,
        model_name: str = "base_franka",
        # rendering
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["front", "left"],
        calib_dict=None,
        use_rgb=True,
        use_depth=False,
        img_height=480,
        img_width=640,
        # domain randomization
        visual_dr = False,
    ):
        super().__init__(
            robot_type=robot_type,
            control_hz=control_hz,
            gripper=gripper,
        )

        self.robot_desc_mjcf_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"assets/{model_name}.xml"
        )
       
        # mujoco setup
        self.model = mujoco.MjModel.from_xml_path(self.robot_desc_mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.gravity = np.array([0, 0, -9.81])
        self._max_gripper_width = 0.08

        self.frame_skip = int((1 / self.control_hz) / self.model.opt.timestep)

        # get mujoco ids
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EEF")
        self.franka_joint_ids = []
        for i in range(7):
            self.franka_joint_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i+1}"
                )
            ]
        self.franka_finger_joint_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:finger_joint1"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:finger_joint2"
            ),
        ]

        # rendering
        assert (
            has_renderer and has_offscreen_renderer
        ) is False, "both has_renderer and has_offscreen_renderer not supported"
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.img_width = img_width
        self.img_height = img_height
        self.camera_names = camera_names
        self.viewer = None
        # calibrate cameras
        self.calib_dict = calib_dict
        self.reset_camera_pose()

        # domain randomization
        self.visual_dr = visual_dr
        if self.visual_dr:
            self.init_randomize()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def update_joints(
        self, command, velocity=False, blocking=False
    ):
        
        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self.get_joint_positions()

        if blocking:
            time_to_go = self.adaptive_time_to_go(command)
            self.move_to_joint_positions(command, time_to_go=time_to_go)
        else:
            self.update_desired_joint_positions(command)

    def update_desired_joint_positions(self, joint_pos_desired=None, kp=None, kd=None):
        """update joint pos"""

        self.data.ctrl[: len(self.franka_joint_ids)] = joint_pos_desired
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def set_desired_joint_positions(self, joint_pos_desired=None):
        # set position -> sim only has to be stepped once
        self.data.qpos[self.franka_joint_ids] = joint_pos_desired
        mujoco.mj_step(self.model, self.data)

    def move_to_joint_positions(self, joint_pos_desired=None, time_to_go=3):
        
        # use position control -> skip sim for time_to_go
        self.data.ctrl[: len(self.franka_joint_ids)] = joint_pos_desired
        mujoco.mj_step(self.model, self.data, nstep=int(time_to_go//self.model.opt.timestep))

        # # fast reset for simulation -> jump to joint positions
        # self.data.qpos[self.franka_joint_ids] = joint_pos_desired
        # mujoco.mj_step(self.model, self.data)

    def update_gripper(self, command, velocity=False, blocking=False):
        # 1. -> close, 0. -> open
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        if command > 0.0:
            self.data.ctrl[len(self.franka_joint_ids) :] = 0.0
        else:
            self.data.ctrl[len(self.franka_joint_ids) :] = 255.0

        if blocking:
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def set_robot_state(self, robot_state):
        log.warning(
            "set_robot_state is numerically unstable for mujoco_manipulator, proceed with caution...",
        )
        self.data.qpos = robot_state.joint_positions
        self.data.qvel = robot_state.joint_velocities
        self.data.ctrl = self.data.qfrc_bias
        mujoco.mj_step(self.model, self.data)
        if self.gui:
            self.render()

    def get_ee_pose(self):
        return np.concatenate((self.get_ee_pos(), self.get_ee_angle()))

    def get_ee_pos(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        return ee_pos

    def get_ee_angle(self, quat=False):
        ee_mat = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        ee_angle = rmat_to_euler(ee_mat)
        if quat:
            return euler_to_quat(ee_angle)
        else:
            return ee_angle

    def get_joint_positions(self):
        qpos = self.data.qpos[self.franka_joint_ids].copy()
        return qpos

    def get_joint_velocities(self):
        qvel = self.data.qvel[self.franka_joint_ids].copy()
        return qvel

    def get_robot_state(self):
        joint_positions = self.get_joint_positions()
        joint_velocities = self.get_joint_velocities()

        gripper_position = self.get_gripper_state()
        pos, quat = self.get_ee_pos(), self.get_ee_angle(quat=False)

        state_dict = {
            "cartesian_position": np.concatenate([pos, quat]),
            "gripper_position": gripper_position,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
        }
        timestamp_dict = {}

        return state_dict, timestamp_dict

    def get_gripper_state(self):
        if self._gripper:
            return (
                self.data.qpos[self.franka_finger_joint_ids[0]]
                + self.data.qpos[self.franka_finger_joint_ids[1]]
            )
        else:
            return 0.0

    def get_gripper_position(self):
        if self._gripper:
            return 1 - (self.get_gripper_state() / self._max_gripper_width)
        else:
            return 0.0

    def _adaptive_time_to_go_polymetis(
        self, joint_displacement: torch.Tensor, time_to_go_default=1.0
    ):
        """Compute adaptive time_to_go
        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)

        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.
        """
        # TODO verify those limits
        # https://frankaemika.github.io/docs/control_parameters.html
        joint_vel_limits = torch.tensor([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]).float() # robot_model.get_joint_velocity_limits()
        joint_pos_diff = torch.abs(torch.tensor(joint_displacement)).float()
        time_to_go = torch.max(joint_pos_diff / joint_vel_limits * 8.0)
        return max(time_to_go, time_to_go_default)

    def adaptive_time_to_go(self, desired_joint_position, t_min=0, t_max=4):
        curr_joint_position = self.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._adaptive_time_to_go_polymetis(
            displacement
        )
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go
    
    def reset_camera_pose(self):
        if self.calib_dict is not None:

            for sn, camera_name in zip(self.calib_dict.keys(), self.camera_names):
                self.set_camera_intrinsic(
                    camera_name,
                    self.calib_dict[sn]["intrinsic"]["fx"],
                    self.calib_dict[sn]["intrinsic"]["fy"],
                    self.calib_dict[sn]["intrinsic"]["ppx"],
                    self.calib_dict[sn]["intrinsic"]["ppy"],
                    self.calib_dict[sn]["intrinsic"]["fovy"],
                )

            for sn, camera_name in zip(self.calib_dict.keys(), self.camera_names):
                R = np.eye(4)
                R[:3, :3] = np.array(self.calib_dict[sn]["extrinsic"]["ori"])
                R[:3, 3] = np.array(self.calib_dict[sn]["extrinsic"]["pos"]).reshape(-1)
                self.set_camera_extrinsic(camera_name, R)
            self.img_width = self.calib_dict[sn]["intrinsic"]["width"]
            self.img_height = self.calib_dict[sn]["intrinsic"]["height"]

            # push changes from model to data | reset mujoco data
            # mujoco.mj_resetData(self.model, self.data)

    def viewer_setup(self):
        if self.has_renderer:
            return mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                height=self.img_height,
                width=self.img_width,
                hide_menus=True,
            )
        if self.has_offscreen_renderer:
            return mujoco.Renderer(
                self.model, height=self.img_height, width=self.img_width
            )

    def render(self):
        imgs = []

        if not self.viewer:
            self.viewer = self.viewer_setup()

        if self.has_renderer:
            self.viewer.render()
        elif self.has_offscreen_renderer:
            for camera in self.camera_names:
                self.viewer.update_scene(self.data, camera=camera)
                
                color_image = None
                if self.use_rgb:
                    color_image = self.viewer.render().copy()
                    dict_1 = {
                    "serial_number": camera,
                    "array": color_image,
                    "shape": color_image.shape if color_image is not None else None,
                    "type": "rgb",
                }
                    imgs.append(dict_1)
                    
                depth = None
                if self.use_depth:
                    self.viewer.enable_depth_rendering()
                    depth = self.viewer.render().copy()
                    self.viewer.disable_depth_rendering()
                    dict_2 = {
                        "serial_number": camera,
                        "array": depth,
                        "shape": depth.shape if depth is not None else None,
                        "type": "depth",
                    }
                    imgs.append(dict_2)

        return imgs

    def set_camera_intrinsic(self, camera_name, fx, fy, cx, cy, fovy):
        """
        Set camera intrinsic matrix.

        Args:
            camera_name (str): name of camera
            K (np.array): 3x3 camera matrix
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        # testing this
        self.model.cam_fovy[cam_id] = np.degrees(
            2 * np.arctan(self.img_height / (2 * fy))
        )
        # self.model.cam_fovy[cam_id] = fovy

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

    def set_camera_extrinsic(self, camera_name, R):
        """
        Set camera extrinsic matrix.

        Args:
            camera_name (str): name of camera
            R (np.array): 4x4 camera extrinsic matrix
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        cam_base_pos = R[:3, 3]
        cam_base_ori = R[:3, :3]
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )

        self.model.cam_pos[cam_id] = cam_base_pos
        self.model.cam_quat[cam_id] = mat_to_quat_mujoco(
            cam_base_ori @ camera_axis_correction
        ).reshape(-1)

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

    def randomize_camera_pose(self):
        if self.calib_dict is not None:

            # randomize fy (~385 +/- 10) -> used to compute fovy
            for sn, camera_name in zip(self.calib_dict.keys(), self.camera_names):
                
                # compute noise
                high_low = 50.
                fy_noise = np.random.uniform(low=-high_low, high=high_low)
                
                # add noise
                self.set_camera_intrinsic(
                    camera_name,
                    self.calib_dict[sn]["intrinsic"]["fx"],
                    self.calib_dict[sn]["intrinsic"]["fy"] + fy_noise,
                    self.calib_dict[sn]["intrinsic"]["ppx"],
                    self.calib_dict[sn]["intrinsic"]["ppy"],
                    self.calib_dict[sn]["intrinsic"]["fovy"],
                )

            # randomize pos and ori
            for sn, camera_name in zip(self.calib_dict.keys(), self.camera_names):
                
                # compute noise
                pos_noise = np.random.normal(loc=0.0, scale=1e-2, size=(3,))
                ori_noise = np.random.normal(loc=0.0, scale=1e-2, size=(3,))

                # add noise
                pos = np.array(self.calib_dict[sn]["extrinsic"]["pos"]).reshape(-1) + pos_noise
                ori_mat = np.array(self.calib_dict[sn]["extrinsic"]["ori"])
                ori_euler = mat_to_euler_mujoco(ori_mat) + ori_noise
                ori_mat = euler_to_mat_mujoco(ori_euler)

                R = np.eye(4)
                R[:3, :3] = ori_mat
                R[:3, 3] = pos
                self.set_camera_extrinsic(camera_name, R)
            
            # no need to change these
            # self.img_width = self.calib_dict[sn]["intrinsic"]["width"]
            # self.img_height = self.calib_dict[sn]["intrinsic"]["height"]

    def init_randomize(self):
        # camera
        self.calib_dict_copy = self.calib_dict.copy()
        # color
        self.geom_rgba = self.model.geom_rgba.copy()
        # light
        self.light_pos = self.model.light_pos.copy()
        self.light_dir = self.model.light_dir.copy()
        self.light_castshadow = self.model.light_castshadow.copy()
        self.light_ambient = self.model.light_ambient.copy()
        self.light_diffuse = self.model.light_diffuse.copy()
        self.light_specular = self.model.light_specular.copy()

    def reset_randomize(self):
        # color
        self.model.geom_rgba = self.geom_rgba
        # camera pose
        self.calib_dict = self.calib_dict_copy
        self.reset_camera_pose()
        # light
        self.model.light_pos = self.light_pos
        self.model.light_dir = self.light_dir
        self.model.light_castshadow = self.light_castshadow
        self.model.light_ambient = self.light_ambient
        self.model.light_diffuse = self.light_diffuse
        self.model.light_specular = self.light_specular

    def randomize_all_color(self):
        self.model.geom_rgba[:, :3] *= np.random.uniform(
            0.95, 1.05, (self.model.geom_rgba.shape[0], 3)
        )

    def randomize_background_color(self):
       
        geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]

        self.wall_geom_ids = []
        self.table_geom_ids = []
        for name in geom_names:
            if name is None:
                continue
            if "wall" in name:
                self.wall_geom_ids += [
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                    )
                ]
            if "table" in name:
                self.table_geom_ids += [
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                    )
                ]
        geom_ids = self.wall_geom_ids + self.table_geom_ids

        # full color randomization
        # self.model.geom_rgba[geom_ids, :3] = np.random.uniform(
        #     0, 1, (len(geom_ids), 3)
        # )
        # color jitter
        self.model.geom_rgba[geom_ids, :3] *= np.random.uniform(
            0.9, 1.1, (len(geom_ids), 3)
        )
        
    def randomize_light(self):
        
        # change light position and direction -> low impact
        light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "headlight")
        scale = 3e-2
        # Adjust position and direction slightly
        self.model.light_pos[light_id] += np.random.normal(0.0, scale, size=3)
        self.model.light_dir[light_id] += np.random.normal(0.0, scale, size=3)
        
        self.model.light_castshadow = np.random.choice([0, 1])

        # change light color -> large impact
        scale = 3e-2
        self.model.light_ambient += np.random.normal(0.0, scale, size=3)
        self.model.light_diffuse += np.random.normal(0.0, scale, size=3)
        self.model.light_specular += np.random.normal(0.0, scale, size=3)
    
    def randomize(self):

        # reset to intial values
        self.reset_randomize()

        # randomize camera, background color, light
        self.randomize_camera_pose()
        self.randomize_background_color()
        # self.randomize_all_color()
        self.randomize_light()

        # push changes from model to data | reset mujoco data
        # mujoco.mj_resetData(self.model, self.data)