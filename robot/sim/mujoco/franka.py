# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from typing import List, Tuple

import mujoco
import mujoco_viewer
import numpy as np
import omegaconf
import torch
import torchcontrol as toco
import yaml

# from polymetis.utils.data_dir import get_full_path_to_urdf
# from polysim.envs import AbstractControlledEnv
from helpers.transformations import (
    euler_to_quat,
    quat_to_euler,
    rmat_to_euler,
    rmat_to_quat,
)

log = logging.getLogger(__name__)
from torchcontrol.modules.feedback import JointSpacePD
from torchcontrol.policies import JointImpedanceControl

from robot.controllers.utils import generate_joint_space_min_jerk
from robot.franka_base import FrankaBase


class MujocoManipulatorEnv(FrankaBase):
    """A manipulator environment using MuJoCo.

    Args:
        robot_model_cfg: A Hydra configuration file containing information needed for the
                        robot model, e.g. URDF. For an example, see
                        `polymetis/conf/robot_model/franka_panda.yaml`

                        NB: When specifying the path to a URDF file in
                        `robot_description_path`, ensure an MJCF file exists at the
                        same path and same filename with a .mjcf extension. For an
                        example, see `polymetis/data/franka_panda/panda_arm.[urdf|mjcf]`

        use_grav_comp: If True, adds gravity compensation torques to the input torques.

        gravity: Value of gravity, default to 9.81
    """

    def __init__(
        self,
        # robot_model_cfg: DictConfig,
        config_name: str = "franka_panda_with_hand",
        model_name: str = "base_franka",
        use_grav_comp: bool = False,
        gravity: float = 9.81,
        impedance_control: bool = False,
        torque_control: bool = False,
        robot_type="panda",
        control_hz=15,
        gripper=True,
        custom_controller=True,
        gain_scale=1.0,
        reset_gain_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["front", "left"],
        use_rgb=True,
        use_depth=True,
        img_height=480,
        img_width=640,
    ):
        super().__init__(
            robot_type=robot_type,
            control_hz=control_hz,
            gripper=gripper,
            custom_controller=custom_controller,
        )

        self.policy = None
        self.torque_control = torque_control
        self.impedance_control = impedance_control
        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale

        assert (
            has_renderer and has_offscreen_renderer
        ) is False, "both has_renderer and has_offscreen_renderer not supported"
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.viewer = None
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.img_width = img_width
        self.img_height = img_height
        self.camera_names = camera_names

        # import hydra
        # hydra.initialize(config_path=".")
        # robot_model_cfg = hydra.compose(config_name=config_name)

        robot_model_cfg = omegaconf.OmegaConf.load(f"{config_name}.yaml")
        self.model_cfg = robot_model_cfg
        # self.robot_description_path = get_full_path_to_urdf(
        #     self.model_cfg.robot_description_path
        # )
        self.robot_description_path = "panda_arm.urdf"
        # robot_desc_mjcf_path = (
        #     os.path.splitext(self.robot_description_path)[0] + ".mjcf"
        # )

        robot_desc_mjcf_path = f"robot/sim/mujoco/assets/{model_name}.xml"
        assert os.path.exists(
            robot_desc_mjcf_path
        ), f"No MJCF file found. Create an MJCF file at {robot_desc_mjcf_path} to use the MuJoCo simulator."
        self.model = mujoco.MjModel.from_xml_path(robot_desc_mjcf_path)
        # self.model.opt.timestep = 1e-4 -> tried 1000hz, doesn't make a difference
        self.data = mujoco.MjData(self.model)

        # shouldnt make a difference if use_grav_comp
        self.model.opt.gravity = np.array([0, 0, -gravity])

        self.frame_skip = int((1 / self.control_hz) / self.model.opt.timestep)

        self.n_dofs = self.model_cfg.num_dofs
        # # no gripper
        # assert (
        #     len(self.model_cfg.controlled_joints - 1) == self.n_dofs
        # ), f"Number of controlled joints ({len(self.model_cfg.controlled_joints - 1)}) != number of DOFs ({self.n_dofs})"
        # assert (
        #     self.model.nu == self.n_dofs
        # ), f"Number of actuators ({self.model.nu}) != number of DOFs ({self.n_dofs})"

        self.ee_link_idx = self.model_cfg.ee_link_idx
        self.ee_link_name = self.model_cfg.ee_link_name # hand | 'panda_link8'
        self.rest_pose = self.model_cfg.rest_pose
        self.joint_limits_low = np.array(self.model_cfg.joint_limits_low)
        self.joint_limits_high = np.array(self.model_cfg.joint_limits_high)
        if self.model_cfg.joint_damping is None:
            self.joint_damping = None
        else:
            self.joint_damping = np.array(self.model_cfg.joint_damping)
        if self.model_cfg.torque_limits is None:
            self.torque_limits = np.inf * np.ones(self.n_dofs)
        else:
            self.torque_limits = np.array(self.model_cfg.torque_limits)
        
        self.use_grav_comp = use_grav_comp

        self.prev_torques_commanded = np.zeros(self.n_dofs)
        self.prev_torques_applied = np.zeros(self.n_dofs)
        self.prev_torques_measured = np.zeros(self.n_dofs)
        self.prev_torques_external = np.zeros(self.n_dofs)

        # polymetis control: load robot model
        self.toco_robot_model = toco.models.RobotModelPinocchio(
            self.robot_description_path, robot_model_cfg["ee_link_name"]
        )

        # polymetis control: load robot hardware config
        # polymetis/polymetis/conf/robot_client/franka_hardware.yaml -> replaced by R2D2
        with open("robot/real/config/franka_hardware.yaml", "r") as stream:
            try:
                franka_hardware_conf = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.metadata = franka_hardware_conf["robot_client"]["metadata_cfg"]

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EEF")
        self.franka_joint_ids = []
        for i in range(7):
            self.franka_joint_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i+1}"
                )
            ]

        if self.torque_control:
            # self.model.dof_damping[self.franka_joint_ids] = self.joint_damping * 100
            self.model.actuator_ctrlrange[self.franka_joint_ids] = np.array(
                [self.joint_limits_low, self.joint_limits_high]
            ).T
            self.torque_limits = np.array(self.model_cfg.torque_limits)
            self.model.actuator_forcerange[self.franka_joint_ids] = np.stack([-self.torque_limits, self.torque_limits], axis=-1)

            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

    def get_current_joint_torques(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            np.ndarray: Torques received from apply_joint_torques
            np.ndarray: Torques sent to robot (e.g. after clipping)
            np.ndarray: Torques generated by the actuators (e.g. after grav comp)
            np.ndarray: Torques exerted onto the robot
        """
        return (
            self.prev_torques_commanded,
            self.prev_torques_applied,
            self.prev_torques_measured,
            self.prev_torques_external,  # zeros
        )

    def apply_joint_torques(self, torques: np.ndarray):
        """
        input:
            np.ndarray: Desired torques
        Returns:
            np.ndarray: final applied torque
        """

        self.prev_torques_commanded = torques

        applied_torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        self.prev_torques_applied = applied_torques.copy()

        if self.use_grav_comp:
            # applied_torques += (self.data.qfrc_bias[self.franka_joint_ids] * 0.97)
            applied_torques += self.data.qfrc_bias[self.franka_joint_ids]
        self.prev_torques_measured = applied_torques.copy()

        self.data.ctrl[self.franka_joint_ids] = applied_torques

        mujoco.mj_step(self.model, self.data)

        return applied_torques

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
                if self.use_depth:
                    self.viewer.enable_depth_rendering()
                    depth = self.viewer.render().copy()
                    self.viewer.disable_depth_rendering()

                if not self.use_rgb and self.use_depth:
                    color_image = np.zeros((depth.shape[0], depth.shape[1], 3))
                else:
                    color_image = self.viewer.render().copy()

                dict_1 = {
                    "serial_number": camera,
                    "array": color_image,
                    "shape": color_image.shape,
                    "type": "rgb",
                }
                dict_2 = {
                    "serial_number": camera,
                    "array": depth,
                    "shape": depth.shape,
                    "type": "depth",
                }
                imgs.append(dict_1)
                imgs.append(dict_2)

        return imgs

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

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def get_ee_pose(self):
        return self.get_ee_pos(), self.get_ee_angle()

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
            "joint_positions": torch.tensor(joint_positions, dtype=torch.float32),
            "joint_velocities": torch.tensor(joint_velocities, dtype=torch.float32),
            # "joint_torques_computed": list(robot_state.joint_torques_computed),
            # "prev_joint_torques_computed": list(
            #     robot_state.prev_joint_torques_computed
            # ),
            # "prev_joint_torques_computed_safened": list(
            #     robot_state.prev_joint_torques_computed_safened
            # ),
            # "motor_torques_measured": list(robot_state.motor_torques_measured),
            # "prev_controller_latency_ms": robot_state.prev_controller_latency_ms,
            # "prev_command_successful": robot_state.prev_command_successful,
        }

        # timestamp_dict = {
        #     "robot_timestamp_seconds": robot_state.timestamp.seconds,
        #     "robot_timestamp_nanos": robot_state.timestamp.nanos,
        # }
        timestamp_dict = {}

        return state_dict, timestamp_dict

    def is_running_policy(self):
        return self.policy is not None

    def _start_custom_controller(self, q_desired=None):
        if self.impedance_control:
            self.policy = JointImpedanceControl(
                joint_pos_current=torch.tensor(self.get_joint_positions())
                if q_desired is None
                else q_desired,
                Kp=1.0 # 0.4
                * self.gain_scale
                * torch.tensor(self.metadata["default_Kq"], dtype=torch.float32),
                Kd=1.0
                * self.gain_scale
                * torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32),
                robot_model=self.toco_robot_model,
                ignore_gravity=True,
            )
        else:
            self.policy = JointSpacePD(
                Kp=1.0
                * self.gain_scale
                * torch.tensor(self.metadata["default_Kq"], dtype=torch.float32),
                Kd=1.0
                * self.gain_scale
                * torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32),
            )

        return self.policy

    def _terminate_current_controller(self):
        self.policy = None

    def _update_current_controller(self, udpate_pkt):
        if self.impedance_control and "q_desired" in udpate_pkt.keys():
            self.policy.joint_pos_desired = torch.nn.Parameter(udpate_pkt["q_desired"])
            return self.policy.forward(self.get_robot_state()[0])

        elif "q_desired" in udpate_pkt.keys():
            return {
                "joint_torques": self.policy.forward(
                    joint_pos_current=self.get_robot_state()[0][
                        "joint_positions"
                    ].clone(),
                    joint_vel_current=self.get_robot_state()[0][
                        "joint_velocities"
                    ].clone(),
                    joint_pos_desired=udpate_pkt["q_desired"].clone().detach(),
                    joint_vel_desired=torch.zeros_like(udpate_pkt["q_desired"]),
                )
            }

    def _adaptive_time_to_go_polymetis(
        self, robot_model, joint_displacement: torch.Tensor, time_to_go_default=1.0
    ):
        """Compute adaptive time_to_go
        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)

        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.
        """
        joint_vel_limits = robot_model.get_joint_velocity_limits()
        joint_pos_diff = torch.abs(joint_displacement)
        time_to_go = torch.max(joint_pos_diff / joint_vel_limits * 8.0)
        return max(time_to_go, time_to_go_default)

    def adaptive_time_to_go(self, desired_joint_position, t_min=1, t_max=4):
        curr_joint_position = self.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._adaptive_time_to_go_polymetis(
            self.toco_robot_model, displacement
        )
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go

    def update_joints(
        self, command, velocity=False, blocking=False, cartesian_noise=None
    ):
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self.get_joint_positions()

        # BLOCKING EXECUTION
        # make sure custom controller is running
        if blocking and self.custom_controller:
            if not self.is_running_policy():
                self._start_custom_controller()
            time_to_go = self.adaptive_time_to_go(command)
            self.move_to_joint_positions(command, time_to_go=time_to_go)
        # kill cartesian impedance
        elif blocking:
            if self.is_running_policy():
                self._terminate_current_controller()

            time_to_go = self.adaptive_time_to_go(command)
            self._robot.move_to_joint_positions(command, time_to_go=time_to_go)

            self._robot.start_cartesian_impedance()

        # NON BLOCKING
        else:

            def helper_non_blocking():
                if not self.is_running_policy():
                    if self.custom_controller:
                        self._start_custom_controller()
                    else:
                        self._robot.start_cartesian_impedance()
                if self.custom_controller:
                    # run controller loop for int((1/self.control_hz) / self.model.opt.timestep) instead of time.sleep(1/self.control_hz)
                    for _ in range(self.frame_skip):
                        self.update_desired_joint_positions(command)
                        # self.update_desired_joint_positions(command + torch.normal(mean=0., std=1e-1, size=command.shape))
                else:
                    self._robot.update_desired_joint_positions(command)

            # run_threaded_command(helper_non_blocking)
            helper_non_blocking()

    # def update_gripper(self, gripper, velocity=False, blocking=False):
    #     # TODO gripper in sim
    #     self.data.ctrl[-1] = gripper * 255
    #     mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def update_desired_joint_positions(self, q_desired=None, kp=None, kd=None):
        """update joint pos"""

        udpate_pkt = {}

        if q_desired is not None:
            udpate_pkt["q_desired"] = (
                q_desired if torch.is_tensor(q_desired) else torch.tensor(q_desired)
            )
        if kp is not None:
            udpate_pkt["kp"] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt["kd"] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        output_pkt = self._update_current_controller(udpate_pkt)

        if output_pkt is not None and self.torque_control:
            # WARNING: actuator must be motor
            torques = output_pkt["joint_torques"].detach().numpy()
            self.apply_joint_torques(torques)
        elif "q_desired" in udpate_pkt:
            # WARNING: actuator must be general or position
            self.data.ctrl[: len(self.franka_joint_ids)] = udpate_pkt["q_desired"]
            # mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

    def move_to_joint_positions(self, q_desired=None, time_to_go=3):
        # fast reset for simulation -> jump to joint positions
        # self.data.qpos[self.franka_joint_ids] = q_desired
        # mujoco.mj_step(self.model, self.data)
        # return

        # Use registered controller
        q_current = torch.tensor(self.get_joint_positions())

        # generate min jerk trajectory
        dt = 0.1
        waypoints = generate_joint_space_min_jerk(
            start=q_current, goal=q_desired, time_to_go=time_to_go, dt=dt
        )
        # reset using min_jerk traj
        for i in range(len(waypoints)):
            # run simulation for dt instead of time.sleep(dt)
            for _ in range(int(dt // self.model.opt.timestep)):
                self.update_desired_joint_positions(
                    q_desired=waypoints[i]["position"],
                    kp=self.reset_gain_scale
                    * torch.nn.Parameter(
                        torch.tensor(self.metadata["default_Kq"], dtype=torch.float32)
                    ),
                    kd=self.reset_gain_scale
                    * torch.nn.Parameter(
                        torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32)
                    ),
                )
            # time.sleep(dt)

        # reset back gains to gain-policy
        self.update_desired_joint_positions(
            kp=self.gain_scale
            * torch.nn.Parameter(
                torch.tensor(self.metadata["default_Kq"], dtype=torch.float32)
            ),
            kd=self.gain_scale
            * torch.nn.Parameter(
                torch.tensor(self.metadata["default_Kqd"], dtype=torch.float32)
            ),
        )
        if self.has_renderer:
            self.render()

    def update_gripper(self, command, velocity=True, blocking=False):
        # TODO grasping for sim
        return

        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        # https://github.com/facebookresearch/fairo/issues/1398
        # for robotiq consider using
        # self._gripper.grasp(grasp_width=self._max_gripper_width * (1 - command), speed=0.05, force=0.5, blocking=blocking)
        # for franka gripper, use discrete grasp/ungrasp
        if command > 0.0:
            self._gripper.grasp(
                grasp_width=0.0, speed=0.5, force=5.0, blocking=blocking
            )
        else:
            self._gripper.grasp(
                grasp_width=self._max_gripper_width,
                speed=0.5,
                force=5.0,
                blocking=blocking,
            )

    def get_gripper_state(self):
        return np.array(0)

    def get_gripper_position(self):
        return np.array(0)
