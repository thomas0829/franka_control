import os
import time

import numpy as np
import torch
from robot.real.server_interface import ServerInterface

from robot.controllers.utils import generate_joint_space_min_jerk
from robot.franka_base import FrankaBase

from utils.transformations import (add_poses, euler_to_quat, pose_diff,
                                   quat_to_euler)


class FrankaHardware(FrankaBase):
    def __init__(
        self,
        ip_address,
        robot_type="panda",
        control_hz=15,
        gripper=True,
        custom_controller=True,
        gain_scale=1.5,
        reset_gain_scale=1.0,
    ):
        super().__init__(
            robot_type=robot_type,
            control_hz=control_hz,
            gripper=gripper,
            custom_controller=custom_controller,
        )

        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale

        self.launch_robot(ip_address=ip_address, gripper=gripper)

        self._grasping = False

    # def reset(self):
    #     self.update_joints(self._robot.home_pose, velocity=False, blocking=True)

    def launch_robot(self, ip_address, gripper=True):
        self._robot = ServerInterface(ip_address=ip_address)
    
    def update_joints(
        self, command, velocity=False, blocking=False, cartesian_noise=None
    ):
        self._robot.update_joints(command, velocity=velocity, blocking=blocking, cartesian_noise=None)
    
    def update_desired_joint_positions(self, joint_pos_desired=None, kp=None, kd=None):
        """update joint pos"""

        if joint_pos_desired is not None:

            self._robot.update_desired_joint_positions(joint_pos_desired)

    def move_to_ee_pose(self, pos, quat, time_to_go=3):
        self._robot.move_to_ee_pose(pos, quat, time_to_go)

    def move_to_joint_positions(self, joint_pos_desired=None, time_to_go=3):
        # Use registered controller
        q_current = self._robot.get_joint_positions()

        # generate min jerk trajectory
        dt = 0.1
        waypoints = generate_joint_space_min_jerk(
            start=q_current, goal=joint_pos_desired, time_to_go=time_to_go, dt=dt
        )
        # reset using min_jerk traj
        for i in range(len(waypoints)):
            self.update_desired_joint_positions(
                joint_pos_desired=waypoints[i]["position"].tolist(),
                # kp=self.reset_gain_scale
                # * torch.Tensor(self._robot.metadata.default_Kq),
                # kd=self.reset_gain_scale
                # * torch.Tensor(self._robot.metadata.default_Kqd),
            )
            time.sleep(dt)

    def update_gripper(self, command, velocity=False, blocking=False):
        self._robot.update_gripper(command, velocity=velocity, blocking=blocking)

    def add_noise_to_joints(self, original_joints, cartesian_noise):
        original_joints = torch.Tensor(original_joints)

        pos, quat = self._robot.robot_model.forward_kinematics(original_joints)
        curr_pose = pos.tolist() + quat_to_euler(quat).tolist()
        new_pose = add_poses(cartesian_noise, curr_pose)

        new_pos = torch.Tensor(new_pose[:3])
        new_quat = torch.Tensor(euler_to_quat(new_pose[3:]))

        noisy_joints, success = self._robot.solve_inverse_kinematics(
            new_pos, new_quat, original_joints
        )

        if success:
            desired_joints = noisy_joints
        else:
            desired_joints = original_joints

        return desired_joints.tolist()

    def get_joint_positions(self):
        return self._robot.get_joint_positions()

    def get_joint_velocities(self):
        return self._robot.get_joint_velocities()

    def get_gripper_position(self):
        return self._robot.get_gripper_position()

    def get_ee_pose(self):
        return self._robot.get_ee_pose()

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_angle(self):
        return self.get_ee_pose()[3:]

    def get_gripper_state(self): 
        if self._gripper:
            return self._robot.get_gripper_state()
        else:
            return 0.0

    def get_robot_state(self):
        return self._robot.get_robot_state()

    def adaptive_time_to_go(self, desired_joint_position, t_min=1, t_max=4):
        curr_joint_position = self._robot.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._robot._adaptive_time_to_go(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go
