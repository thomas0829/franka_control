# ROBOT SPECIFIC IMPORTS
import os
import time

import grpc
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface
from torchcontrol.policies import JointImpedanceControl

# UTILITY SPECIFIC IMPORTS
from helpers.subprocess_utils import run_threaded_command
from helpers.transformations import (add_poses, euler_to_quat, pose_diff,
                                     quat_to_euler)
from robot.controllers.utils import generate_joint_space_min_jerk
from robot.franka_base import FrankaBase


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

    def reset(self):
        self.update_joints(self._robot.home_pose, velocity=False, blocking=True)

    def launch_robot(self, ip_address, gripper=True):
        self._robot = RobotInterface(ip_address=ip_address)
        self._gripper = None
        if gripper:
            self._gripper = GripperInterface(ip_address=ip_address)
            self._max_gripper_width = self._gripper.metadata.max_width

    def _start_custom_controller(self):
        
        self.policy = JointImpedanceControl(
            joint_pos_current=self._robot.get_joint_positions(),
            Kp=1.0 * self.gain_scale * torch.Tensor(self._robot.metadata.default_Kq),
            Kd=1.0 * self.gain_scale * torch.Tensor(self._robot.metadata.default_Kqd),
            robot_model=self._robot.robot_model,
            ignore_gravity=True,
        )
        self._robot.send_torch_policy(self.policy, blocking=False)

    def update_joints(
        self, command, velocity=False, blocking=False, cartesian_noise=None
    ):
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        # BLOCKING EXECUTION
        # make sure custom controller is running
        if blocking and self.custom_controller:
            if not self._robot.is_running_policy():
                self._start_custom_controller()
            time_to_go = self.adaptive_time_to_go(command)
            self.move_to_joint_positions(command, time_to_go=time_to_go)
        # kill cartesian impedance
        elif blocking:
            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
            try:
                time_to_go = self.adaptive_time_to_go(command)
                self._robot.move_to_joint_positions(command, time_to_go=time_to_go)
            except grpc.RpcError:
                pass

            self._robot.start_cartesian_impedance()

        # NON BLOCKING
        else:

            def helper_non_blocking():
                if not self._robot.is_running_policy():
                    if self.custom_controller:
                        self._start_custom_controller()
                    else:
                        self._robot.start_cartesian_impedance()
                try:
                    if self.custom_controller:
                        self.update_desired_joint_positions(command)
                    else:
                        self._robot.update_desired_joint_positions(command)
                except grpc.RpcError:
                    pass

            run_threaded_command(helper_non_blocking)

    def update_desired_joint_positions(self, joint_pos_desired=None, kp=None, kd=None):
        """update joint pos"""
        udpate_pkt = {}

        if joint_pos_desired is not None:
            udpate_pkt["joint_pos_desired"] = (
                joint_pos_desired if torch.is_tensor(joint_pos_desired) else torch.tensor(joint_pos_desired)
            )

            # # can't update gains when using joint impedance control -> switch to hybrid in the future
            # if kp is not None:
            #     udpate_pkt["Kp"] = kp if torch.is_tensor(kp) else torch.tensor(kp)
            # if kd is not None:
            #     udpate_pkt["Kd"] = kd if torch.is_tensor(kd) else torch.tensor(kd)
            # assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

            self._robot.update_current_policy(udpate_pkt)

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
                joint_pos_desired=waypoints[i]["position"],
                kp=self.reset_gain_scale
                * torch.Tensor(self._robot.metadata.default_Kq),
                kd=self.reset_gain_scale
                * torch.Tensor(self._robot.metadata.default_Kqd),
            )
            time.sleep(dt)

        # reset back gains to gain-policy
        self.update_desired_joint_positions(
            kp=self.gain_scale * torch.Tensor(self._robot.metadata.default_Kq),
            kd=self.gain_scale * torch.Tensor(self._robot.metadata.default_Kqd),
        )

    def update_gripper(self, command, velocity=True, blocking=False):
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        # https://github.com/facebookresearch/fairo/issues/1398
        # for robotiq consider using
        # self._gripper.goto(width=self._max_gripper_width * (1 - command), speed=0.05, force=0.5, blocking=blocking)
        
        # franka gripper
        # goto interface doesn't grasp -> use discrete grasp/ungrasp
        # gripper crashes when running multiple grasp,grasp,grasp,... or ungrasp,ungrasp,ungrasp,... -> use flag
        if command > 0.0 and not self._grasping:
            self._gripper.grasp(
                grasp_width=0.0, speed=0.5, force=5.0, blocking=blocking
            )
            self._grasping = True
        elif command == 0.0 and self._grasping:
            self._gripper.grasp(
                grasp_width=self._max_gripper_width,
                speed=0.5,
                force=5.0,
                blocking=blocking,
            )
            self._grasping = False

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
        return self._robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        return self._robot.get_joint_velocities().tolist()

    def get_gripper_position(self):
        if self._gripper:
            return 1 - (self._gripper.get_state().width / self._max_gripper_width)
        else:
            return 0.0

    def get_ee_pose(self):
        pos, quat = self._robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle])

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_angle(self):
        return self.get_ee_pose()[3:]

    def get_gripper_state(self):
        if self._gripper:
            return self._gripper.get_state().width
        else:
            return 0.0

    def get_robot_state(self):
        robot_state = self._robot.get_robot_state()
        gripper_position = self.get_gripper_position()
        pos, quat = self._robot.robot_model.forward_kinematics(
            torch.Tensor(robot_state.joint_positions)
        )
        cartesian_position = pos.tolist() + quat_to_euler(quat.numpy()).tolist()

        state_dict = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "joint_positions": list(robot_state.joint_positions),
            "joint_velocities": list(robot_state.joint_velocities),
            "joint_torques_computed": list(robot_state.joint_torques_computed),
            "prev_joint_torques_computed": list(
                robot_state.prev_joint_torques_computed
            ),
            "prev_joint_torques_computed_safened": list(
                robot_state.prev_joint_torques_computed_safened
            ),
            "motor_torques_measured": list(robot_state.motor_torques_measured),
            "prev_controller_latency_ms": robot_state.prev_controller_latency_ms,
            "prev_command_successful": robot_state.prev_command_successful,
        }

        timestamp_dict = {
            "robot_timestamp_seconds": robot_state.timestamp.seconds,
            "robot_timestamp_nanos": robot_state.timestamp.nanos,
        }

        return state_dict, timestamp_dict

    def adaptive_time_to_go(self, desired_joint_position, t_min=1, t_max=4):
        curr_joint_position = self._robot.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._robot._adaptive_time_to_go(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go
