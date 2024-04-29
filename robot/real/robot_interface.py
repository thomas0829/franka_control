# ROBOT SPECIFIC IMPORTS
import os
import time

import grpc
import numpy as np
import torch

from utils.transformations import *

# from r2d2.misc.subprocess_utils import run_terminal_command, run_threaded_command
import multiprocessing
import subprocess
import threading


def run_terminal_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, executable="/bin/bash", encoding="utf8"
    )
    
    return process


def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread


def run_multiprocessed_command(command, args=()):
    process = multiprocessing.Process(target=command, args=args)
    process.start()

    return process

class RobotInterfaceServer:
    def __init__(self, ip_address=""):
        self.launch_robot()
        self._grasping = False

    def launch_controller(self):
        try:
            self.kill_controller()
        except:
            pass

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # self._robot_process = run_terminal_command(
        #     "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_robot.sh"
        # )
        # self._gripper_process = run_terminal_command(
        #     "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_gripper.sh"
        # )
        self._server_launched = True
        time.sleep(5)

    def launch_robot(self):
        from polymetis import GripperInterface, RobotInterface
        
        self._robot = RobotInterface(ip_address="localhost")
        self._gripper = GripperInterface(ip_address="localhost")
        self._max_gripper_width = self._gripper.metadata.max_width

    def kill_controller(self):
        self._robot_process.kill()
        self._gripper_process.kill()

    def update_command(self, command, action_space="cartesian_velocity", blocking=False):
        action_dict = self.create_action_dict(command, action_space=action_space)

        self.update_joints(action_dict["joint_position"], velocity=False, blocking=blocking)
        self.update_gripper(action_dict["gripper_position"], velocity=False, blocking=blocking)

        return action_dict

    def update_pose(self, command, velocity=False, blocking=False):
        if blocking:
            if velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(command)
                command = add_poses(cartesian_delta, curr_pose)

            pos = torch.Tensor(command[:3])
            quat = torch.Tensor(euler_to_quat(command[3:6]))
            curr_joints = self._robot.get_joint_positions()
            desired_joints = self._robot.solve_inverse_kinematics(pos, quat, curr_joints)
            self.update_joints(desired_joints, velocity=False, blocking=True)
        else:
            if not velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = pose_diff(command, curr_pose)
                command = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)

            robot_state = self.get_robot_state()[0]
            joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(command, robot_state=robot_state)

            self.update_joints(joint_velocity, velocity=True, blocking=False)

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        def helper_non_blocking():
            if not self._robot.is_running_policy():
                self._robot.start_cartesian_impedance()
            try:
                self._robot.update_desired_joint_positions(command)
            except grpc.RpcError:
                pass

        if blocking:
            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
            try:
                time_to_go = self.adaptive_time_to_go(command)
                self._robot.move_to_joint_positions(command, time_to_go=time_to_go)
            except grpc.RpcError:
                pass

            self._robot.start_cartesian_impedance()
        else:
            run_threaded_command(helper_non_blocking)

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
                grasp_width=0.0, speed=0.05, force=0.1, blocking=blocking
            )
            self._grasping = True
        elif command == 0.0 and self._grasping:
            self._gripper.grasp(
                grasp_width=self._max_gripper_width,
                speed=0.05,
                force=0.1,
                blocking=blocking,
            )
            self._grasping = False

    def update_desired_joint_positions(self, command):
        command = torch.Tensor(command)
        self._robot.update_desired_joint_positions(command)

    def move_to_joint_positions(self, command, time_to_go=None):
        if time_to_go is None:
            time_to_go = self.adaptive_time_to_go(command)
        self._robot.move_to_joint_positions(command, time_to_go=time_to_go)

    def move_to_ee_pose(self, pos, quat, time_to_go=None):
        self._robot.move_to_ee_pose(pos, quat, time_to_go=time_to_go)
        
    def get_joint_positions(self):
        return self._robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        return self._robot.get_joint_velocities().tolist()

    def get_gripper_state(self):
        return self._gripper.get_state().width
    
    def get_gripper_position(self):
        return 1 - (self._gripper.get_state().width / self._max_gripper_width)

    def get_ee_pose(self):
        pos, quat = self._robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle]).tolist()

    def get_robot_state(self):
        robot_state = self._robot.get_robot_state()
        gripper_position = self.get_gripper_position()
        pos, quat = self._robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
        cartesian_position = pos.tolist() + quat_to_euler(quat.numpy()).tolist()

        state_dict = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "joint_positions": list(robot_state.joint_positions),
            "joint_velocities": list(robot_state.joint_velocities),
            "joint_torques_computed": list(robot_state.joint_torques_computed),
            "prev_joint_torques_computed": list(robot_state.prev_joint_torques_computed),
            "prev_joint_torques_computed_safened": list(robot_state.prev_joint_torques_computed_safened),
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
