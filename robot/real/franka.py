""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import time

import numpy as np
import torch

from polymetis import RobotInterface, GripperInterface
from robot.franka_base import FrankaBase
from robot.real.inverse_kinematics.robot_ik_solver import RobotIKSolver

# from robohive.robot.hardware_base import hardwareBase
# from robohive.utils.min_jerk import generate_joint_space_min_jerk

from robot.controllers.mixed_cartesian_impedance import MixedCartesianImpedanceControl

from helpers.transformations import quat_to_euler, euler_to_quat

import argparse

from robot.controllers.utils import generate_joint_space_min_jerk

# Adapted from : https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/python/torchcontrol/planning/min_jerk.py

class FrankaHardware(FrankaBase):
    def __init__(
        self,
        name,
        ip_address,
        gain_scale=1.0,
        reset_gain_scale=1.0,
        control_hz=10,
        **kwargs,
    ):
        self.name = name
        self.ip_address = ip_address
        self.robot = None
        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale
        self.control_hz = control_hz

        success = self.connect()
        assert success, "Failed to connect to hardware!"

    def connect(self, policy=None):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = RobotInterface(
                ip_address=self.ip_address, enforce_version=False
            )
            self.gripper = GripperInterface(ip_address=self.ip_address)
            self._max_gripper_width = self.gripper.metadata.max_width
            self._ik_solver = RobotIKSolver(self, control_hz=self.control_hz)
            print("Success")
        except Exception as e:
            self.robot = None  # declare dead
            print("Failed with exception: ", e)
            return connection

        print("Testing {} connection: ".format(self.name), end="")
        connection = self.okay()
        if connection:
            print("okay")
            self.reset()  # reset the robot before starting operaions
            if policy == None:
                # Create policy instance
                # s_initial = self.get_sensors()
                # policy = JointPDPolicy(
                #     desired_joint_pos=s_initial["joint_pos"],
                #     kp=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
                #     kd=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kqd),
                # )

                policy = MixedCartesianImpedanceControl(
                    joint_pos_current=self.robot.get_joint_positions(),
                    Kp=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kx),
                    Kd=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kxd),
                    kp_pos=self.gain_scale
                    * torch.Tensor(self.robot.metadata.default_Kq),
                    kd_pos=self.gain_scale
                    * torch.Tensor(self.robot.metadata.default_Kqd),
                    desired_joint_pos=self.robot.get_joint_positions(),
                    robot_model=self.robot.robot_model,
                    ignore_gravity=True,
                )

            # Send policy
            print(f"\nRunning {str(type(policy))} policy...")
            self.robot.send_torch_policy(policy, blocking=False)
        else:
            print("Not ready. Please retry connection")

        return connection

    def okay(self):
        """Return hardware health"""
        okay = False
        if self.robot:
            try:
                state = self.robot.get_robot_state()
                delay = time.time() - (
                    state.timestamp.seconds + 1e-9 * state.timestamp.nanos
                )
                assert delay < 5, "Acquired state is stale by {} seconds".format(delay)
                okay = True
            except:
                self.robot = None  # declare dead
                okay = False
        return okay

    def close(self):
        """Close hardware connection"""
        if self.robot:
            print("Terminating PD policy: ", end="")
            try:
                self.reset()
                state_log = self.robot.terminate_current_policy()
                print("Success")
            except:
                # print("Failed. Resetting directly to home: ", end="")
                print("Resetting Failed. Exiting: ", end="")
            self.robot = None
            print("Done")
        return True

    def reconnect(self):
        print("Attempting re-connection")
        self.connect()
        while not self.okay():
            self.connect()
            time.sleep(2)
        print("Re-connection success")

    def reset(self, reset_pos=None, time_to_go=5):
        """Reset hardware"""

        if self.okay():
            if self.robot.is_running_policy():  # Is user controller?
                print("Resetting using user controller")

                if reset_pos == None:
                    reset_pos = torch.Tensor(self.robot.metadata.rest_pose)
                elif not torch.is_tensor(reset_pos):
                    reset_pos = torch.Tensor(reset_pos)
                self.update_joints_slow(reset_pos)

            else:
                # Use default controller
                print("Resetting using default controller")
                self.robot.go_home(time_to_go=time_to_go)
        else:
            print(
                "Can't connect to the robot for reset. Attemping reconnection and trying again"
            )
            self.reconnect()
            self.reset(reset_pos, time_to_go)

    def _solve_ik(self, desired_pos, desired_euler):
        desired_q, success = self._ik_solver.compute(desired_pos, desired_euler)
        assert success, "IK failed to compute"
        return desired_q

    def update_pose(self, pos=None, angle=None, kp=None, kd=None):
        """update EE pose"""
        udpate_pkt = {}
        udpate_pkt["ctrl_mode"] = torch.Tensor([1.0])

        if pos is not None:
            udpate_pkt["ee_pos_desired"] = (
                pos if torch.is_tensor(pos) else torch.tensor(pos)
            )
        if angle is not None:
            if len(angle) == 3:
                angle = euler_to_quat(angle)
            udpate_pkt["ee_quat_desired"] = (
                angle if torch.is_tensor(angle) else torch.tensor(angle)
            )
        if kp is not None:
            udpate_pkt["kp"] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt["kd"] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        try:
            self.robot.update_current_policy(udpate_pkt)
        except Exception as e:
            print("1> Failed to udpate policy with exception", e)
            self.reconnect()

        return self.robot.get_ee_pose()
        # return feasible_pos, feasible_angle

    def update_joints(self, q_desired=None, kp=None, kd=None):
        """update joint pos"""
        udpate_pkt = {}
        udpate_pkt["ctrl_mode"] = torch.Tensor([0.0])

        if q_desired is not None:
            udpate_pkt["q_desired"] = (
                q_desired if torch.is_tensor(q_desired) else torch.tensor(q_desired)
            )
        if kp is not None:
            udpate_pkt["kp"] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt["kd"] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        try:
            self.robot.update_current_policy(udpate_pkt)
        except Exception as e:
            print("1> Failed to udpate policy with exception", e)
            self.reconnect()

        # return feasible_pos, feasible_angle

    def update_joints_slow(self, q_target, time_to_go=5):
        # Use registered controller
        q_current = self.robot.get_joint_positions()

        # generate min jerk trajectory
        dt = 0.1
        waypoints = generate_joint_space_min_jerk(
            start=q_current, goal=q_target, time_to_go=time_to_go, dt=dt
        )
        # reset using min_jerk traj
        for i in range(len(waypoints)):
            self.update_joints(
                q_desired=waypoints[i]["position"],
                kp=self.reset_gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
                kd=self.reset_gain_scale
                * torch.Tensor(self.robot.metadata.default_Kqd),
            )
            time.sleep(dt)

        # reset back gains to gain-policy
        self.update_joints(
            kp=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
            kd=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kqd),
        )

    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy()

    def get_ee_pose(self):
        pos, angle = self.robot.get_ee_pose()
        return pos.numpy(), quat_to_euler(angle.numpy())

    def get_ee_pos(self):
        return self.get_ee_pose()[0]

    def get_ee_angle(self):
        return self.get_ee_pose()[1]

    def get_gripper_position(self):
        return 1 - (self.gripper.get_state().width / self._max_gripper_width)

    def get_gripper_state(self):
        return self.gripper.get_state().width

    def update_gripper(self, flag):
        if flag < 0:
            desired_gripper = 0.00
        elif flag >= 0:
            desired_gripper = 0.085
        self.gripper.goto(width=desired_gripper, speed=0.1, force=1000)

    def __del__(self):
        self.close()

# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Polymetis based Franka client")

    parser.add_argument(
        "-i",
        "--server_ip",
        type=str,
        help="IP address or hostname of the franka server",
        default="172.16.0.1",
    )  # 10.0.0.123 # "169.254.163.91",
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # user inputs
    time_to_go = 1 * np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    franka = FrankaArm(name="Franka-Demo", ip_address=args.server_ip)

    # connect to robot with default policy
    assert franka.connect(policy=None), "Connection to robot failed."

    # reset using the user controller
    franka.reset()

    # Update policy to execute a sine trajectory on joint 6 for 5 seconds
    print("Starting sine motion updates...")
    q_initial = franka.robot.get_joint_positions().clone()
    q_desired = franka.robot.get_joint_positions().clone()

    for i in range(int(time_to_go * hz)):
        q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
        # q_desired[5] = q_initial[5] + 0.05*np.random.uniform(high=1, low=-1)
        # q_desired = q_initial + 0.01*np.random.uniform(high=1, low=-1, size=7)
        franka.update_joints(q_desired=q_desired)
        time.sleep(1 / hz)

    # Udpate the gains
    kp_new = 0.1 * torch.Tensor(franka.robot.metadata.default_Kq)
    kd_new = 0.1 * torch.Tensor(franka.robot.metadata.default_Kqd)
    franka.update_joints(kp=kp_new, kd=kd_new)

    print("Starting sine motion updates again with updated gains.")
    for i in range(int(time_to_go * hz)):
        q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
        franka.update_joints(q_desired=q_desired)
        time.sleep(1 / hz)

    print("Closing and exiting hardware connection")
    franka.close()
