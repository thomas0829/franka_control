# ROBOT SPECIFIC IMPORTS
import os
import time

import grpc
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface

from droid.misc.parameters import sudo_password
from droid.misc.subprocess_utils import run_terminal_command, run_threaded_command

# UTILITY SPECIFIC IMPORTS
from droid.misc.transformations import add_poses, euler_to_quat, pose_diff, quat_to_euler
from droid.robot_ik.robot_ik_solver import RobotIKSolver


class FrankaRobot:
    def launch_controller(self):
        try:
            self.kill_controller()
        except:
            pass

        dir_path = os.path.dirname(os.path.realpath(__file__))

        robot_script_cmd = "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_robot.sh"
        gripper_script_cmd = "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_gripper.sh"
        self._robot_process = run_terminal_command(robot_script_cmd)
        self._gripper_process = run_terminal_command(gripper_script_cmd)
        self._server_launched = True
        print("robot script command: ", robot_script_cmd)
        print("gripper script command: ", gripper_script_cmd)

        time.sleep(5)

    def launch_robot(self):
        self._robot = RobotInterface(ip_address="localhost")
        self._gripper = GripperInterface(ip_address="localhost")
        # self._max_gripper_width = self._gripper.metadata.max_width # robotiq gripper
        self._max_gripper_width = 0.08
        self._ik_solver = RobotIKSolver()
        self._controller_not_loaded = False
        self._grasping = False



    def kill_controller(self):
        self._robot_process.kill()
        self._gripper_process.kill()

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space=None, blocking=False, use_true_velocity=False, model_type=None):
        """
        Update robot command.
        
        Args:
            command: action command
            action_space: one of ["cartesian_velocity", "joint_velocity", "cartesian_position", "joint_position"]
            gripper_action_space: "velocity" or "position"
            blocking: if True, wait for movement to complete
            use_true_velocity: if True and action_space="joint_velocity", use real velocity control
            model_type: "pi0_droid" or "pi05_droid" for model-specific velocity scaling
        """
        action_dict = self.create_action_dict(command, action_space=action_space, gripper_action_space=gripper_action_space)

        # Use true velocity control for joint_velocity if requested
        if action_space == "joint_velocity" and use_true_velocity:
            self.update_joints(action_dict["joint_velocity"], velocity=True, blocking=blocking, use_true_velocity=True, model_type=model_type)
        else:
            # Default behavior: use position control
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

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None, use_true_velocity=False, model_type=None):
        """
        Update joint positions or velocities.
        
        Args:
            command: joint positions or velocities
            velocity: if True, interpret command as velocity
            blocking: if True, wait for movement to complete
            cartesian_noise: optional noise to add
            use_true_velocity: if True and velocity=True, use real velocity control
            model_type: "pi0_droid" or "pi05_droid" for model-specific velocity scaling
        """
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        # Initialize controller tracking
        if not hasattr(self, '_current_controller'):
            self._current_controller = None

        # TRUE VELOCITY CONTROL PATH
        if velocity and use_true_velocity:
            # Convert normalized velocity [-1,1] to rad/s
            # Different models need different scaling:
            # - PI0.5: outputs large values (0.01-0.19), needs 0.5 rad/s
            # - PI0: outputs small values (0.001-0.01), needs ~0.65 rad/s
            if model_type == "pi0_droid":
                max_joint_velocity = torch.Tensor([0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65])
            else:  # pi05_droid or default
                max_joint_velocity = torch.Tensor([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])  # Slower for smoother control
            command_rad_per_sec = command * max_joint_velocity
            
            print(f"[VELOCITY] Rad/s: {command_rad_per_sec.tolist()}")
            
            # Switch to velocity controller if needed
            if self._current_controller != 'velocity':
                print(f"[CONTROLLER] Switching from {self._current_controller} to velocity...")
                
                if self._robot.is_running_policy():
                    try:
                        self._robot.terminate_current_policy()
                        time.sleep(0.1)
                    except grpc.RpcError as e:
                        print(f"[CONTROLLER] Error terminating: {e}")
                
                print("[CONTROLLER] Starting velocity controller...")
                self._robot.start_joint_velocity_control(joint_vel_desired=command_rad_per_sec)
                
                timeout = time.time() + 5
                while not self._robot.is_running_policy():
                    time.sleep(0.01)
                    if time.time() > timeout:
                        print("[CONTROLLER] Timeout, retrying...")
                        self._robot.start_joint_velocity_control(joint_vel_desired=command_rad_per_sec)
                        timeout = time.time() + 5
                
                self._current_controller = 'velocity'
                print("[CONTROLLER] Velocity controller active!")
            
            # Send velocity command
            try:
                self._robot.update_desired_joint_velocities(command_rad_per_sec)
            except grpc.RpcError as e:
                print(f"[VELOCITY] Error: {e}")
                self._current_controller = None
            
            return

        # POSITION CONTROL PATH (for cartesian and blocking modes)
        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        def helper_non_blocking():
            # Switch to cartesian impedance if needed
            if self._current_controller != 'cartesian_impedance':
                print(f"[CONTROLLER] Switching from {self._current_controller} to cartesian impedance...")
                
                if self._robot.is_running_policy():
                    try:
                        self._robot.terminate_current_policy()
                        time.sleep(0.1)
                    except grpc.RpcError as e:
                        print(f"[CONTROLLER] Error terminating: {e}")
                
                print("[CONTROLLER] Starting cartesian impedance...")
                self._robot.start_cartesian_impedance()
                timeout = time.time() + 5
                while not self._robot.is_running_policy():
                    time.sleep(0.01)
                    if time.time() > timeout:
                        self._robot.start_cartesian_impedance()
                        timeout = time.time() + 5
                
                self._current_controller = 'cartesian_impedance'
                print("[CONTROLLER] Cartesian impedance active!")
            
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
            
            # Restart cartesian impedance after blocking move
            self._robot.start_cartesian_impedance()
            self._current_controller = 'cartesian_impedance'
        else:
            helper_non_blocking()

    def update_gripper(self, command, velocity=True, blocking=False):
        print(f"[NUC DEBUG] update_gripper called: command={command}, velocity={velocity}, blocking={blocking}")
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        target_width = self._max_gripper_width * (1 - command)
        print(f"[NUC DEBUG] Calling gripper.goto: target_width={target_width:.4f}, max_width={self._max_gripper_width}, command={command}")
        self._gripper.goto(width=target_width, speed=0.05, force=0.5, blocking=blocking)
        print(f"[NUC DEBUG] gripper.goto completed")
        
    # def update_gripper(self, command, velocity=True, blocking=False):  # Franka Hand version - COMMENTED OUT
    #     if velocity:
    #         gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
    #         command = gripper_delta + self.get_gripper_position()
    #
    #     command = float(np.clip(command, 0, 1))
    #     print("command: ", command)
    #     # https://github.com/facebookresearch/fairo/issues/1398
    #     # for robotiq consider using
    #     # self._gripper.goto(width=self._max_gripper_width * (1 - command), speed=0.05, force=0.5, blocking=blocking)
    #     # franka gripper
    #     # goto interface doesn't grasp -> use discrete grasp/ungrasp
    #     # gripper crashes when running multiple grasp,grasp,grasp,... or ungrasp,ungrasp,ungrasp,... -> use flag
    #     if command > 0.5 and not self._grasping:
    #         self._gripper.grasp(
    #             grasp_width=0.0, speed=0.05, force=0.5, blocking=blocking
    #         )
    #         self._grasping = True
    #         print("grasping...")
    #     elif command <= 0.5 and self._grasping:
    #         self._gripper.grasp(
    #             grasp_width=self._max_gripper_width,
    #             speed=0.05,
    #             force=0.5,
    #             blocking=blocking,
    #         )
    #         self._grasping = False
    #         print("ungrasping...")
    #     print("self._grasping: ", self._grasping)
        


    def add_noise_to_joints(self, original_joints, cartesian_noise):
        original_joints = torch.Tensor(original_joints)

        pos, quat = self._robot.robot_model.forward_kinematics(original_joints)
        curr_pose = pos.tolist() + quat_to_euler(quat).tolist()
        new_pose = add_poses(cartesian_noise, curr_pose)

        new_pos = torch.Tensor(new_pose[:3])
        new_quat = torch.Tensor(euler_to_quat(new_pose[3:]))

        noisy_joints, success = self._robot.solve_inverse_kinematics(new_pos, new_quat, original_joints)

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
        # DROID convention: 0 = closed, 1 = open
        return self._gripper.get_state().width / self._max_gripper_width

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

    def adaptive_time_to_go(self, desired_joint_position, t_min=0, t_max=4):
        curr_joint_position = self._robot.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._robot._adaptive_time_to_go(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        if robot_state is None:
            robot_state = self.get_robot_state()[0]
        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"
        assert gripper_action_space in ["velocity", "position"]
            

        if gripper_action_space == "velocity":
            action_dict["gripper_velocity"] = action[-1]
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
            gripper_position = robot_state["gripper_position"] + gripper_delta
            action_dict["gripper_position"] = float(np.clip(gripper_position, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
            gripper_delta = action_dict["gripper_position"] - robot_state["gripper_position"]
            gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
            action_dict["gripper_delta"] = gripper_velocity

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1]
                cartesian_delta = pose_diff(action[:-1], robot_state["cartesian_position"])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict["cartesian_velocity"], robot_state=robot_state
            ).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()

        if "joint" in action_space:
            # NOTE: Joint to Cartesian has undefined dynamics due to IK
            if velocity:
                action_dict["joint_velocity"] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            else:
                action_dict["joint_position"] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(robot_state["joint_positions"])
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict["joint_velocity"] = joint_velocity.tolist()

        return action_dict
