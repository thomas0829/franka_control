"""
Basic Robot Environment Wrapper
Robot Specific Functions: self._update_pose(), self.get_ee_pos(), self.get_ee_angle()
Camera Specific Functions: self.render_obs()
Experiment Specific Functions: self.get_info(), self.get_reward(), self.get_observation()
"""
import torch
import numpy as np
import time
import gym

from helpers.transformations import add_angles, angle_diff
from cameras.multi_camera_wrapper import MultiCameraWrapper
from gym.spaces import Box, Dict


class RobotEnv(gym.Env):
    """
    Main interface to interact with the robot.
    """

    def __init__(
        self,
        # control frequency
        hz=10,
        DoF=3,
        gripper=True,
        robot_model="panda",
        # randomize arm position on reset
        randomize_ee_on_reset=False,
        # allows user to pause to reset reset of the environment
        pause_after_reset=False,
        # observation space configuration
        qpos=True,
        ee_pos=True,
        # pass IP if not running on NUC
        ip_address=None,
        # specify path length if resetting after a fixed length
        max_path_length=None,
        # camera type to use: 'realsense', 'zed'
        camera_ids=[], # [0, 1, 2, ...]
        camera_model="realsense",
        # max vel
        max_lin_vel=0.2,
        max_rot_vel=2.0,
    ):
        # initialize gym environment
        super().__init__()

        # physics
        self.use_desired_pose = False
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.DoF = DoF
        self.gripper = gripper
        self.hz = hz

        self._episode_count = 0
        self._max_path_length = max_path_length
        self._curr_path_length = 0

        self.camera_ids = camera_ids

        # resetting configuration
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._pause_after_reset = pause_after_reset
        self._reset_joint_qpos = np.array(
            [-0.1394, -0.0205, -0.0520, -2.0691,  0.0506,  2.0029, -0.9168]
        )

        # observation space config
        self._qpos = qpos
        self._ee_pos = ee_pos

        # action space
        self.action_space = Box(
            np.array([-1] * (self.DoF + 1)),  # dx_low, dy_low, dz_low, dgripper_low
            np.array([1] * (self.DoF + 1)),  # dx_high, dy_high, dz_high, dgripper_high
        )
        # EE position (x, y, z) + gripper width
        if self.DoF == 3:
            self.ee_space = Box(
                np.array([0.38, -0.25, 0.15, 0.00]),
                np.array([0.70, 0.28, 0.35, 0.085]),
            )

        elif self.DoF == 4:
            # EE position (x, y, z) + EE rot (single axis) + gripper width
            self.ee_space = Box(
                np.array([0.55, -0.06, 0.15, -1.57, 0.00]),
                np.array([0.73, 0.28, 0.35, 0.0, 0.085]),
            )

        # # TODO verify rotation
        elif self.DoF == 6:
            # EE position (x, y, z) + EE rot + gripper width
            self.ee_space = Box(
                np.array([0.55, -0.06, 0.15, -1.57, -1.57, -1.57, 0.00]),
                np.array([0.73, 0.28, 0.35, 0.0, 0.0, 0.0, 0.085]),
            )

        # TODO get actual workspace limits
        # increase workspace height
        self.ee_space.high[2] = 0.8

        # joint limits + gripper
        # https://frankaemika.github.io/docs/control_parameters.html
        if robot_model == "panda":
            self._jointmin = np.array(
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045],
                dtype=np.float32,
            )
            self._jointmax = np.array(
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085],
                dtype=np.float32,
            )
        elif robot_model == "FR3":
            self._jointmin = np.array(
                [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, 0.0045],
                dtype=np.float32,
            )
            self._jointmax = np.array(
                [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 0.085],
                dtype=np.float32,
            )

        # joint space + gripper
        self.qpos_space = Box(self._jointmin, self._jointmax)

        # final observation space configuration
        env_obs_spaces = {
            "lowdim_ee": self.ee_space,
            "lowdim_qpos": self.qpos_space,
        }
        
        for id in self.camera_ids:
            env_obs_spaces[f"img_obs_{id}"] = Box(0, 255, (100, 100, 3), np.uint8)

        if not self._qpos:
            env_obs_spaces.pop("lowdim_qpos", None)
        if not self._ee_pos:
            env_obs_spaces.pop("lowdim_ee", None)
        self.observation_space = Dict(env_obs_spaces)
        print(f"configured observation space: {self.observation_space}")

        # robot configuration
        if ip_address is not None:
            from robot.real.franka import FrankaHardware
            self._robot = FrankaHardware(name=robot_model, ip_address=ip_address, control_hz=self.hz)

            if camera_model == "realsense":
                from cameras.realsense_camera import gather_realsense_cameras
                cameras = gather_realsense_cameras()
            elif camera_model == "zed":
                from cameras.zed_camera import gather_zed_cameras
                cameras = gather_zed_cameras()
            self._camera_reader = MultiCameraWrapper(cameras)
            self.sim = False
        else:
            from robot.sim.mujoco.franka import FrankaMujoco
            self._robot = FrankaMujoco()
            self.sim = True

    def step(self, action):
        start_time = time.time()

        assert len(action) == (self.DoF + 1)
        assert (action.max() <= 1) and (action.min() >= -1)

        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        # clipping + any safety corrections for position
        desired_pos, gripper = self._get_valid_pos_and_gripper(
            self._curr_pos + lin_vel, gripper
        )
        desired_angle = add_angles(rot_vel, self._curr_angle)
        if self.DoF == 4:
            desired_angle[2] = desired_angle[2].clip(
                self.ee_space.low[3], self.ee_space.high[3]
            )
        elif self.DoF == 6:
            desired_angle = desired_angle.clip(
                self.ee_space.low[3:6], self.ee_space.high[3:6]
            )
        self._update_robot(desired_pos, desired_angle, gripper)

        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        time.sleep(sleep_left)
        obs = self.get_observation()

        self._curr_path_length += 1
        done = False
        if (
            self._max_path_length is not None
            and self._curr_path_length >= self._max_path_length
        ):
            done = True
        return obs, 0.0, done, {}

    def normalize_ee_obs(self, obs):
        """Normalizes low-dim obs between [-1,1]."""
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2 + min(x)
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        normalized_obs = (
            2 * (obs - self.ee_space.low) / (self.ee_space.high - self.ee_space.low) - 1
        )
        return normalized_obs

    def unnormalize_ee_obs(self, obs):
        return (obs + 1) * (
            self.ee_space.high - self.ee_space.low
        ) / 2 + self.ee_space.low

    def normalize_qpos(self, qpos):
        """Normalizes qpos between [-1,1]."""
        # The ranges for the joint limits are taken from
        # the franka emika page: https://frankaemika.github.io/docs/control_parameters.html
        norm_qpos = (
            2
            * (qpos - self.qpos_space.low)
            / (self.qpos_space.high - self.qpos_space.low)
            - 1
        )
        return norm_qpos

    def reset_gripper(self):
        self._robot.update_gripper(0)

    def reset(self):

        self.reset_gripper()
        for _ in range(5):
            if self.sim:
                self._robot.update_joints(self._reset_joint_qpos)
            else:
                self._robot.update_joints_slow(torch.tensor(self._reset_joint_qpos))
            if self.is_robot_reset():
                break
            else:
                print('reset failed, trying again')

        # fix default angle at first joint reset
        if self._episode_count == 0:
            self._default_angle = self._robot.get_ee_angle()

        if self._randomize_ee_on_reset:
            self._desired_pose = {
                "position": self._robot.get_ee_pos(),
                "angle": self._robot.get_ee_angle(),
                "gripper": 1,
            }
            self._randomize_reset_pos()
            time.sleep(1)

        if self._pause_after_reset:
            user_input = input(
                "Enter (s) to wait 5 seconds & anything else to continue: "
            )
            if user_input in ["s", "S"]:
                time.sleep(5)

        # initialize desired pose correctly for env.step
        self._desired_pose = {
            "position": self._robot.get_ee_pos(),
            "angle": self._robot.get_ee_angle(),
            "gripper": 1,
        }

        self._curr_path_length = 0
        self._episode_count += 1

        return self.get_observation()

    def _format_action(self, action):
        """Returns [x,y,z], [yaw, pitch, roll], close_gripper"""
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = (
                action[:-1],
                default_delta_angle,
                action[-1],
            )
        elif self.DoF == 4:
            delta_pos, delta_angle, gripper = (
                action[:3],
                [default_delta_angle[0], default_delta_angle[1], action[3]],
                action[-1],
            )
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
        lin_vel, rot_vel = (
            lin_vel * self.max_lin_vel / self.hz,
            rot_vel * self.max_rot_vel / self.hz,
        )
        return lin_vel, rot_vel

    def _get_valid_pos_and_gripper(self, pos, gripper):
        """To avoid situations where robot can break the object / burn out joints,
        allowing us to specify (x, y, z, gripper) where the robot cannot enter. Gripper is included
        because (x, y, z) is different when gripper is open/closed.

        There are two ways to do this: (a) reject action and maintain current pose or (b) project back
        to valid space. Rejecting action works, but it might get stuck inside the box if no action can
        take it outside. Projection is a hard problem, as it is a non-convex set :(, but we can follow
        some rough heuristics."""

        # clip commanded position to satisfy box constraints
        x_low, y_low, z_low = self.ee_space.low[:3]
        x_high, y_high, z_high = self.ee_space.high[:3]
        pos[0] = pos[0].clip(x_low, x_high)  # new x
        pos[1] = pos[1].clip(y_low, y_high)  # new y
        pos[2] = pos[2].clip(z_low, z_high)  # new z

        return pos, gripper

    def _update_robot(self, pos, angle, gripper):
        """input: the commanded position (clipped before).
        feasible position (based on forward kinematics) is tracked and used for updating,
        but the real position is used in observation."""
        feasible_pos, feasible_angle = self._robot.update_pose(pos, angle)
        self._robot.update_gripper(gripper)
        self._desired_pose = {
            "position": feasible_pos,
            "angle": feasible_angle,
            "gripper": gripper,
        }

    @property
    def _curr_pos(self):
        if self.use_desired_pose:
            return self._desired_pose["position"].copy()
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        if self.use_desired_pose:
            return self._desired_pose["angle"].copy()
        return self._robot.get_ee_angle()

    def get_images(self):
        if self.sim:
            return self._robot.render()
        else:
            return self._camera_reader.read_cameras()
    
    def get_state(self):
        state_dict = {}
        gripper_state = self._robot.get_gripper_state()

        state_dict["control_key"] = (
            "desired_pose" if self.use_desired_pose else "current_pose"
        )

        state_dict["desired_pose"] = np.concatenate(
            [
                self._desired_pose["position"],
                self._desired_pose["angle"],
                [self._desired_pose["gripper"]],
            ]
        )

        state_dict["current_pose"] = np.concatenate(
            [self._robot.get_ee_pos(), self._robot.get_ee_angle(), [gripper_state]]
        )

        state_dict["joint_positions"] = self._robot.get_joint_positions()
        state_dict["joint_velocities"] = self._robot.get_joint_velocities()
        # don't track gripper velocity
        state_dict["gripper_velocity"] = 0

        return state_dict

    def _randomize_reset_pos(self):
        """takes random action along x-y plane, no change to z-axis / gripper"""
        random_xy = np.random.uniform(-0.5, 0.5, (2,))
        random_z = np.random.uniform(-0.2, 0.2, (1,))
        if self.DoF == 4:
            random_rot = np.random.uniform(-0.5, 0.0, (1,))
            act_delta = np.concatenate(
                [random_xy, random_z, random_rot, np.zeros((1,))]
            )
        elif self.DoF == 6:
            random_rot = np.random.uniform(-0.5, 0.0, (3,))
            act_delta = np.concatenate(
                [random_xy, random_z, *random_rot, np.zeros((1,))]
            )
        else:
            act_delta = np.concatenate([random_xy, random_z, np.zeros((1,))])
        for _ in range(10):
            self.step(act_delta)

    def get_observation(self):
        # get state and images
        current_state = self.get_state()
        current_images = self.get_images()

        # set gripper width
        gripper_width = current_state["current_pose"][-1:]
        # compute and normalize ee/qpos state
        if self.DoF == 3:
            ee_pos = np.concatenate([current_state["current_pose"][:3], gripper_width])
        elif self.DoF == 4:
            ee_pos = np.concatenate(
                [
                    current_state["current_pose"][:3],
                    current_state["current_pose"][5:6],
                    gripper_width,
                ]
            )
        elif self.DoF == 6:
            ee_pos = np.concatenate(
                [
                    current_state["current_pose"][:6],
                    gripper_width,
                ]
            )
        qpos = np.concatenate([current_state["joint_positions"], gripper_width])
        normalized_ee_pos = self.normalize_ee_obs(ee_pos)
        normalized_qpos = self.normalize_qpos(qpos)

        obs_dict = {
            "lowdim_ee": normalized_ee_pos,
            "lowdim_qpos": normalized_qpos,
        }

        for id in self.camera_ids:
            obs_dict[f"img_obs_{id}"] = current_images[id]["array"]

        if not self._qpos:
            obs_dict.pop("lowdim_qpos", None)
        if not self._ee_pos:
            obs_dict.pop("lowdim_ee", None)

        return obs_dict

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self._reset_joint_qpos)
        return joint_dist < epsilon

    @property
    def num_cameras(self):
        return len(self.get_images())
