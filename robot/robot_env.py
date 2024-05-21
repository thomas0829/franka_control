import json
import time

import gym
import numpy as np
# import torch
from gym.spaces import Box, Dict

from utils.pointclouds import (compute_camera_extrinsic,
                               compute_camera_intrinsic, crop_points,
                               depth_to_points, points_to_pcd,
                               read_calibration_file, visualize_pcds)
from utils.transformations import add_angles, angle_diff


class RobotEnv(gym.Env):
    """
    Main interface to interact with the robot.
    """

    def __init__(
        self,
        # control frequency
        control_hz=10,
        blocking_control=False,
        DoF=3,
        gripper=True,
        # Franka model: 'panda', 'fr3'
        robot_type="panda",
        # randomize arm position on reset
        randomize_ee_on_reset=0.0,
        # allows user to pause to reset reset of the environment
        pause_after_reset=False,
        # observation space configuration
        qpos=True,
        ee_pos=True,
        imgs=True,
        normalize=False,
        # pass IP if not running on NUC, "localhost" if running on NUC, None if running sim
        ip_address=None,
        # specify path length if resetting after a fixed length
        max_path_length=None,
        # cameras to use in sim
        camera_names=["front"],
        camera_rgb=True,
        camera_depth=False,
        # camera type to use: 'realsense', 'zed'
        camera_model="realsense",
        camera_resolution=None,  # (128, 128) -> HxW
        calibration_file=None,
        # Mujoco: model name
        model_name="base_franka",
        on_screen_rendering=False,
        visual_dr=False,
        device_id=0,
        # debugging
        verbose=False,
    ):
        # initialize gym environment
        super().__init__()

        self.verbose = verbose

        # physics
        self.DoF = DoF
        self.gripper = gripper
        self.control_hz = control_hz
        self.blocking_control = blocking_control

        self._episode_count = 0
        self._max_path_length = max_path_length
        self.curr_path_length = 0

        # resetting configuration
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._set_randomize_ee_on_reset(randomize_ee_on_reset)

        self._pause_after_reset = pause_after_reset
        # polymetis _robot.home_pose
        self._reset_joint_qpos = np.array(
            # [-0.1394, -0.0205, -0.0520, -2.0691, 0.0506, 2.0029, -0.9168]
            [
                -0.13677763938903809,
                0.006021707784384489,
                -0.048125553876161575,
                -2.0723488330841064,
                -0.021774671971797943,
                2.0718562602996826,
                0.5588430762290955,
            ]
        )

        self._reset_joint_qpos = np.array([-5.65335140e-05, -1.47445112e-01,  5.44415554e-03, -2.57991934e+00  , 2.13176832e-02,  2.43316126e+00,  7.82760382e-01])

        if self.DoF == 2:
            self._reset_joint_qpos = np.array(
                # [
                #     -0.06315325,
                #     # 0.33202057,
                #     0.27,
                #     -0.0462324,
                #     -2.79372462,
                #     0.07651035,
                #     3.18670704,
                #     0.44067877,
                # ]
                [
                    0.85290707,
                    0.29776727,
                    0.0438237,
                    -2.70994978,
                    -0.00481878,
                    2.89241547,
                    1.67766532,
                ]
            )

        # observation space config
        self._qpos = qpos
        self._ee_pos = ee_pos
        self._imgs = imgs
        self.normalize = normalize

        # action space
        # action_low, action_high = -1., 1.
        # TODO this limits rotation (euler) -> increase for angle!
        action_low, action_high = -0.1, 0.1
        self.action_space = Box(
            np.array(
                [action_low] * (self.DoF + 1 if self.gripper else self.DoF),
                dtype=np.float32,
            ),  # dx_low, dy_low, dz_low, dgripper_low
            np.array(
                [action_high] * (self.DoF + 1 if self.gripper else self.DoF),
                dtype=np.float32,
            ),  # dx_high, dy_high, dz_high, dgripper_high
        )
        self.action_shape = self.action_space.shape

        # EE position (x, y, z) + EE rot (roll, pitch, yaw) + gripper width
        ee_space_low = np.array([0.12, -1.0, 0.11, -np.pi, -np.pi, -np.pi, 0.00])
        ee_space_high = np.array([1.0, 1.0, 0.7, np.pi, np.pi, np.pi, 0.085])

        # EE position (x, y, fixed z)
        if self.DoF == 2:
            ee_space_low = ee_space_low[:3]
            ee_space_high = ee_space_high[:3]
        # EE position (x, y, z)
        if self.DoF == 3:
            ee_space_low = ee_space_low[:3]
            ee_space_high = ee_space_high[:3]
        # EE position (x, y, z) + EE rot (single axis)
        elif self.DoF == 4:
            ee_space_low = np.concatenate((ee_space_low[:3], ee_space_low[5:6]))
            ee_space_high = np.concatenate((ee_space_high[:3], ee_space_high[5:6]))
        # EE position (x, y, z) + EE rot
        elif self.DoF == 6:
            ee_space_low = ee_space_low[:6]
            ee_space_high = ee_space_high[:6]

        # gripper width
        if self.gripper:
            ee_space_low = np.concatenate((ee_space_low, ee_space_low[-1:]))
            ee_space_high = np.concatenate((ee_space_high, ee_space_high[-1:]))

        self.ee_space = Box(
            low=np.float32(ee_space_low), high=np.float32(ee_space_high)
        )

        # joint limits + gripper
        # https://frankaemika.github.io/docs/control_parameters.html
        if robot_type == "panda":
            self._jointmin = np.array(
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045],
                dtype=np.float32,
            )
            self._jointmax = np.array(
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085],
                dtype=np.float32,
            )
        elif robot_type == "fr3":
            self._jointmin = np.array(
                [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, 0.0045],
                dtype=np.float32,
            )
            self._jointmax = np.array(
                [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 0.085],
                dtype=np.float32,
            )

        # robot configuration
        self.camera_resolution = camera_resolution

        # TODO move to RobotBase
        if calibration_file:
            calib_dict = (
                read_calibration_file(calibration_file)
                if calibration_file is not None
                else None
            )
        else:
            calib_dict = None

        if ip_address is not None:
            from robot.real.franka import FrankaHardware

            self._robot = FrankaHardware(
                robot_type=robot_type,
                gripper=self.gripper,
                ip_address=ip_address,
                control_hz=self.control_hz,
            )

            if camera_model == "realsense":
                from perception.cameras.realsense_camera import \
                    gather_realsense_cameras

                cameras = gather_realsense_cameras(hardware_reset=False)
            elif camera_model == "zed":
                from perception.cameras.zed_camera import gather_zed_cameras

                cameras = gather_zed_cameras()

            else:
                cameras = []

            from perception.cameras.multi_camera_wrapper import \
                MultiCameraWrapper

            self._camera_reader = MultiCameraWrapper(cameras)

            # TODO move to RobotHardware
            if calib_dict is not None:
                self.camera_intrinsic = {}
                self.camera_extrinsic = {}
                for sn in calib_dict.keys():
                    self.camera_intrinsic[sn] = compute_camera_intrinsic(
                        calib_dict[sn]["intrinsic"]["fx"],
                        calib_dict[sn]["intrinsic"]["fy"],
                        calib_dict[sn]["intrinsic"]["ppx"],
                        calib_dict[sn]["intrinsic"]["ppy"],
                    )
                    self.camera_extrinsic[sn] = compute_camera_extrinsic(
                        pos=np.array(calib_dict[sn]["extrinsic"]["pos"]).reshape(-1),
                        ori=calib_dict[sn]["extrinsic"]["ori"],
                    )
            self.depth_scale = 1000.0

            self.sim = False

        else:
            from robot.sim.mujoco.franka import MujocoManipulatorEnv

            self._robot = MujocoManipulatorEnv(
                robot_type=robot_type,
                model_name=model_name,
                control_hz=self.control_hz,
                has_renderer=on_screen_rendering,
                has_offscreen_renderer=not on_screen_rendering,
                calib_dict=calib_dict,
                use_rgb=camera_rgb,
                use_depth=camera_depth,
                camera_names=camera_names,
                visual_dr=visual_dr,
            )

            # TODO move to MujocoManipulatorEnv
            # self.camera_intrinsic = {}
            # self.camera_extrinsic = {}
            # for cn in self._robot.camera_names:
            #     self.camera_intrinsic[cn] = self._robot.get_camera_intrinsic(cn)
            #     self.camera_extrinsic[cn] = self._robot.get_camera_extrinsic(cn)
            self.depth_scale = 1.0

            self.sim = True

        # joint space + gripper
        self.qpos_space = Box(self._jointmin, self._jointmax)

        # final observation space configuration
        env_obs_spaces = {}

        if self._qpos:
            env_obs_spaces["lowdim_ee"] = self.ee_space
        if self._ee_pos:
            env_obs_spaces["lowdim_qpos"] = self.qpos_space

        if self._imgs:
            imgs = self.get_images()
            if len(imgs) > 0:
                for sn, img in imgs.items():
                    for m, modality in img.items():
                        if m == "rgb":
                            env_obs_spaces[f"{sn}_{m}"] = Box(
                                0, 255, modality.shape, np.uint8
                            )
                        elif m == "depth":
                            env_obs_spaces[f"{sn}_{m}"] = Box(
                                0, 65535, modality.shape, np.uint16
                            )
                        elif m == "points":
                            pass

        self.observation_space = Dict(env_obs_spaces)

        self.observation_shape = {}
        self.observation_type = {}
        for k in env_obs_spaces.keys():
            self.observation_shape[k] = env_obs_spaces[k].shape
            self.observation_type[k] = env_obs_spaces[k].dtype

        self._seed = 0

    def get_spaces(self):
        return self.observation_space, self.action_space

    def step(self, action):
        start_time = time.time()

        if not self.gripper:
            assert len(action) == (
                self.DoF
            ), f"Expected action shape: ({self.DoF},) got {action.shape}"
        else:
            assert len(action) == (
                self.DoF + 1
            ), f"Expected action shape: ({self.DoF+1},) got {action.shape}"
        
        # BLOCKING CONTROL -> for BC inference
        if self.blocking_control:
            
            # keep track of desired pose in case controller drops actions
            pos_action, angle_action, gripper = self._format_action(action)
            
            self._init_pos += pos_action
            self._init_angle = add_angles(angle_action, self._init_angle) # 
            
            gripper = gripper
            
            # cartesian position control w/ blocking
            self._update_robot(
                # np.concatenate((self._curr_pos + pos_action, add_angles(angle_action, self._curr_angle), [gripper])),
                np.concatenate((self._init_pos, self._init_angle, [gripper])),
                action_space="cartesian_position",
                blocking=False,
            )
            
            comp_time = time.time() - start_time
            sleep_left = max(0, (1 / self.control_hz) - comp_time)
            if not self.sim:
                time.sleep(sleep_left)

        # NON BLOCKING CONTROL -> for everything else
        else:
    
            # clip action to action space
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # formate action to DoF
            pos_action, angle_action, gripper = self._format_action(action)
            

            # clipping + any safety corrections for position
            desired_pos = self._get_valid_pos(self._curr_pos + pos_action)
            desired_angle = add_angles(angle_action, self._curr_angle)

            # cartesian position control
            self._update_robot(
                np.concatenate((desired_pos, desired_angle, [gripper])),
                action_space="cartesian_position",
                blocking=False,
            )

            # sleep to maintain control_hz
            comp_time = time.time() - start_time
            sleep_left = max(0, (1 / self.control_hz) - comp_time)
            if not self.sim:
                time.sleep(sleep_left)
        # self.desired_pos = desired_pos

        # get observations
        obs = self.get_observation()

        self.curr_path_length += 1
        done = False
        if (
            self._max_path_length is not None
            and self.curr_path_length >= self._max_path_length
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
        self._robot.update_gripper(0.0, velocity=False, blocking=True)

    def reset(self):

        if self.sim and self._robot.visual_dr:
            self._robot.randomize()

        # ensure robot releases grasp before reset
        if self.gripper:
            self.reset_gripper()
        else:
            # default is closed gripper if not self.gripper
            self._robot.update_gripper(1.0, velocity=False, blocking=True)
        # reset to home pose
        for _ in range(3):
            self._robot.update_joints(
                self._reset_joint_qpos.tolist(), velocity=False, blocking=True
            )

            epsilon = 0.1
            is_reset, joint_dist = self.is_robot_reset(epsilon=epsilon)

            if is_reset:
                break
            else:
                if self.verbose:
                    print(
                        f"WARNING: reset failed w/ joint_dist={np.round(joint_dist,4)} > {epsilon}, trying again"
                    )
                if not self.sim:
                    time.sleep(1.0)

        # fix default pos and angle at first joint reset
        if self._episode_count == 0:
            self._default_pos = self._robot.get_ee_pos()
            self._default_angle = self._robot.get_ee_angle()

            # overwrite fixed z for 2DoF EE control with reset z
            if self.DoF == 2:
                self.ee_space.low[2] = self._default_pos[2]
                self.ee_space.high[2] = self._default_pos[2]

                # overwrite fixed z for 2DoF EE control with 0.13
                self.ee_space.low[2] = 0.13
                self.ee_space.high[2] = 0.13

        if self.blocking_control:
            self._init_pos = self._default_pos.copy()
            self._init_angle = self._default_angle.copy()

        if self._randomize_ee_on_reset:
            self._randomize_reset_pos()
            if not self.sim:
                time.sleep(1)

        if self._pause_after_reset:
            user_input = input(
                "Enter (s) to wait 5 seconds & anything else to continue: "
            )
            if user_input in ["s", "S"]:
                time.sleep(5)

        self.curr_path_length = 0
        self._episode_count += 1

        return self.get_observation()

    def _format_action(self, action):
        """Returns [x,y,z], [yaw, pitch, roll], close_gripper"""
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)

        if self.DoF == 2:
            delta_pos, delta_angle = (
                np.concatenate(
                    (action[:2], self._default_pos[2:]),
                ),
                default_delta_angle,
            )
        if self.DoF == 3:
            delta_pos, delta_angle = (
                action[:3],
                default_delta_angle,
            )
        elif self.DoF == 4:
            delta_pos, delta_angle = (
                action[:3],
                [default_delta_angle[0], default_delta_angle[1], action[3]],
            )
        elif self.DoF == 6:
            delta_pos, delta_angle = action[:3], action[3:6]

        if self.gripper:
            gripper = action[-1]
        else:
            # default is closed gripper if not self.gripper
            gripper = 1.0
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _get_valid_pos(self, pos):
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

        return pos

    def _update_robot(self, action, action_space, blocking=False):
        assert action_space in [
            "cartesian_position",
            "joint_position",
            "cartesian_velocity",
            "joint_velocity",
        ]
        action_info = self._robot.update_command(
            action, action_space=action_space, blocking=blocking
        )
        return action_info

    @property
    def _curr_pos(self):
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        return self._robot.get_ee_angle()

    @property
    def _num_cameras(self):
        if self.sim:
            return len(self._robot.camera_names)
        else:
            return len(self._camera_reader._all_cameras)

    def render(self, mode=None, sn=None):
        if self.sim and self._robot.has_renderer:
            self._robot.render()
        else:
            imgs = self.get_images()
            if sn is None:
                sn = next(iter(imgs))
            return imgs[sn]["rgb"]

    def get_images(self):
        imgs = []
        if self.sim and not self._robot.has_renderer:
            imgs = self._robot.render()
        else:
            imgs = self._camera_reader.read_cameras()

        img_dict = {}
        for img in imgs:
            sn = img["serial_number"].split("/")[0]

            if img_dict.get(sn) is None:
                img_dict[sn] = {}

            if img["type"] == "depth":
                img_dict[sn]["depth"] = img["array"]
            elif img["type"] == "rgb":
                img_dict[sn]["rgb"] = img["array"]

        return img_dict

    def get_state(self):
        state_dict = {}
        if self.gripper:
            gripper_state = self._robot.get_gripper_state()

        state_dict["control_key"] = "current_pose"

        state_dict["current_pose"] = (
            np.concatenate(
                [self._robot.get_ee_pos(), self._robot.get_ee_angle(), [gripper_state]]
            )
            if self.gripper
            else np.concatenate([self._robot.get_ee_pos(), self._robot.get_ee_angle()])
        )

        state_dict["joint_positions"] = self._robot.get_joint_positions()
        state_dict["joint_velocities"] = self._robot.get_joint_velocities()
        # don't track gripper velocity
        state_dict["gripper_velocity"] = 0

        return state_dict

    def get_images_and_points(self):
        img_dict = self.get_images()

        # TODO move to MujocoManipulatorEnv
        if self.sim:
            self.camera_intrinsic = {}
            self.camera_extrinsic = {}
            for cn in self._robot.camera_names:
                self.camera_intrinsic[cn] = self._robot.get_camera_intrinsic(cn)
                self.camera_extrinsic[cn] = self._robot.get_camera_extrinsic(cn)

        for sn, img in img_dict.items():
            img_dict[sn]["points"] = depth_to_points(
                img["depth"],
                self.camera_intrinsic[sn],
                self.camera_extrinsic[sn],
                depth_scale=self.depth_scale,
            )

        return img_dict

    def show_points(self):
        print("WARNING: mujoco rendering and open3d don't like each other :(")
        img_points = self.get_images_and_points()
        points = []
        for k in img_points.keys():
            pts, clr = crop_points(
                img_points[k]["points"],
                colors=img_points[k]["rgb"].reshape(-1, 3) / 255.0,
                crop_min=[-1.0, -1.0, -1.0],
                crop_max=[1.0, 1.0, 1.0],
            )
            points.append(points_to_pcd(pts, colors=clr))

        x = np.zeros((1, 3))
        for d in np.arange(0, 1, 0.1):
            x[:, 0] = d
            points.append(points_to_pcd(x, colors=[[255.0, 0.0, 0.0]]))
            y = np.zeros((1, 3))
            y[:, 1] = d
            points.append(points_to_pcd(y, colors=[[0.0, 255.0, 0.0]]))
            z = np.zeros((1, 3))
            z[:, 2] = d
            points.append(points_to_pcd(z, colors=[[0.0, 0.0, 255.0]]))
        # points.append(zero)

        visualize_pcds(points)

    def _set_randomize_ee_on_reset(self, randomize_ee_on_reset):
        self.xy_min_max = randomize_ee_on_reset
        self.z_min_max = randomize_ee_on_reset
        self.random_rot_min = randomize_ee_on_reset

    def _randomize_reset_pos(self):
        """takes random action along x-y plane, no change to z-axis / gripper"""
        random_xy = np.random.uniform(-self.xy_min_max, self.xy_min_max, (2,))
        random_z = np.random.uniform(-self.z_min_max, self.z_min_max, (1,))
        
        if self.DoF == 4:
            random_rot = np.random.uniform(-self.random_rot_min, 0.0, (1,))
            act_delta = np.concatenate(
                [random_xy, random_z, random_rot, np.zeros((1,))]
            )
        elif self.DoF == 6:
            random_rot = np.random.uniform(-self.random_rot_min, 0.0, (3,))
            act_delta = np.concatenate(
                [random_xy, random_z, random_rot, np.zeros((1,))]
            )
        else:
            act_delta = np.concatenate([random_xy, random_z, np.zeros((1,))])
        for _ in range(10):
            self.step(act_delta)

    def get_observation(self):
        # get state and images
        current_state = self.get_state()

        # set gripper width
        gripper_width = current_state["current_pose"][-1:]
        # compute and normalize ee/qpos state
        # if self.DoF == 2:
        #     ee_pos = (
        #         np.concatenate([current_state["current_pose"][:2], gripper_width])
        #         if self.gripper
        #         else current_state["current_pose"][:2]
        #     )
        if self.DoF == 3 or self.DoF == 2:
            ee_pos = (
                np.concatenate([current_state["current_pose"][:3], gripper_width])
                if self.gripper
                else current_state["current_pose"][:3]
            )
        elif self.DoF == 4:
            ee_pos = (
                np.concatenate(
                    [
                        current_state["current_pose"][:3],
                        current_state["current_pose"][5:6],
                        gripper_width,
                    ]
                )
                if self.gripper
                else np.concatenate(
                    [
                        current_state["current_pose"][:3],
                        current_state["current_pose"][5:6],
                    ]
                )
            )
        elif self.DoF == 6:
            ee_pos = (
                np.concatenate(
                    [
                        current_state["current_pose"][:6],
                        gripper_width,
                    ]
                )
                if self.gripper
                else current_state["current_pose"][:6]
            )

        qpos = np.concatenate([current_state["joint_positions"], gripper_width])

        obs_dict = {}

        if self._qpos:
            obs_dict["lowdim_qpos"] = (
                self.normalize_ee_obs(qpos) if self.normalize else qpos
            )
        if self._ee_pos:
            obs_dict["lowdim_ee"] = (
                self.normalize_ee_obs(ee_pos) if self.normalize else ee_pos
            )

        if self._imgs:
            current_images = self.get_images()
            if len(current_images) > 0:
                for sn, img in current_images.items():
                    for m, modality in img.items():
                        obs_dict[f"{sn}_{m}"] = modality
         
            #obs_dict[f"{sn}_rgb_pcd"] = self.get_images_and_points()  #rgb point cloud

        return obs_dict

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self._reset_joint_qpos)
        return joint_dist < epsilon, joint_dist

    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)
        # torch.manual_seed(seed)
