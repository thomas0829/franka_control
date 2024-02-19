import copy

import gym
import mujoco
import numpy as np

from utils.transformations_mujoco import euler_to_quat_mujoco


class ASIDWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obj_id="rod",
        obj_pos_noise=True,
        obs_keys=["lowdim_ee", "lowdim_qpos", "obj_pose"],
        flatten=True,
        verbose=False,
    ):
        super().__init__(env)

        self.verbose = verbose

        if self.env.DoF == 2:
            self.env._reset_joint_qpos = np.array(
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

        self.obj_id = obj_id

        # Mujoco object ids
        self.obj_body_id = mujoco.mj_name2id(
            self.env._robot.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.obj_id}_body"
        )
        self.obj_joint_id = mujoco.mj_name2id(
            self.env._robot.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.obj_id}_freejoint"
        )

        self.obj_geom_ids = mujoco.mj_name2id(
            self.env._robot.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.obj_id}_geom",
        )
        # if single geom not found, check for multiple geoms
        if self.obj_geom_ids == -1:
            self.obj_geom_ids = []
            for i in range(5):
                obj_geom_id = mujoco.mj_name2id(
                    self.env._robot.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{self.obj_id}_geom_{i}",
                )
                if obj_geom_id == -1:
                    break
                self.obj_geom_ids.append(obj_geom_id)

        if -1 in self.obj_geom_ids:
            self.obj_geom_ids = mujoco.mj_name2id(
                self.env._robot.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{self.obj_id}_geom",
            )

        # assert (
        #     self.obj_body_id != -1
        #     and self.obj_joint_id != -1
        #     and -1 not in self.obj_geom_ids
        # ), f"Object not found. Make sure RobotEnv(model_name) is passed!"

        # Object position
        self.obj_pose_noise_dict = {
            "x": {"min": 0.0, "max": 0.1},
            # "x": {"min": -0.1, "max": 0.1},
            "y": {"min": -0.1, "max": 0.1},
            "yaw": {"min": 0.0, "max": 3.14},
        }
        self.obj_pos_noise = obj_pos_noise
        self.init_obj_pose = self.get_obj_pose()
        self.curr_obj_pose = None

        # Physics parameters
        self.parameter_dict = {
            "inertia": {"type": "uniform", "min": -0.1, "max": 0.1, "value": None},
            # careful when adjusting friction -> too high values cause the object to penetrate the table and give huge reward signals even if not touched
            "friction": {"type": "gaussian", "mean": 1.0, "std": 0.1, "value": None},
        }

        self.reset_parameters()
        self.resample_parameters()

        # Exploration reward
        self.last_action = np.zeros(
            self.env.DoF if not self.env.gripper else self.env.DoF + 1
        )
        self.reward_first = True
        self.exp_reward = None

        # Observations
        self.obs_keys = obs_keys
        # obs space dict to array
        for k in copy.deepcopy(self.env.observation_space.keys()):
            if k not in self.obs_keys:
                del self.env.observation_space.spaces[k]

        self.flatten = flatten
        obj_pose_low = -np.inf * np.ones(7)
        obj_pose_high = np.inf * np.ones(7)
        if self.flatten:
            low = np.concatenate([v.low for v in self.env.observation_space.values()])
            high = np.concatenate([v.high for v in self.env.observation_space.values()])
            # add obj pose
            low = np.concatenate([low, obj_pose_low])
            high = np.concatenate([high, obj_pose_high])
            # overwrite observation_space
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=low.shape)
        else:
            self.observation_space["obj_pose"] = gym.spaces.Box(
                low=obj_pose_low, high=obj_pose_high
            )

    def augment_observations(self, obs, flatten=True):
        obs["obj_pose"] = self.get_obj_pose()

        if flatten:
            tmp = []
            for k in self.obs_keys:
                tmp.append(obs[k])
            obs = np.concatenate(tmp)

        return obs

    def step(self, action):

        self.last_action = action

        if self.reward_first:
            reward = self.compute_reward(action)

        obs, _, done, info = self.env.step(action)

        if not self.reward_first:
            reward = self.compute_reward(action)

        return self.augment_observations(obs, flatten=self.flatten), reward, done, info

    def reset(self, *args, **kwargs):

        # randomize obj parameters | mujoco reset data
        self.resample_parameters()

        # randomize obj position |
        self.resample_obj_pose()

        if self.curr_obj_pose is None:
            obj_pose = self.init_obj_pose.copy()
        else:
            obj_pose = self.curr_obj_pose.copy()
        # set obj qpos | mujoco forward
        self.update_obj(obj_pose)

        # reset robot |
        obs = self.env.reset()

        return self.augment_observations(obs, flatten=self.flatten)

    def set_parameters(self, parameters):
        assert parameters.shape == self.get_parameters().shape
        if type(parameters) is dict:
            for k in parameters.key():
                self.parameter_dict[k]["value"] = parameters[k]
        else:
            for k, v in zip(self.parameter_dict.keys(), parameters):
                self.parameter_dict[k]["value"] = v
        self.params_set = True
        self.reset()

    def get_parameters(self):
        parameters = []
        for k in self.parameter_dict.keys():
            parameters.append(self.parameter_dict[k]["value"])
        return np.array(parameters)

    def get_parameters_distribution(self):
        return self.parameter_dict

    def set_parameters_distribution(self, parameter_dict):
        self.parameter_dict = parameter_dict

    def reset_parameters(self):
        self.params_set = False

    # def reset_task(self, task=None):
    #     self.resample_parameters()

    def resample_parameters(self):
        for key in self.parameter_dict:
            # sample new parameter value
            if self.params_set:
                value = self.parameter_dict[key]["value"]
            elif self.parameter_dict[key]["type"] == "uniform":
                value = np.random.uniform(
                    low=self.parameter_dict[key]["min"],
                    high=self.parameter_dict[key]["max"],
                )
            elif self.parameter_dict[key]["type"] == "gaussian":
                value = np.random.normal(
                    loc=self.parameter_dict[key]["mean"],
                    scale=self.parameter_dict[key]["std"],
                )
            self.parameter_dict[key]["value"] = value

            # set new parameter value
            if key == "inertia":
                com_body_ids = mujoco.mj_name2id(
                    self.env._robot.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{self.obj_id}_com",
                )
                self.env._robot.model.body_pos[com_body_ids][1] = value

            elif key == "friction":
                for geom_id in self.obj_geom_ids:
                    self.env._robot.model.geom_friction[geom_id][0] = value

        if self.verbose:
            print(f"Parameters: {self.get_parameters()} - seed {self.env._seed}")

        # update sim
        mujoco.mj_resetData(self.env._robot.model, self.env._robot.data)
        mujoco.mj_setConst(self.env._robot.model, self.env._robot.data)

    def get_obj_pose(self):
        obj_pos = self.env._robot.data.qpos[
            self.obj_joint_id : self.obj_joint_id + 3
        ].copy()
        obj_quat = self.env._robot.data.qpos[
            self.obj_joint_id + 3 : self.obj_joint_id + 7
        ].copy()
        return np.concatenate((obj_pos, obj_quat))

    def set_obj_pose(self, obj_pose):
        self.obj_pos_noise = False
        self.init_obj_pose = obj_pose.copy()
        self.update_obj(obj_pose)

    def resample_obj_pose(self):
        pose = self.init_obj_pose.copy()
        if self.obj_pos_noise:
            pose[0] += np.random.uniform(
                self.obj_pose_noise_dict["x"]["min"],
                self.obj_pose_noise_dict["x"]["max"],
            )
            pose[1] += np.random.uniform(
                self.obj_pose_noise_dict["y"]["min"],
                self.obj_pose_noise_dict["y"]["max"],
            )
            pose[3:7] = euler_to_quat_mujoco(
                [
                    0.0,
                    0.0,
                    np.random.uniform(
                        self.obj_pose_noise_dict["yaw"]["min"],
                        self.obj_pose_noise_dict["yaw"]["max"],
                        size=1,
                    ).item(),
                ]
            )
        if self.verbose:
            print(f"Object pose: {pose} - seed {self.env._seed}")
        self.curr_obj_pose = pose.copy()

    def update_obj(self, qpos):
        self.env._robot.data.qpos[self.obj_joint_id : self.obj_joint_id + 3] = qpos[:3]
        self.env._robot.data.qpos[self.obj_joint_id + 3 : self.obj_joint_id + 7] = qpos[
            3:
        ]
        mujoco.mj_forward(self.env._robot.model, self.env._robot.data)

    def set_data(self, new_data):
        self.env._robot.data = new_data
        mujoco.mj_forward(self.env._robot.model, self.env._robot.data)

    def get_data(self):
        return copy.deepcopy(self.env._robot.data)

    def get_full_state(self):
        full_state = {}
        full_state["last_action"] = self.last_action.copy()
        full_state["curr_path_length"] = copy.copy(self.env._curr_path_length)
        full_state["robot_data"] = self.get_data()
        return full_state

    def set_full_state(self, full_state):
        self.last_action = full_state["last_action"]
        self.env._curr_path_length = full_state["curr_path_length"]
        self.set_data(full_state["robot_data"])

    def create_exp_reward(
        self,
        env_func,
        robot_cfg_dict,
        env_cfg_dict,
        seed,
        device_id=0,
        delta=5e-2,
        normalization=1e-3,
    ):
        robot_cfg_dict["on_screen_rendering"] = False
        env_cfg_dict["obj_pos_noise"] = False

        exp_env = env_func(
            robot_cfg_dict,
            env_cfg_dict,
            seed=seed,
            device_id=device_id,
            asid_wrapper=True,
            asid_reward=False,
            verbose=False,
        )

        from asid.wrapper.sim.asid_reward import ASIDRewardWrapper

        self.exp_reward = ASIDRewardWrapper(
            exp_env,
            delta=delta,
            normalization=normalization,
        )

    def compute_reward(self, action):
        if self.exp_reward:
            full_state = self.get_full_state()
            current_param = self.get_parameters()
            return self.exp_reward.get_reward(
                full_state,
                action,
                params=current_param,
                verbose=False,
            )
        else:
            return 0.0
