import gym
import copy
import mujoco

import numpy as np

from helpers.transformations_mujoco import euler2quat


class ASIDWrapper(gym.Wrapper):
    def __init__(
        self, env, obj_id="rod", obs_keys=["lowdim_ee", "lowdim_qpos", "obj_pose"]
    ):
        super().__init__(env)

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
                self.obj_geom_ids.append(
                    mujoco.mj_name2id(
                        self.env._robot.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"{self.obj_id}_geom_{i}",
                    )
                )

        if -1 in self.obj_geom_ids:
            self.obj_geom_ids = mujoco.mj_name2id(
                self.env._robot.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{self.obj_id}_geom",
            )

        assert (
            self.obj_body_id != -1
            and self.obj_joint_id != -1
            and -1 not in self.obj_geom_ids
        ), f"Object not found. Make sure RobotEnv(model_name) is passed!"

        # Object position
        self.obj_pose_noise_dict = {
            "x": {"min": -0.1, "max": 0.1},
            "y": {"min": -0.2, "max": 0.2},
            "yaw": {"min": 0.0, "max": 3.14},
        }
        self.obj_pos_noise = True
        self.init_obj_pose = self.get_obj_pose()
        self.curr_obj_pose = None

        # Physics parameters
        self.parameter_dict = {
            "inertia": {"type": "uniform", "min": -0.12, "max": 0.12, "value": None},
            "friction": {"type": "gaussian", "mean": 0.0, "std": 0.1, "value": None},
        }

        self.reset_parameters()
        self.resample_parameters()

        # Exploration reward
        self.last_action = np.zeros(
            self.env.DoF - 1 if not self.env.gripper else self.env.DoF
        )
        self.reward_first = True
        self.exp_reward = None

        # Observations
        self.obs_keys = obs_keys

    def augment_observations(self, obs):
        obs["obj_pose"] = self.get_obj_pose()

        tmp = []
        for k in self.obs_keys:
            tmp.append(obs[k])
        obs = np.concatenate(tmp)

        return obs

    def step(self, action):
        if self.reward_first:
            reward = self.compute_reward(action)

        obs, _, done, info = self.env.step(action)

        if not self.reward_first:
            reward = self.compute_reward(action)

        return self.augment_observations(obs), reward, done, info

    def reset(self, *args, **kwargs):
        # reset robot
        obs = self.env.reset()

        # randomize obj parameters
        self.resample_parameters()

        # randomize obj position
        self.resample_obj_pose()

        if self.curr_obj_pose is None:
            obj_pose = self.init_obj_pose.copy()
        else:
            obj_pose = self.curr_obj_pose.copy()
        self.update_obj(obj_pose)

        return self.augment_observations(obs)

    def set_parameters(self, parameters):
        assert parameters.shape == self.get_parameters().shape
        if type(parameters) is dict:
            for k in parameters.key():
                self.parameter_dict[k]["value"] = parameters[k]
        else:
            for k, v in zip(self.parameter_dict.keys(), parameters):
                self.parameter_dict[k]["value"] = v
        self.params_set = True

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

    def reset_task(self, task=None):
        self.resample_parameters()

    def resample_parameters(self):
        for key in self.parameter_dict:
            # sample new parameter value
            if self.params_set:
                value = self.parameter_dict[key]["value"]
            elif self.parameter_dict[key]["type"] == "uniform":
                value = np.random.uniform(
                    self.parameter_dict[key]["min"], self.parameter_dict[key]["max"]
                )
            elif self.parameter_dict[key]["type"] == "gaussian":
                value = np.random.normal(
                    self.parameter_dict[key]["mean"], self.parameter_dict[key]["std"]
                )
            self.parameter_dict[key]["value"] = value

            # set new parameter value
            if key == "inertia":
                self.env._robot.model.body_ipos[self.obj_body_id][1] = value
                self.env._robot.model.body_inertia[self.obj_body_id] = np.array(
                    [0.0002, 0.0002, 0.0002]
                )
            elif key == "friction":
                for geom in self.obj_geom_ids:
                    self.env._robot.model.geom_friction[geom][0] = value

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
            pose[3:7] = euler2quat(
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
        self.curr_obj_pose = pose.copy()

    def update_obj(self, qpos):
        self.env._robot.data.qpos[self.obj_joint_id : self.obj_joint_id + 3] = qpos[:3]
        self.env._robot.data.qpos[self.obj_joint_id + 3 : self.obj_joint_id + 7] = qpos[
            3:
        ]
        mujoco.mj_forward(self.env._robot.model, self.env._robot.data)

    def reset_data(self, new_data):
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
        self.reset_data(full_state["robot_data"])

    def create_exp_reward(self, cfg, seed, normalization=0.02):
        exp_env = type(self.env)(**cfg)
        exp_env = ASIDWrapper(exp_env)
        exp_env.seed(seed)
        exp_env.reset()
        # exp_env._robot.set_noiseless()
        from robot.sim.mujoco.asid_reward import ASIDRewardWrapper

        self.exp_reward = ASIDRewardWrapper(
            exp_env,
            normalization=normalization,
            articulation=False,
        )

    def compute_reward(self, action):
        if self.exp_reward:
            full_state = self.get_full_state()
            current_param = self.get_parameters()
            return self.exp_reward.get_reward(
                full_state,
                action[: self.env.DoF],
                params=current_param,
                verbose=False,
            )
        else:
            return 0.0
