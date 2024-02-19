import copy

import gym
import mujoco
import numpy as np

from utils.transformations_mujoco import euler_to_quat_mujoco


class ASIDWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        verbose=False,
    ):
        super(ASIDWrapper, self).__init__(env)

        from robot.sim.mujoco.obj_wrapper import ObjWrapper

        assert type(env) is ObjWrapper, "Environment must be wrapped in ObjWrapper!"
        self.verbose = verbose

        if self.env.DoF == 2:
            self.env.unwrapped._reset_joint_qpos = np.array(
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

        # Mujoco object ids
        self.obj_geom_ids = mujoco.mj_name2id(
            self.env.unwrapped._robot.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.obj_id}_geom",
        )
        # if single geom not found, check for multiple geoms
        if self.obj_geom_ids == -1:
            self.obj_geom_ids = []
            for i in range(5):
                obj_geom_id = mujoco.mj_name2id(
                    self.env.unwrapped._robot.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{self.obj_id}_geom_{i}",
                )
                if obj_geom_id == -1:
                    break
                self.obj_geom_ids.append(obj_geom_id)

        if -1 in self.obj_geom_ids:
            self.obj_geom_ids = mujoco.mj_name2id(
                self.env.unwrapped._robot.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{self.obj_id}_geom",
            )

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

    def step(self, action):

        self.last_action = action

        if self.reward_first:
            reward = self.compute_reward(action)

        obs, _, done, info = self.env.step(action)

        if not self.reward_first:
            reward = self.compute_reward(action)

        return obs, reward, done, info

    def reset(self, *args, **kwargs):

        # randomize obj parameters | mujoco reset data
        self.resample_parameters()

        obs = self.env.reset()

        return obs

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
                    self.env.unwrapped._robot.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{self.obj_id}_com",
                )
                self.env.unwrapped._robot.model.body_pos[com_body_ids][1] = value

            elif key == "friction":
                for geom_id in self.obj_geom_ids:
                    self.env.unwrapped._robot.model.geom_friction[geom_id][0] = value

        if self.verbose:
            print(
                f"Parameters: {self.get_parameters()} - seed {self.env.unwrapped._seed}"
            )

        # update sim
        mujoco.mj_resetData(
            self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
        )
        mujoco.mj_setConst(
            self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
        )

    def set_data(self, new_data):
        self.env.unwrapped._robot.data = new_data
        mujoco.mj_forward(
            self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
        )

    def get_data(self):
        return copy.deepcopy(self.env.unwrapped._robot.data)

    def get_full_state(self):
        full_state = {}
        full_state["last_action"] = self.last_action.copy()
        full_state["curr_path_length"] = copy.copy(self.env.curr_path_length)
        full_state["robot_data"] = self.get_data()
        return full_state

    def set_full_state(self, full_state):
        self.last_action = full_state["last_action"]
        self.env.curr_path_length = full_state["curr_path_length"]
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

        from asid.wrapper.asid_reward import ASIDRewardWrapper

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
