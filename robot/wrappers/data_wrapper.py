import os
import time

import dm_env
import gym
import numpy as np
from dm_env import StepType, TimeStep, specs


def wrap_env_in_rlds_logger(env, exp, save_dir, max_episodes_per_shard=1):

    import envlogger
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from envlogger.backends import tfds_backend_writer

    rlds_env = RLDSWrapper(env)
    rlds_env.action_spec()
    rlds_env.observation_spec()

    def step_fn(unused_timestep, unused_action, unused_env):
        return {"timestamp_ns": time.time_ns()}

    # observation specs
    obs_dict = {}
    for k in rlds_env.observation_spec().keys():
        if "rgb" in k or "depth" in k:
            obs_dict[k] = tfds.features.Image(
                shape=rlds_env.observation_spec()[k].shape,
                dtype=rlds_env.observation_spec()[k].dtype,
            )
        else:
            obs_dict[k] = tfds.features.Tensor(
                shape=rlds_env.observation_spec()[k].shape,
                dtype=rlds_env.observation_spec()[k].dtype,
            )
    obs_feat_dict = tfds.features.FeaturesDict(obs_dict)

    # action specs
    act_feat_dict = tfds.features.Tensor(
        shape=rlds_env.action_spec().shape, dtype=rlds_env.action_spec().dtype
    )

    # dataset specs
    ds_config = tfds.rlds.rlds_base.DatasetConfig(
        name=exp,
        observation_info=obs_feat_dict,
        action_info=act_feat_dict,
        reward_info=np.float64,
        discount_info=np.float64,
        step_metadata_info={"timestamp_ns": np.int64},
    )

    # create logger
    rlds_env_logger = envlogger.EnvLogger(
        rlds_env,
        backend=tfds_backend_writer.TFDSBackendWriter(
            data_directory=save_dir,
            split_name="train",
            max_episodes_per_file=max_episodes_per_shard,
            ds_config=ds_config,
        ),
        step_fn=step_fn,
    )

    return rlds_env_logger


def load_rlds_dataset(save_dir):

    import tensorflow_datasets as tfds

    loaded_dataset = tfds.builder_from_directory(save_dir).as_dataset(split="all")
    for e in loaded_dataset:
        print(f"Trajectory of length {len(e['steps'])}")
    return loaded_dataset


def convert_rlds_to_np(save_dir):

    import rlds
    import tensorflow as tf
    import tensorflow_datasets as tfds

    trajs = []
    builder = tfds.builder_from_directory(save_dir).as_dataset(
        decoders={rlds.STEPS: tfds.decode.SkipDecoding()}
    )
    for e in builder["train"]:
        traj_np = tf.data.Dataset.from_tensors(e[rlds.STEPS]).as_numpy_iterator().next()
        trajs.append(traj_np)
        print(traj_np["is_first"].shape)

    return trajs


class RLDSWrapper(gym.Wrapper, dm_env.Environment):

    def __init__(self, env):
        super().__init__(env)

    def type_action(self, act):
        act = act.astype(self.action_spec().dtype)
        return act

    def type_observation(self, obs):
        for k in obs.keys():
            obs[k] = obs[k].astype(self.observation_spec()[k].dtype)
        return obs

    def reset(self):
        obs = self.env.reset()

        reward = 0.0
        discount = 0.0
        step_type = StepType.FIRST

        return TimeStep(step_type, reward, discount, self.type_observation(obs))

    def step(self, act):

        obs, reward, done, info = self.env.step(act)

        step_type = StepType.LAST if done else StepType.MID
        discount = 0.0

        return TimeStep(step_type, reward, discount, self.type_observation(obs))

    def observation_spec(self):
        """Returns the observation spec."""
        obs_spec = {}
        for k in self.env.observation_space.keys():
            obs_spec[k] = specs.BoundedArray(
                shape=self.env.observation_space[k].shape,
                dtype=self.env.observation_space[k].dtype,
                name=k,
                minimum=self.env.observation_space[k].low,
                maximum=self.env.observation_space[k].high,
            )
        return obs_spec

    def action_spec(self):
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype,
            name="action",
            minimum=self.env.action_space.low,
            maximum=self.env.action_space.high,
        )


class DataCollectionWrapper(gym.Wrapper):

    def __init__(self, env, language_instruction=None, fake_blocking=False, act_noise_std=0., save_dir=None):
        super().__init__(env)
        self.buffer = []
        self.language_instruction = language_instruction
        self.fake_blocking = fake_blocking
        self.act_noise_std = act_noise_std

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.traj_count = 0

    def reset(self):
        
        self.reset_buffer()

        obs = self.env.reset()
        
        # self.gripper_state = self.env.unwrapped._robot.get_gripper_state()

        # store obs
        self.curr_obs = obs.copy()
        
        return obs

    def step(self, act):
        
        # extend obs and push to buffer
        self.curr_obs["language_instruction"] = self.language_instruction

        self.buffer.append(self.curr_obs)
        
        # apply noise to executed action, not to saved one
        act[:-1] += np.random.normal(loc=0.0, scale=self.act_noise_std, size=act[:-1].shape)
        # binarize gripper
        act[-1] = 1. if act[-1] > 0. else 0.

        obs, reward, done, info = self.env.step(act)

        # overwrite action with actual delta
        if self.fake_blocking:
            act[:-1] = obs["lowdim_ee"][:-1] - self.curr_obs["lowdim_ee"][:-1]
            act[-1] = act[-1]
            
        # ############################ -> buggy af
        # # pause until gripper is done
        # max_gripper_width = 0.08 if act[-1] == 1. else 0.0
        
        # start = time.time()
        # while True:
        #     gripper_state_curr = self.env.unwrapped._robot.get_gripper_state()
        #     # print("States", self.gripper_state, gripper_state_curr, np.abs(self.gripper_state - gripper_state_curr))
        #     # if gripper is moving, sleep for control cycle
        #     # print("Gripper is moving", np.abs(self.gripper_state - gripper_state_curr))
        #     if np.abs(self.gripper_state - gripper_state_curr) > 1e-4:
        #         self.gripper_state = gripper_state_curr
        #         time.sleep(0.2)
        #     else:
        #         break
        # # print(f"Waited for gripper for {time.time() - start} seconds")
        # # get new observation
        # obs = self.env.unwrapped.get_observation()
        # ############################

        self.curr_obs["action"] = act

        self.curr_obs = obs.copy()

        return obs, reward, done, info
    
    def get_buffer(self):
        return self.buffer

    def save_buffer(self):
        assert self.save_dir is not None, "save_dir is not set"
        filename = os.path.join(self.save_dir, f"episode_{self.traj_count}.npy")

        self.traj_count += 1

        np.save(filename, self.buffer)
        print(f"Buffer saved to {filename}")

    def reset_buffer(self):
        self.buffer = []

