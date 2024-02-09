import time

import gym
import dm_env
from dm_env import specs, TimeStep, StepType


def wrap_env_in_rlds_logger(env, exp, save_dir, max_episodes_per_shard=1):

    import envlogger
    from envlogger.backends import tfds_backend_writer
    import tensorflow as tf
    import tensorflow_datasets as tfds

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
        reward_info=tf.float64,
        discount_info=tf.float64,
        step_metadata_info={"timestamp_ns": tf.int64},
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
    builder = tfds.builder_from_directory(save_dir).as_dataset(decoders={rlds.STEPS: tfds.decode.SkipDecoding()})
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
