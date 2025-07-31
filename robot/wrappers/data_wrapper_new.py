import os
import time

import dm_env
import gym
import numpy as np
from dm_env import StepType, TimeStep, specs
from PIL import Image
import json
import shutil
import pickle
import concurrent.futures

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

    def step(self, act, add_noise=False):
        
        # extend obs and push to buffer
        self.curr_obs["language_instruction"] = self.language_instruction

        self.buffer.append(self.curr_obs)
        
        # apply noise to executed action, not to saved one
        if add_noise:
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

    # def save_buffer(self):
    #     assert self.save_dir is not None, "save_dir is not set"
    #     filename = os.path.join(self.save_dir, f"episode_{self.traj_count}.npy")

    #     self.traj_count += 1

    #     np.save(filename, self.buffer)
    #     print(f"Buffer saved to {filename}")
    
    def _get_buffer_dic(self):
        dic = {}
        keys = self.buffer[0].keys()
        for key in keys:
            dic[key] = np.stack([d[key] for d in self.buffer])
        return dic

    def shortest_angle(self, angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def action_preprocessing(self, dic, actions):
            # compute actual deltas s_t+1 - s_t (keep gripper actions)
        actions_tmp = actions.copy()
        actions_tmp[:-1, ..., :6] = (
            dic["lowdim_ee"][1:, ..., :6] - dic["lowdim_ee"][:-1, ..., :6]
        )
        actions = actions_tmp[:-1]
        

            # compute shortest angle -> avoid wrap around
        actions[..., 3:6] = self.shortest_angle(actions[..., 3:6])

        # real data source
        #actions[..., [3,4,5]] = actions[..., [4,3,5]]
        #actions[...,4] = -actions[...,4]
        # actions[...,3] = -actions[...,3] this is a bug

        # print(f'Action min & max: {actions[...,:6].min(), actions[...,:6].max()}')

        return actions

    def save_image(self, img, path, mode=None):
        if mode:
            Image.fromarray(img, mode).save(path)
        else:
            Image.fromarray(img).save(path)
        
    def save_buffer(self):
        dic = self._get_buffer_dic()
        
        actions = dic["action"]
        actions = self.action_preprocessing(dic, actions)  # delta action
        img_paths = {}
        task_name = dic["language_instruction"]

        dir_path = os.path.join(self.save_dir, f'{self.traj_count:06d}')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
                    

        for key in dic.keys():
            # if not "rgb" in key and not "depth" in key:
            #     continue
            # save_dir = os.path.join(self.save_dir, f'{self.traj_count:06d}', key)
            if not "rgb" in key:
                continue
            save_dir = os.path.join(self.save_dir, key, f'{self.traj_count:06d}')
            
            os.makedirs(save_dir, exist_ok=True)

            paths = []
            tasks = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, img in enumerate(dic[key]):
                    img_path = os.path.join(save_dir, f'{i:06d}.png')
                    tasks.append(executor.submit(self.save_image, img, img_path))
                    paths.append(img_path)

                # elif "depth" in key:
                #     for i, depth in enumerate(dic[key]):
                #         depth_uint16 = (depth * 1000).astype(np.uint16)
                #         depth_path = os.path.join(save_dir, f'{i:06d}.png')
                #         tasks.append(executor.submit(self.save_image, depth_uint16, depth_path, 'I;16'))
                #         # paths.append(depth_path)

                # Wait for all parallel tasks to complete
                concurrent.futures.wait(tasks)
                img_paths.setdefault(key, []).extend(paths)

        json_data = []
        for i in range(len(actions)):
            json_data_obs = {
                "task": task_name[i],
                "raw_action": str(actions[i].tolist()),
            }
            json_data_obs["image"] = img_paths["213522250587_rgb"][i]  # front view image path
            json_data.append(json_data_obs)

        json_save_path = os.path.join(self.save_dir, 'unified.json')
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)

        if not os.path.exists(json_save_path):
            with open(json_save_path, "w") as f:
                json.dump(json_data, f, indent=4)
        else:
            # Append new json_data to existing JSON array
            with open(json_save_path, "r") as f:
                existing_data = json.load(f)
            existing_data.extend(json_data)
            with open(json_save_path, "w") as f:
                json.dump(existing_data, f, indent=4)
    
        # save pickle as backup
        rgb_or_depth_keys = [k for k in dic.keys() if "_rgb" in k or "_depth" in k]
        for key in rgb_or_depth_keys:
            del dic[key]
        pickle_save_path = os.path.join(self.save_dir + "_pickle", f'{self.traj_count:06d}.pkl')
        os.makedirs(os.path.dirname(pickle_save_path), exist_ok=True)
        with open(pickle_save_path, 'wb') as f:
            pickle.dump(dic, f)
        print(f"Saved {pickle_save_path}")
        self.traj_count += 1

    def reset_buffer(self):
        self.buffer = []

class MultiTasksDataCollectionWrapper(gym.Wrapper):
    def __init__(self, env, lang_even=None, lang_odd=None, fake_blocking=False, act_noise_std=0., even_savedir=None, odd_savedir=None):
        super().__init__(env)
        self.lang_even = lang_even
        self.lang_odd = lang_odd
        self.fake_blocking = fake_blocking
        self.act_noise_std = act_noise_std
        self.even_savedir = even_savedir
        self.odd_savedir = odd_savedir

        self.buffer = []

        if self.even_savedir is not None:
            os.makedirs(self.even_savedir, exist_ok=True)
        if self.odd_savedir is not None:
            os.makedirs(self.odd_savedir, exist_ok=True)
        self.even_traj_count = 0
        self.odd_traj_count = 0
        
        self.is_even = True

    def reset(self):
        self.reset_buffer()
        obs = self.env.reset()
        # store obs
        self.curr_obs = obs.copy()
        
        return obs

    def step(self, act, add_noise=False):
        
        # extend obs and push to buffer
        self.curr_obs["language_instruction"] = self.lang_even if self.is_even else self.lang_odd
        self.buffer.append(self.curr_obs)
        
        # apply noise to executed action, not to saved one
        if add_noise:
            act[:-1] += np.random.normal(loc=0.0, scale=self.act_noise_std, size=act[:-1].shape)
        
        # binarize gripper
        act[-1] = 1. if act[-1] > 0. else 0.

        obs, reward, done, info = self.env.step(act)

        # overwrite action with actual delta
        if self.fake_blocking:
            act[:-1] = obs["lowdim_ee"][:-1] - self.curr_obs["lowdim_ee"][:-1]
            act[-1] = act[-1]

        self.curr_obs["action"] = act
        self.curr_obs = obs.copy()
        return obs, reward, done, info
    
    def get_buffer(self):
        return self.buffer

    def save_buffer(self):
        assert self.even_savedir is not None or self.odd_savedir is not None, "save_dir is not set"
        
        if self.is_even:
            filename = os.path.join(self.even_savedir, f"episode_{self.even_traj_count}.npy")
            self.even_traj_count += 1
            self.is_even = False
        else:
            filename = os.path.join(self.odd_savedir, f"episode_{self.odd_traj_count}.npy")
            self.odd_traj_count += 1
            self.is_even = True

        np.save(filename, self.buffer)
        print(f"Buffer saved to {filename} with language instruction '{self.buffer[-1]['language_instruction']}'")

    def reset_buffer(self):
        self.buffer = []


