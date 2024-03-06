import gym
import numpy as np
import dlimp as dl


class OctoPreprocessingWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def augment_observation(self, obs):
        
        # crop
        img = obs["215122255213_rgb"][:,160:]
        # resize
        img = dl.transforms.resize_image(img, size=(256,256))

        obs = {
            "proprio": np.concatenate((obs["lowdim_ee"], obs["lowdim_qpos"])),
            "image_primary": img,
        }
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.augment_observation(obs), {}

    def step(self, act):
        obs, reward, done, info = self.env.step(act)
        trunc = False
        return self.augment_observation(obs), reward, done, trunc, info
