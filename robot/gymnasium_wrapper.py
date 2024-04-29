import gym
import numpy as np
import dlimp as dl


class OctoPreprocessingWrapper(gym.Wrapper):

    def __init__(self, env, img_keys=["215122255213_rgb"], proprio_keys=["lowdim_ee", "lowdim_qpos"]):
        super().__init__(env)
        self.img_keys = img_keys
        self.proprio_keys = proprio_keys

    def augment_observation(self, obs):
        
        # crop
        img = obs[self.img_keys[0]][:,160:]
        # resize
        img = dl.transforms.resize_image(img, size=(256,256))

        proprio = []
        for k in self.proprio_keys:
            proprio.append(obs[k])

        obs = {
            "proprio": np.concatenate(proprio) if len(proprio) > 1 else proprio[0],
            "image_primary": img,
        }
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.augment_observation(obs), {}

    def step(self, act):
        # TODO somehow octo flips actions ...
        act[...,-1] = -act[...,-1]
        print(act)
        obs, reward, done, info = self.env.step(act)
        trunc = False
        return self.augment_observation(obs), reward, done, trunc, info
