import gym
import cv2

class ResizeImageWrapper(gym.Wrapper):

    def __init__(self, env, size=(224,224), image_keys=None):
        super().__init__(env)
        self.image_keys = image_keys
        self.size = size

    def resize(self, img):
        return cv2.resize(img, self.size)
    
    def reset(self):

        obs = self.env.reset()

        if self.image_keys is not None:
            for key in self.image_keys:
                obs[key] = self.resize(obs[key])

        return obs

    def step(self, act):

        obs, reward, done, info = self.env.step(act)

        if self.image_keys is not None:
            for key in self.image_keys:
                obs[key] = self.resize(obs[key])

        return obs, reward, done, info
