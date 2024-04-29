import gym

class CropImageWrapper(gym.Wrapper):

    def __init__(self, env, x_min=0, x_max=None, y_min=0, y_max=None, image_keys=None, crop_render=False):
        super().__init__(env)
        self.image_keys = image_keys
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.crop_render = crop_render

    def crop(self, img):
        return img[self.x_min : self.x_max, self.y_min : self.y_max]

    def reset(self):

        obs = self.env.reset()

        if self.image_keys is not None:
            for key in self.image_keys:
                obs[key] = self.crop(obs[key])

        return obs

    def step(self, act):

        obs, reward, done, info = self.env.step(act)

        if self.image_keys is not None:
            for key in self.image_keys:
                obs[key] = self.crop(obs[key])

        return obs, reward, done, info

    def render(self):
        if self.crop_render:
            return self.crop(self.env.render())
        else:
            return self.env.render()