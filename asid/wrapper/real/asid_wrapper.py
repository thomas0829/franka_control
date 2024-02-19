import gym
import numpy as np

from perception.trackers.color_tracker import ColorTracker
from utils.pointclouds import crop_points


class ASIDWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obj_id="rod",
        obs_keys=["lowdim_ee", "lowdim_qpos", "obj_pose"],
        # tracking
        color_track="red",
        crop_min=[0.0, -0.4, -0.1],
        crop_max=[0.5, 0.4, 0.5],
        filter=False,
        cutoff_freq=1,
        verbose_track=False,
    ):
        super().__init__(env)

        self.obj_id = obj_id

        # Tracking
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.tracker = ColorTracker(outlier_removal=True)
        self.color_track = color_track
        self.verbose_track = verbose_track
        self.filter = filter
        self.cutoff_freq = cutoff_freq

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
        obs, reward, done, info = self.env.step(action)
        return self.augment_observations(obs), reward, done, info

    def reset(self, *args, **kwargs):
        self.tracker.reset()
        obs = self.env.reset()

        return self.augment_observations(obs)

    def get_obj_pose(self):
        # prepare obs
        obs_dict = self.env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]["rgb"])
            points.append(obs_dict[key]["points"])

        # track points
        tracked_points = self.tracker.track_multiview(
            rgbs, points, color=self.color_track, show=self.verbose_track
        )
        # crop to workspace
        cropped_points = crop_points(
            tracked_points, crop_min=self.crop_min, crop_max=self.crop_max
        )

        # get raw or filtered rod pose
        if self.obj_id == "rod":
            if self.filter:
                return self.tracker.get_rod_pose(
                    cropped_points,
                    lowpass_filter=True,
                    cutoff_freq=self.cutoff_freq,
                    control_hz=self.env.control_hz,
                    show=self.verbose_track,
                )
            else:
                return self.tracker.get_rod_pose(
                    cropped_points, lowpass_filter=False, show=self.verbose_track
                )
