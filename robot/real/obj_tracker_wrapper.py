import copy
import gym
import numpy as np

from perception.trackers.color_tracker import ColorTracker
from utils.pointclouds import crop_points
from utils.transformations import quat_to_euler
from utils.transformations_mujoco import euler_to_quat_mujoco, quat_to_euler_mujoco

class ObjectTrackerWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obj_id="rod",
        obs_keys=None, # ["lowdim_ee", "lowdim_qpos", "obj_pose"],
        # tracking
        color_track="red",
        # crop_min=[0.0, -0.4, -0.1],
        # crop_max=[0.5, 0.4, 0.5],
        crop_min=-np.ones(3),
        crop_max=np.ones(3),
        filter=False,
        cutoff_freq=1,
        flatten=True,
        verbose=False,
        **kwargs
    ):
        super().__init__(env)

        print(f"WARNING: ObjectTrackerWrapper doesn't take {kwargs}!")
        self.obj_id = obj_id

        # Tracking
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.tracker = ColorTracker(outlier_removal=True)
        self.color_track = color_track
        self.verbose = verbose
        self.filter = filter
        self.cutoff_freq = cutoff_freq

        # Observations
        self.obs_keys = obs_keys if obs_keys is not None else self.env.observation_space.keys()
        # obs space dict to array
        for k in copy.deepcopy(self.env.observation_space.keys()):
            if k not in self.obs_keys:
                del self.env.observation_space.spaces[k]

        self.flatten = flatten
        obj_pose_low = -np.inf * np.ones(7)
        obj_pose_high = np.inf * np.ones(7)
        if self.flatten:
            low = np.concatenate([v.low for v in self.env.observation_space.values()])
            high = np.concatenate([v.high for v in self.env.observation_space.values()])
            # add obj pose
            low = np.concatenate([low, obj_pose_low])
            high = np.concatenate([high, obj_pose_high])
            # overwrite observation_space
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=low.shape)
        else:
            self.observation_space["obj_pose"] = gym.spaces.Box(
                low=obj_pose_low, high=obj_pose_high
            )

    def augment_observations(self, obs, flatten=True):
        obs["obj_pose"] = self.get_obj_pose()

        if flatten:
            tmp = []
            for k in self.obs_keys:
                tmp.append(obs[k])
            obs = np.concatenate(tmp)

        return obs

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        return self.augment_observations(obs, flatten=self.flatten), reward, done, info

    def reset(self, *args, **kwargs):
        
        # reset tracker history
        self.tracker.reset()

        # reset robot |
        obs = self.env.reset()

        return self.augment_observations(obs, flatten=self.flatten)

    def get_obj_pose(self):
        # prepare obs
        obs_dict = self.env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]["rgb"])
            points.append(obs_dict[key]["points"])

        # track points
        tracked_points = self.tracker.track_multiview(
            rgbs, points, color=self.color_track, show=self.verbose
        )
        # crop to workspace
        cropped_points = crop_points(
            tracked_points, crop_min=self.crop_min, crop_max=self.crop_max
        )

        # get raw or filtered rod pose
        if self.obj_id == "rod":
            if self.filter:
                obj_pose = self.tracker.get_rod_pose(
                    cropped_points,
                    lowpass_filter=True,
                    cutoff_freq=self.cutoff_freq,
                    control_hz=self.env.control_hz,
                    show=self.verbose,
                )
            else:
                obj_pose = self.tracker.get_rod_pose(
                    cropped_points, lowpass_filter=False, show=self.verbose
                )
            # convert to mujoco quaternion
            obj_pose[-4:] = euler_to_quat_mujoco(quat_to_euler(obj_pose[-4:]))
            return obj_pose
