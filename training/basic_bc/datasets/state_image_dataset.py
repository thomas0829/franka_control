import glob
import os
import pickle

import h5py
import numpy as np
import torch
from tqdm import tqdm

from training.weird_diffusion.datasets.utils import *


class StateImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        num_trajectories: int = -1,
        image_keys: list = ["left_rgb"],
        state_keys: str = ["lowdim_ee", "lowdim_qpos"],
    ):
        self.image_keys = image_keys
        self.state_keys = state_keys
        assert len(image_keys) or len(
            state_keys
        ), "At least one of image_keys or state_keys should be non-empty"

        # load the demonstration dataset:
        file_names = glob.glob(f"{dataset_path}/episode_*.npy")
        assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

        episode_ends = []

        # load first sample to get shapes
        tmp = np.load(file_names[0], allow_pickle=True)
        # assuming trajectories have similar lenght + 20% for safety
        approx_n_samples = int(len(file_names[:num_trajectories])*len(tmp)*1.2)
        
        # get image size
        img_tmp = tmp[0][self.image_keys[0]].transpose(2,0,1)
        image_size = img_tmp.shape
        image_size = [approx_n_samples, len(self.image_keys), *image_size]
        # get state size
        state_size = [approx_n_samples, np.sum([len(tmp[0][k]) for k in self.state_keys])]
        # get action size
        act_size = [approx_n_samples, len(tmp[0]["action"])]
        # initializing full numpy arrays vs stack/concatenate them saves memory!
        images = np.zeros(image_size, dtype=np.uint8)
        states = np.zeros(state_size, dtype=np.float32)
        actions = np.zeros(act_size, dtype=np.float32)

        sample_ctr = 0
        for file in tqdm(file_names[:num_trajectories]):

            episode = np.load(file, allow_pickle=True)

            for j, step in enumerate(episode):
                
                if len(state_keys):
                    sts = []
                    for key in state_keys:
                        try:
                            st = step[key]
                        except:
                            print(f"WARNING: {key} not in {step.keys()}!")
                            exit()
                        sts.append(st)
                    state = np.concatenate(sts, axis=0)
                    states[sample_ctr] = state

                if len(image_keys):
                    for key in image_keys:
                        try:
                            img = step[key]
                        except:
                            print(f"WARNING: {key} not in {step.keys()}!")
                            exit()
                        img = img.transpose(2,0,1)
                        images[sample_ctr] = img

                actions[sample_ctr] = step["action"]

                if j == len(episode) - 1:
                    episode_ends.append(j + 1)
                sample_ctr += 1

        # cut arrays to actual size
        actions = actions[:sample_ctr]
        states = states[:sample_ctr]
        images = images[:sample_ctr]

        # (N, D)
        train_data = {"action": actions}
        if len(state_keys):
            train_data["state"] = states

        # compute statistics and normalize data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # set image stats
        if len(image_keys):

            stats["image"] = {
                "min": 0 * np.ones(images.shape[1:], dtype=np.uint8),
                "max": 255 * np.ones(images.shape[1:], dtype=np.uint8),
            }

            normalized_train_data["image"] = images

        self.stats = stats
        self.normalized_train_data = normalized_train_data

    def __len__(self):
        return len(self.normalized_train_data["action"])

    def __getitem__(self, idx):
        # get normalized data using index
        nsample = {}
        if len(self.image_keys):
            nsample["image"] = self.normalized_train_data["image"][idx]
        if len(self.state_keys):
            nsample["state"] = self.normalized_train_data["state"][idx]
        nsample["action"] = self.normalized_train_data["action"][idx]
        return nsample


class HDF5StateImageDataset(StateImageDataset):
    def __init__(
        self,
        dataset_path: str,
        num_trajectories: int = -1,
        image_keys: list = ["world_camera_low_res_image"],
        state_keys: str = ["eef_pos", "eef_quat", "gripper_qpos", "joint_pos"],
    ):
        self.image_keys = image_keys
        self.state_keys = state_keys
        assert len(image_keys) or len(
            state_keys
        ), "At least one of image_keys or state_keys should be non-empty"

        # load the demonstration dataset:
        file_names = glob.glob(f"{dataset_path}/*.hdf5")
        assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

        actions = []
        images = []
        states = []
        episode_ends = []

        for file in tqdm(file_names[:num_trajectories]):

            data = h5py.File(file, "r+")
            demo = data["data"]["demo_0"]
            obs = demo["obs"]
            act = demo["actions"]

            horizon = len(demo["actions"])

            for j in range(horizon):
                if len(state_keys):
                    sts = []
                    for key in state_keys:
                        try:
                            st = obs[key][j]
                        except:
                            print(f"WARNING: {key} not in {obs.keys()}!")
                            exit()
                        sts.append(st)
                    state = np.concatenate(sts, axis=0)
                    states.append(state)
                if len(image_keys):
                    imgs = []
                    for key in image_keys:
                        try:
                            img = obs[key][j]
                        except:
                            print(f"WARNING: {key} not in {obs.keys()}!")
                            exit()
                        img = np.moveaxis(img, -1, 0)
                        imgs.append(img)
                    image = np.stack(imgs, axis=0)
                    images.append(image)

                actions.append(act[j])

                if j == horizon - 1:
                    episode_ends.append(j + 1)

        actions = np.array(actions).astype(np.float32)
        # unnormlize actions w/ Helen's max/min
        # [-1, 1] -> [-?, ?]
        actions[:, :6] *= np.array(
            [[0.05, 0.05, 0.05, 0.17453293, 0.17453293, 0.17453293]]
        )
        states = np.array(states).astype(np.float32)
        self.episode_ends = np.array(episode_ends)
        images = np.array(images).astype(np.uint8)
        # images = np.array(images).astype(np.float32)

        # (N, D)
        train_data = {"action": actions}
        if len(state_keys):
            train_data["state"] = states

        # compute statistics and normalize data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # compute statistics and normalize data to [-1,1]
        if len(image_keys):
            # img_min = images.min()
            # img_max = images.max()
            # stats["image"] = {
            #     "min": img_min * np.ones(images.shape[1:]),
            #     "max": img_max * np.ones(images.shape[1:]),
            # }
            # images = (images - img_min) / (img_max - img_min)
            # images = images * 2 - 1

            stats["image"] = {
                "min": 0 * np.ones(images.shape[1:], dtype=np.uint8),
                "max": 255 * np.ones(images.shape[1:], dtype=np.uint8),
            }

            normalized_train_data["image"] = images

        self.stats = stats
        self.normalized_train_data = normalized_train_data


if __name__ == "__main__":
    dataset_path = "data/green_block/train"
    dataset1 = StateImageDataset(
        dataset_path=dataset_path,
        num_trajectories=-1,
        image_keys=["215122255213_rgb"],  # ['left_rgb'],
        state_keys=["lowdim_ee", "lowdim_qpos"],
    )
    x = dataset1[0]

    dataset_path = "data/helen_1k/train"
    dataset2 = HDF5StateImageDataset(
        dataset_path=dataset_path,
        num_trajectories=10,
        # image_keys=["215122255213_rgb"],  # ['left_rgb'],
        # state_keys=["lowdim_ee", "lowdim_qpos"],
    )
    x = dataset2[0]
