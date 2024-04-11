import glob
import os
import pickle

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

        actions = []
        images = []
        states = []
        episode_ends = []

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
                    states.append(state)
                if len(image_keys):
                    imgs = []
                    for key in image_keys:
                        try:
                            img = step[key]
                        except:
                            print(f"WARNING: {key} not in {step.keys()}!")
                            exit()
                        img = np.moveaxis(img, -1, 0)
                        imgs.append(img)
                    image = np.stack(imgs, axis=0)
                    images.append(image)
                actions.append(step["action"])
                if j == len(episode) - 1:
                    episode_ends.append(j + 1)

        actions = np.array(actions).astype(np.float32)
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


if __name__ == "__main__":
    dataset_path = "data/green_block/train"
    dataset1 = StateImageDataset(
        dataset_path=dataset_path,
        num_trajectories=10,
        image_keys=["215122255213_rgb"],  # ['left_rgb'],
        state_keys=["lowdim_ee", "lowdim_qpos"],
    )
    x = dataset1[0]
