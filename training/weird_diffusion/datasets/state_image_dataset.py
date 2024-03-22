import glob
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from training.weird_diffusion.datasets.utils import *


class StateImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 num_trajectories: int = -1,
                 image_keys: list = ['left_rgb'],
                 state_keys: str = ['lowdim_ee', 'lowdim_qpos'],
                 ):
        
        self.image_keys = image_keys
        self.state_keys = state_keys
        assert len(image_keys) or len(state_keys), "At least one of image_keys or state_keys should be non-empty"

        # load the demonstration dataset:
        file_names = glob.glob(f"{dataset_path}/episode_*.npy")
        
        data = []
        for file in tqdm(file_names[:num_trajectories]):
            data.append(np.load(file, allow_pickle=True))
        
        assert len(data) > 1, f"WARNING: no data loaded from {dataset_path}!"

        actions = []
        images = []
        states = []
        episode_ends = []

        for i, episode in enumerate(data):
            for j, step in enumerate(episode):
                if len(state_keys):
                    sts = []
                    for key in state_keys:
                        st = step[key]
                        sts.append(st)
                    state = np.concatenate(sts, axis=0)
                    states.append(state)
                if len(image_keys):
                    imgs = []
                    for key in image_keys:
                        img = step[key]
                        # TODO deal with cropping properly
                        img = img[:, 160:, :]
                        img = np.moveaxis(img, -1, 0)
                        imgs.append(img)
                    image = np.stack(imgs, axis=0)
                    images.append(image)
                actions.append(step['action'])
                if j == len(episode) - 1:
                    episode_ends.append(j + 1)

        actions = np.array(actions).astype(np.float32)
        states = np.array(states).astype(np.float32)
        episode_ends = np.array(episode_ends)
        images = np.array(images).astype(np.float32)

        # (N, D)
        train_data = {'action': actions}
        if len(state_keys):
            train_data['state'] = states

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # compute statistics and normalize data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        if len(image_keys):
            normalized_train_data['image'] = images

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        if len(self.image_keys):
            nsample['image'] = nsample['image'][:self.obs_horizon, :]
        if len(self.state_keys):
            nsample['state'] = nsample['state'][:self.obs_horizon, :]
        return nsample


if __name__ == "__main__":
    dataset_path = 'data/pick_red_cube_synthetic/train'
    dataset1 = StateImageDataset(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_trajectories=10,
        image_keys = ['left_rgb'],
        state_keys = ['lowdim_ee', 'lowdim_qpos'],
    )
    dataset2 = StateImageDataset(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_trajectories=10,
        image_keys = ['left_rgb'],
        state_keys = [],
    )

    dataset3 = StateImageDataset(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_trajectories=10,
        image_keys = [],
        state_keys = ['lowdim_ee', 'lowdim_qpos'],
    )


    x = dataset1[0] 
    y = dataset2[0]
    z = dataset3[0]
    print('here')
