import torch


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, obs_dict, acts, device="cpu"):

        self.obs_dict = obs_dict
        self.acts = acts

        for k in self.obs_dict.keys():
            self.obs_dict[k] = torch.as_tensor(
                self.obs_dict[k], dtype=torch.float32
            ).to(device)
        self.act = torch.as_tensor(acts, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, index):

        obs = {}
        for k in self.obs_dict.keys():
            obs[k] = self.obs_dict[k][index]
        act = self.acts[index]

        return obs, act
