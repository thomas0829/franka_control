import numpy as np
import torch.nn as nn
from training.networks import GaussianMLP, CNN

class GaussianPolicy(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dim):
        super().__init__()
        self.pixel_obs = len(obs_shape) == 3
        if self.pixel_obs:
            self.encoder = CNN(
                input_dim=obs_shape,
                output_dim=hidden_dim,
                output_act="ReLU",
            )
            input_dim = hidden_dim
        else:
            input_dim = np.prod(obs_shape)
        hidden_dims = [hidden_dim for _ in range(2)]
        self.head = GaussianMLP(input_dim, hidden_dims, np.prod(act_shape))

    def forward_dist(self, obs):
        if self.pixel_obs:
            obs = self.encoder(obs)
        return self.head.forward_dist(obs)

    def forward(self, obs, deterministic=False):
        # Return action and log prob
        dist = self.forward_dist(obs)
        if deterministic:
            act = dist.mean
            log_prob = None
        else:
            act = dist.rsample()
            log_prob = dist.log_prob(act).sum(-1, True)
        return act

    def evaluate(self, obs, act):
        # Return log prob
        dist = self.forward_dist(obs)
        log_prob = dist.log_prob(act).sum(-1, True)
        return log_prob

    def compute_loss(self, obs, actions):
        if not self.pixel_obs:
            obs = obs.reshape(-1, obs.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])

        log_probs = self.evaluate(obs, actions)

        loss = -log_probs.mean()
        return loss
