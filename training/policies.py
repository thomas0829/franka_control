import numpy as np
import torch
import torch.nn as nn
from training.networks import GaussianMLP, CNN, MLP

class GaussianPolicy(nn.Module):
    def __init__(self, act_shape, hidden_dims=[64, 64], state_embed_dim=64, state_obs_shape=None, img_embed_dim=64, img_obs_shape=None):
        super().__init__()

        assert state_obs_shape is not None or img_obs_shape is not None

        self.state_obs_shape = state_obs_shape
        self.img_obs_shape = img_obs_shape
        input_dim = 0

        if self.img_obs_shape:
            self.img_encoder = CNN(
                input_dim=img_obs_shape,
                output_dim=img_embed_dim,
                output_act="ReLU",
            )
            input_dim += img_embed_dim

        if self.state_obs_shape:
            self.state_encoder = MLP(input_dim=np.prod(state_obs_shape), hidden_dims=[], output_dim=state_embed_dim, output_act="ReLU")
            input_dim += state_embed_dim

        self.head = GaussianMLP(input_dim, hidden_dims, np.prod(act_shape), act="ReLU")

    def forward_dist(self, obs):

        assert "img" in obs.keys() or "state" in obs.keys()

        embeds = []
        if self.img_obs_shape:
            embeds.append(self.img_encoder(obs["img"] / 255.))
        if self.state_obs_shape:
            embeds.append(self.state_encoder(obs["state"]))

        if len(embeds) == 2:
            obs = torch.cat(embeds, dim=-1)
        else:
            obs = embeds[0]
        
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
        
        # if not self.img_obs:
        #     obs = obs.reshape(-1, obs.shape[-1])
        #     actions = actions.reshape(-1, actions.shape[-1])

        log_probs = self.evaluate(obs, actions)

        loss = -log_probs.mean()
        return loss
   