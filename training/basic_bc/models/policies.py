import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from functools import partial

from training.basic_bc.models.utils import TanhDistribution
from training.basic_bc.models.cnns import CNN
from training.basic_bc.models.mlps import MLP, GaussianMLP
from training.weird_diffusion.models.networks import get_resnet

class MixedGaussianPolicy(nn.Module):
    def __init__(self, img_shape, state_shape, act_shape, hidden_dim):
        super().__init__()
        
        input_dim = 0

        # # Custom CNN
        # visual_cnn = CNN(
        #     input_chn=img_shape[0],
        #     output_dim=hidden_dim,
        #     output_act="ReLU",
        # )
        # visual_feat_dim = visual_cnn(torch.zeros(1, *img_shape)).shape[1]
        # visual_mlp = MLP(input_dim=visual_feat_dim, hidden_dims=[hidden_dim], output_dim=hidden_dim, act="ReLU", output_act="ReLU")
        # self.visual_encoder = nn.Sequential(visual_cnn, visual_mlp)
        # input_dim += hidden_dim
        
        # Img ResNet18
        self.visual_encoder = get_resnet("resnet18")
        with torch.no_grad():
            input_dim += self.visual_encoder(torch.zeros(1, *img_shape)).shape[1]

        # State MLP
        self.state_encoder = MLP(input_dim=state_shape[0], hidden_dims=[hidden_dim], output_dim=hidden_dim, act="ReLU", output_act="ReLU")
        input_dim += hidden_dim

        # Head MLP
        hidden_dims = [hidden_dim for _ in range(2)]
        self.head = GaussianMLP(input_dim, hidden_dims, np.prod(act_shape))

    def forward_dist(self, imgs, states):
        visual_feat = self.visual_encoder(imgs.reshape((imgs.shape[0]*imgs.shape[1], *imgs.shape[2:])))
        state_feat = self.state_encoder(states)
        return self.head.forward_dist(torch.cat([visual_feat, state_feat], dim=-1))

    def forward(self, imgs, states, deterministic=False):
        # Return action and log prob
        dist = self.forward_dist(imgs, states)
        if deterministic:
            act = dist.mean
            log_prob = None
        else:
            act = dist.rsample()
            log_prob = dist.log_prob(act).sum(-1, True)
        return act

    def evaluate(self, imgs, states, act):
        # Return log prob
        dist = self.forward_dist(imgs, states)
        log_prob = dist.log_prob(act).sum(-1, True)
        return log_prob
    
    def compute_loss(self, imgs, states, actions):
        
        log_probs = self.evaluate(imgs, states, actions)

        loss = -log_probs.mean()
        return loss
   
class GaussianPolicy(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dim):
        super().__init__()
        self.pixel_obs = len(obs_shape) == 3
        if self.pixel_obs:
            self.encoder = CNN(
                input_chn=obs_shape[0],
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
        
        obs = obs.reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        log_probs = self.evaluate(obs, actions)

        loss = -log_probs.mean()
        return loss

class TanhGaussianPolicy(GaussianPolicy):
    def __init__(self, obs_shape, act_shape, hidden_dim, act_space=None):
        super().__init__(obs_shape, act_shape, hidden_dim)
        if act_space is None:
            self.loc = torch.tensor(0.0)
            self.scale = torch.tensor(1.0)
        else:
            self.loc = torch.tensor((act_space.high + act_space.low) / 2.0)
            self.scale = torch.tensor((act_space.high - act_space.low) / 2.0)

    def forward_dist(self, obs):
        dist = super().forward_dist(obs)
        return TanhDistribution(dist, self.loc, self.scale)