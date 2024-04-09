from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from training.basic_bc.models.utils import init_weights


class CNN(nn.Module):
    def __init__(self, input_chn, output_dim, act="ReLU", output_act="Identity"):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        self.layers = nn.Sequential(
            nn.Conv2d(input_chn, 32, 4, stride=2),
            act_fn(), 
            nn.Conv2d(32, 64, 4, stride=2),
            act_fn(), 
            nn.Conv2d(64, 64, 3, stride=2),
            act_fn(),
            nn.Conv2d(64, 32, 3, stride=2),
            act_fn(),
            nn.Flatten(),
            # nn.Linear(32 * 35 * 35, output_dim),
            output_act_fn(),
        )
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)