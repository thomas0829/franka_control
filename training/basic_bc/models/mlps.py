import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from functools import partial
from training.basic_bc.models.utils import init_weights

class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, act="ReLU", output_act="Identity"
    ):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        layers = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, dim))
            layers.append(act_fn())
            curr_dim = dim
        layers.append(nn.Linear(curr_dim, output_dim))
        layers.append(output_act_fn())

        self.layers = nn.Sequential(*layers)
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x.float())
    
class GaussianMLP(MLP):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        act="ReLU",
        max_logvar=0.5,
        min_logvar=-10.0,
    ):
        super().__init__(input_dim, hidden_dims, 2 * output_dim, act=act)
        self.output_dim = output_dim
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar

    def forward(self, x):
        out = super().forward(x)
        mean, logvar = out.split(int(self.output_dim), -1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def forward_dist(self, x):
        mean, logvar = self.forward(x)
        dist = Normal(mean, torch.exp(0.5 * logvar))
        return dist

    def sample(self, x, deterministic=False):
        if deterministic:
            mean, _ = self.forward(x)
            return mean
        else:
            dist = self.forward_dist(x)
            return dist.rsample()