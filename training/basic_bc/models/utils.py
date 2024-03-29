import torch
import torch.nn as nn

from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

def map_activation(act):
    if act == "LeakyReLU":
        return "leaky_relu"
    else:
        return act.lower()


def init_weights(m, act):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain(map_activation(act))
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class TanhDistribution(TransformedDistribution):
    def __init__(self, base_dist, loc=0.0, scale=1.0):
        transforms = [
            TanhTransform(cache_size=1),
            AffineTransform(loc=loc, scale=scale, cache_size=1),
        ]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        mean = self.base_dist.mean
        for transform in self.transforms:
            mean = transform(mean)
        return mean

    def entropy(self):
        return self.base_dist.entropy()