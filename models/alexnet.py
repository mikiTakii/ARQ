from typing import List
import torch
import torch.nn as nn
from torchvision.models import alexnet as tv_alexnet

__all__ = ['alexnet']

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class NormalizeLayer(nn.Module):
    def __init__(self, means: List[float], sds: List[float]):
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means).view(1, -1, 1, 1))
        self.register_buffer('sds', torch.tensor(sds).view(1, -1, 1, 1))

    def forward(self, input):
        return (input - self.means) / self.sds


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.normalize_layer = NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        self.net = tv_alexnet(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.net(self.normalize_layer(x))


def alexnet(pretrained=False, num_classes=1000, **kwargs):
    return AlexNet(num_classes=num_classes)