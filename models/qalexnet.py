from typing import List
import os
import torch
import torch.nn as nn
from lib.utils.quantize_utils import QConv2d, QLinear

__all__ = ['qalexnet', 'alexnet']

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class NormalizeLayer(nn.Module):
    def __init__(self, means: List[float], sds: List[float]):
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means).view(1, -1, 1, 1))
        self.register_buffer('sds', torch.tensor(sds).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.means) / self.sds


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, conv_layer=nn.Conv2d, linear_layer=nn.Linear):
        super(AlexNet, self).__init__()
        self.normalize_layer = NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        self.features = nn.Sequential(
            conv_layer(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv_layer(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv_layer(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            conv_layer(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            conv_layer(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            linear_layer(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            linear_layer(4096, 4096),
            nn.ReLU(inplace=True),
            linear_layer(4096, num_classes),
        )

    def forward(self, x):
        x = self.normalize_layer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def qalexnet(pretrained=False, num_classes=1000, **kwargs):
    model = AlexNet(num_classes=num_classes, conv_layer=QConv2d, linear_layer=QLinear)
    if pretrained:
        path = './checkpoint/alexnet_imagenette/model_best.pth.tar'
        assert os.path.isfile(path), 'Error: no checkpoint found at ' + path
        ch = torch.load(path, map_location='cpu')
        # checkpoint keys are net.features.* / net.classifier.*; strip the "net." prefix
        remapped = {}
        for k, v in ch['state_dict'].items():
            if k.startswith('net.'):
                remapped[k[len('net.'):]] = v
            else:
                remapped[k] = v   # normalize_layer.means / .sds stay as-is
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print('  loaded qalexnet weights | missing:', len(missing), 'unexpected:', len(unexpected))
    return model


def alexnet(pretrained=False, num_classes=1000, **kwargs):
    return AlexNet(num_classes=num_classes, conv_layer=nn.Conv2d, linear_layer=nn.Linear)