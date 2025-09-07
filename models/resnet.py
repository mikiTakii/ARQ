from __future__ import absolute_import
from lib.utils.quantize_utils import QConv2d, QLinear
import torch.nn as nn
import math
import os
import torch
from typing import List


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, conv_layer, stride=1):
    if conv_layer == nn.Conv2d:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    else:
        return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, half_wave=True)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        conv_layer = QConv2d
        self.conv1 = conv3x3(inplanes, planes, conv_layer, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', conv_layer=nn.Conv2d):
        super(ResNet, self).__init__()
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
        _CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]
        self.normalize_layer = NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        if conv_layer == nn.Conv2d:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                bias=False)
        else:
            self.conv1 = conv_layer(3, 16, kernel_size=3, padding=1,
                                bias=False, half_wave=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, conv_layer)
        self.layer2 = self._make_layer(block, 32, n, conv_layer, stride=2)
        self.layer3 = self._make_layer(block, 64, n, conv_layer, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if conv_layer == nn.Conv2d:
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.fc = QLinear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, conv_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if conv_layer == nn.Conv2d:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv_layer(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False, half_wave=True),
                    nn.BatchNorm2d(planes * block.expansion),
                )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def resnet20(pretrained=False, **kwargs):
    model = ResNet(depth=20, num_classes=10, conv_layer=nn.Conv2d)
    return model


def qresnet20(pretrained=False, num_classes=10, **kwargs):
    model = ResNet(depth=20, num_classes=num_classes, conv_layer=QConv2d)
    if pretrained:
        # Load pretrained model.
        print('==> load pretrained resnet20 model..')
        path = './checkpoint/resnet20/checkpoint.pth.tar'
        assert os.path.isfile(path), 'Error: no checkpoint directory found!'
        ch = torch.load(path)
        state_dict = ch['state_dict']
        model.load_state_dict(state_dict, strict=False)
    return model


class NormalizeLayer(torch.nn.Module):

    def __init__(self, means: List[float], sds: List[float]):

        super(NormalizeLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.means = torch.tensor(means).to(self.device)
        self.sds = torch.tensor(sds).to(self.device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        return (input - means) / sds
