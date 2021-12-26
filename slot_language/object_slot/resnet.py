import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, List, Optional

from obj_utils import get_normalizer


def conv(in_planes: int,
         out_planes: int,
         kernel_size: int = 3,
         stride: int = 1,
         groups: int = 1,
         dilation: int = 1):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - stride) // 2
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if isinstance(norm_layer, str):
            self.bn1 = get_normalizer(norm_layer, planes)
            self.bn2 = get_normalizer(norm_layer, planes)
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.bn1 = norm_layer(planes)
            self.bn2 = norm_layer(planes)

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv(
            inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size)
        self.downsample = downsample
        self.stride = stride

        if self.stride == 2 and self.downsample is None:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(inplanes, planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StackedResBlock(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, norm) -> None:
        super().__init__()

        assert len(channels) % 2 == 0

        modules = [conv(in_channels, channels[0], kernel_size=kernel_size)]
        in_channels = channels[0]
        for i in range(len(channels) // 2):
            assert channels[i * 2] == channels[i * 2 + 1]
            modules.append(
                BasicBlock(
                    in_channels,
                    channels[i * 2],
                    kernel_size=kernel_size,
                    norm_layer=norm))
            in_channels = channels[i * 2 + 1]
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        return self.convs(x)
