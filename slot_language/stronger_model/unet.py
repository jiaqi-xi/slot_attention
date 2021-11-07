import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    """Conv --> BN (potential) --> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=True):
        super().__init__()

        padding = kernel_size // 2
        self.single_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) if use_bn else nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(Conv --> BN (potential) --> ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 kernel_size=3,
                 use_bn=True,
                 residual=False):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            SingleConv(in_channels, mid_channels, kernel_size, use_bn),
            SingleConv(mid_channels, out_channels, kernel_size, use_bn),
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            res = x
        out = self.double_conv(x)
        if self.residual:
            out = out + res
        return out


class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_double_conv,
                 mid_channels=None,
                 kernel_size=3,
                 use_bn=True,
                 residual=False):
        super().__init__()

        self.conv = DoubleConv(
            in_channels,
            out_channels,
            mid_channels,
            kernel_size,
            use_bn=use_bn,
            residual=residual) if use_double_conv else SingleConv(
                in_channels, out_channels, kernel_size, use_bn=use_bn)

    def forward(self, x):
        return self.conv(x)


class UNetDown(nn.Module):
    """Downscaling with Pool then Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, use_double_conv,
                 use_maxpool, use_bn):
        super().__init__()

        self.downsample = nn.MaxPool2d(2) if use_maxpool else \
            nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        self.conv = Conv(
            in_channels,
            out_channels,
            use_double_conv,
            kernel_size=kernel_size,
            use_bn=use_bn)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        return x


class UNetUp(nn.Module):
    """Upscaling then Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, use_double_conv,
                 use_bilinear, use_bn):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear',
            align_corners=True) if use_bilinear else nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)

        mid_channel = in_channels // 2 if use_bilinear else None
        self.conv = Conv(in_channels, out_channels, use_double_conv,
                         mid_channel, kernel_size, use_bn)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, use_double_conv,
                 use_maxpool, use_bn):
        super().__init__()

        self.channels = list(copy.deepcopy(channels))
        in_conv = Conv(
            in_channels,
            channels[0],
            use_double_conv,
            kernel_size=kernel_size,
            use_bn=use_bn)
        down_convs = [in_conv]
        for i in range(len(channels) - 1):
            down_convs.append(
                UNetDown(channels[i], channels[i + 1], kernel_size,
                         use_double_conv, use_maxpool, use_bn))
        self.down_convs = nn.ModuleList(down_convs)

    def forward(self, x):
        outputs = []
        for conv in self.down_convs:
            x = conv(x)
            outputs.append(x)
        return outputs


class UNetDecoder(nn.Module):

    def __init__(self, enc_channels, channels, kernel_size, use_double_conv,
                 use_bilinear, use_bn):
        super().__init__()

        assert len(enc_channels) == len(channels) + 1
        channels = list(copy.deepcopy(channels))
        # make it from small to large
        if channels[0] > channels[-1]:
            channels = channels[::-1]
        in_channels = [
            enc_channels[i] + channels[i + 1]
            for i in range(len(channels) - 1)
        ]
        in_channels.append(enc_channels[-2] + enc_channels[-1])
        up_convs = []
        for i in range(len(channels)):
            up_convs.append(
                UNetUp(in_channels[i], channels[i], kernel_size,
                       use_double_conv, use_bilinear, use_bn))
        self.up_convs = nn.ModuleList(up_convs)

    def forward(self, encoder_out):
        assert len(encoder_out) == len(self.up_convs) + 1
        x = self.up_convs[-1](encoder_out[-1], encoder_out[-2])
        for i in range(len(self.up_convs) - 2, -1, -1):
            x = self.up_convs[i](x, encoder_out[i])
        return x


class UNet(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, use_double_conv,
                 use_maxpool, use_bilinear, use_bn):
        super().__init__()

        self.encoder = UNetEncoder(
            in_channels,
            channels,
            kernel_size,
            use_double_conv=use_double_conv,
            use_maxpool=use_maxpool,
            use_bn=use_bn)

        self.decoder = UNetDecoder(
            self.encoder.channels,
            channels[:-1],
            kernel_size,
            use_double_conv=use_double_conv,
            use_bilinear=use_bilinear,
            use_bn=use_bn)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, use_double_conv,
                 use_bilinear, use_bn):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear',
            align_corners=True) if use_bilinear else nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1)

        mid_channels = None
        self.conv = Conv(in_channels, out_channels, use_double_conv,
                         mid_channels, kernel_size, use_bn)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
