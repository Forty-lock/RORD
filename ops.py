import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils

class conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(conv5x5, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                              padding=2*dilation, dilation=dilation, bias=False)
        self.conv = utils.spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=dilation, dilation=dilation, padding_mode='reflect', bias=False)
        self.conv = utils.spectral_norm(self.conv)
    def forward(self, x):
        return self.conv(x)

class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.conv = utils.spectral_norm(self.conv)
    def forward(self, x):
        return self.conv(x)

class conv_zeros(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_zeros, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 0)

    def forward(self, x):
        return self.conv(x)

class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=2)

    def forward(self, x):
        h = self.conv1(x)
        h = F.elu(h)
        h = self.conv2(h)
        h = F.elu(h)
        return h

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels*4)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        h = self.conv1(x)
        h = F.pixel_shuffle(h, 2)
        h = F.elu(h)
        h = self.conv2(h)
        h = F.elu(h)
        return h
