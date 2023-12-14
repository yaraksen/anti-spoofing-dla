from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(self).__init__()
        self.conv1x1_before = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv1x1_after = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.act_norm = nn.Sequential(nn.PReLU(), nn.BatchNorm1d(out_channels))
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.act_norm(self.conv1x1_before(x))
        out = self.bn(self.conv1x1_after(out))
        if hasattr(self, "residual_conv"):
            x = self.residual_conv(x)
        out = self.prelu(out + x)
        out = self.maxpool(out)
        return out