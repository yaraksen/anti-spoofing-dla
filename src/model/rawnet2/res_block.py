from torch import nn
import torch.nn.functional as F


class FMSBlock(nn.Module):
    def __init__(self, num_channels, **kwargs):
        super(self).__init__()
        self.fc = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        scales = F.adaptive_avg_pool1d(x, 1)
        B, N = x.shape[:2]
        scales = scales.view(B, -1)
        scales = F.sigmoid(self.fc(scales))
        scales = scales.view(B, N, -1)
        return x * scales + scales


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(self).__init__()

        if in_channels != out_channels:
            self.residual_conv1x1 = nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        else:
            self.residual_conv1x1 = nn.Identity()

        self.convs = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.maxpool = nn.MaxPool1d(3)
        self.fms = FMSBlock(out_channels)

    def forward(self, x):
        residual = self.residual_conv1x1(x)
        x = self.convs(x)
        x = self.maxpool(x + residual)
        x = self.fms(x)
        return x
