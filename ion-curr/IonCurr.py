import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(out_channels)
        dilation2 = max(dilation // 2, 2)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3,
                               padding=dilation2, dilation=dilation2)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class GlobalContextPredictor(nn.Module):
    def __init__(self, input_nc=1):
        super().__init__()
        self.feature_net = nn.Sequential(
            DilatedResBlock(input_nc, 12, dilation=4),
            nn.MaxPool3d(2),

            DilatedResBlock(12, 24, dilation=6),
            nn.AdaptiveAvgPool3d(2),

            nn.Conv3d(24, 48, 2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        x = self.feature_net(x)
        return self.regressor(x).squeeze(-1)
