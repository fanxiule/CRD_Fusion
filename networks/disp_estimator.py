import torch
import torch.nn as nn

from .layers import ConvBlock3D


class DispEstimator(nn.Module):
    def __init__(self):
        """
        A module to compute the cost distribution for each pixel across the whole disparity range
        """
        super(DispEstimator, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock3D(32, 32, 3, 1, 1, 1, 'leaky_relu', True),
            ConvBlock3D(32, 32, 3, 1, 1, 1, 'leaky_relu', True),
            ConvBlock3D(32, 32, 3, 1, 1, 1, 'leaky_relu', True),
            ConvBlock3D(32, 32, 3, 1, 1, 1, 'leaky_relu', True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, cost_vol):
        """
        Forward pass for the disparity estimator to aggregate a given cost volume

        :param cost_vol: cost volume formed by using left and right features
        :return: cost across the whole disparity range
        """
        x = self.conv(cost_vol)
        disp_cost = torch.squeeze(x, dim=1)
        return disp_cost
