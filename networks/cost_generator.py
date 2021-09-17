import torch
import torch.nn as nn


class CostGenerator(nn.Module):
    def __init__(self, max_disp_at_scale):
        """
        A non-learnable module to generate the 3D cost volume using difference between left and right features within
        the given disparity range

        :param max_disp_at_scale: maximum disparity range at the lowest resolution scale
        """
        super(CostGenerator, self).__init__()
        self.max_disp_at_scale = max_disp_at_scale
        # self.fusion = fusion

    def forward(self, left_feature, right_feature):
        """
        Forward pass for the cost generation module

        :param left_feature: high level left feature at the lowest scale
        :param right_feature: high level right feature at the lowest scale
        :return: cost
        """
        batch, channel, height, width = left_feature.size()
        cost = left_feature.new_zeros(batch, channel, self.max_disp_at_scale, height, width)
        for disp in range(self.max_disp_at_scale):
            if disp > 0:
                cost[:, :, disp, :, disp:] = left_feature[:, :, :, disp:] - right_feature[:, :, :, :-disp]
            else:
                cost[:, :, disp, :, :] = left_feature - right_feature
        """if self.fusion:
            raw_cost = -raw_prob  # raw_cost in the range of [-1, 0]
            k = torch.max(cost) - torch.min(cost)
            raw_cost = k * raw_cost + torch.max(cost)  # normalize raw cost to match max and min or the data cost
            raw_cost = torch.unsqueeze(raw_cost, dim=1)
            conf = torch.unsqueeze(conf, dim=1)
            cost = conf * raw_cost + (1 - conf) * cost"""
        return cost
