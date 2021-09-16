import torch
import torch.nn as nn
from .layers import ResidualBlock


class DispRegressor(nn.Module):
    def __init__(self, max_disp_at_scale, fusion=True):
        """
        A module to regress coarse disparity from the cost volume

        :param max_disp_at_scale: maximum disparity range at the lowest resolution scale
        :param fusion: if set to True, the pseudo probability from raw disparity is fused with the cost volume
        """
        super(DispRegressor, self).__init__()
        self.candidate_disp = torch.arange(max_disp_at_scale, dtype=torch.float)
        self.candidate_disp = torch.reshape(self.candidate_disp, (-1, 1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.fusion = fusion
        if self.fusion:
            self.conv = nn.Sequential(
                ResidualBlock(max_disp_at_scale, 3, 'leaky_relu'),
                ResidualBlock(max_disp_at_scale, 3, 'leaky_relu'),
                ResidualBlock(max_disp_at_scale, 3, 'leaky_relu'),
                nn.Conv2d(in_channels=max_disp_at_scale, out_channels=max_disp_at_scale, kernel_size=3, stride=1,
                          padding=1)
            )

    def to(self, *args, **kwargs):
        """
        Override the default to() function so that candidate_disp can be put into the proper device as well

        :param args:
        :param kwargs:
        :return: None
        """
        self.candidate_disp = self.candidate_disp.to(*args, **kwargs)
        if self.fusion:
            self.conv.to(*args, **kwargs)

    def _soft_argmin(self, prob):
        """
        Soft argmin operation to regress disparity from probability distribution

        :param prob: probability distribution
        :return: disparity prediction
        """
        candidate = self.candidate_disp.repeat(prob.shape[0], 1, prob.shape[2], prob.shape[3])
        weighted_candidate = candidate * prob
        output = torch.sum(weighted_candidate, dim=1, keepdim=True)
        return output

    def forward(self, cost, raw_prob, conf):
        """
        Forward pass for the disparity regression module

        :param cost: cost volume at the lowest image resolution
        :param raw_prob: pseudo probability distribution of raw disparity at lowest scale
        :param conf: confidence mask for the raw disparity at lowest scale
        :return: Coarse estimated disparity at lowest scale
        """
        prob = self.softmax(-cost)
        if self.fusion:
            prob = (1 - conf) * prob + conf * raw_prob
            prob = prob + self.conv(prob)
            prob = self.softmax(prob)
        prelim_disp = self._soft_argmin(prob)
        return prelim_disp
