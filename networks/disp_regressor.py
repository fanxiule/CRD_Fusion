import torch
import torch.nn as nn


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

    def to(self, *args, **kwargs):
        """
        Override the default to() function so that candidate_disp can be put into the proper device as well

        :param args:
        :param kwargs:
        :return: None
        """
        self.candidate_disp = self.candidate_disp.to(*args, **kwargs)

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

    def forward(self, cost, raw_disp, conf):
        """
        Forward pass for the disparity regression module

        :param cost: cost volume at the lowest image resolution
        :param raw_disp: raw disparity at lowest scale
        :param conf: confidence mask for the raw disparity at lowest scale
        :return: Coarse estimated disparity at lowest scale
        """
        prob = self.softmax(-cost)
        prelim_disp = self._soft_argmin(prob)
        if self.fusion:
            prelim_disp = (1 - conf) * prelim_disp + conf * raw_disp
        return prelim_disp
