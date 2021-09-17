import torch
import torch.nn as nn
import torch.nn.functional as f


class DispProcessor(nn.Module):
    def __init__(self, max_disp_at_scale, down_scale):
        """
        A non-learnable module to downsample raw disparity and confidence

        :param max_disp_at_scale: maximum disparity range at the lowest resolution scale
        :param down_scale: downscaling factor of the lowest scale compared to the original scale
        """
        super(DispProcessor, self).__init__()
        self.max_disp_at_scale = max_disp_at_scale
        self.down_scale = down_scale

    """def _gen_disp_coding(self, disp):
        # Generate one hot encoding for available disparity range of a given disparity tensor

        # :param disp: disparity tensor
        # :return: one hot encoded version of the input disparity
        
        encoded_disp = f.one_hot(disp, num_classes=self.max_disp_at_scale)
        encoded_disp = torch.squeeze(encoded_disp, dim=1)
        encoded_disp = encoded_disp.permute(0, 3, 1, 2)
        return encoded_disp"""

    def forward(self, raw_disp, mask):
        """
        Forward pass for the raw disparity processor

        :param raw_disp: raw disparity tensor normalized between [0, 1]
        :param mask: confidence mask tensor
        :return: raw disparity and confidence mask at the lowest scale
        """
        # downsample raw disparity and confidence mask
        raw_disp_at_scale = f.interpolate(raw_disp, scale_factor=self.down_scale, mode='nearest',
                                          recompute_scale_factor=False)
        raw_disp_at_scale *= self.max_disp_at_scale
        mask_at_scale = f.interpolate(mask, scale_factor=self.down_scale, mode='nearest', recompute_scale_factor=False)

        """
        # weights for floor and ceiling disparity
        raw_disp_floor = (torch.floor(raw_disp_at_scale)).to(torch.int64)  # must be torch.int64 to use with f.one_hot
        raw_disp_ceiling = torch.clamp(raw_disp_floor + 1, min=0, max=self.max_disp_at_scale - 1)
        floor_weight = raw_disp_ceiling - raw_disp_at_scale
        ceiling_weight = 1 - floor_weight

        # one-hot version of the floor and ceiling disparity
        raw_disp_floor = self._gen_disp_coding(raw_disp_floor)
        raw_disp_ceiling = self._gen_disp_coding(raw_disp_ceiling)

        pseudo_disp_prob = floor_weight * raw_disp_floor + ceiling_weight * raw_disp_ceiling
        """

        return raw_disp_at_scale, mask_at_scale
