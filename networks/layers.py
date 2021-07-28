import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, dilation, activation, bn=True):
        """
        A module for a common 2D convolution process including batch normalization and activation

        :param in_ch: number of input channels
        :param out_ch: number of output channels
        :param kernel: kernel size of the convolution layer
        :param stride: stride for the convolution layer
        :param pad: padding for the convolution layer
        :param dilation: dilation for the convolution layer
        :param activation: activation function, suchs as 'relu', 'sigmoid', 'leaky_relu', etc.
        :param bn: a flag to indicate whether or not to include a batch normalization after conv, default to True
        """
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad,
                              dilation=dilation, bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = nn.Identity()

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, input_vol):
        """
        Forward pass for the 2D conv block

        :param input_vol: input feature/volume
        :return: output feature/volume
        """
        output = self.conv(input_vol)
        output = self.bn(output)
        output = self.activation(output)
        return output


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, dilation, activation, bn=True):
        """
        A module for a common 3D convolution process including batch normalization and activation

        :param in_ch: number of input channels
        :param out_ch: number of output channels
        :param kernel: kernel size of the convolution layer
        :param stride: stride for the convolution layer
        :param pad: padding for the convolution layer
        :param dilation: dilation for the convolution layer
        :param activation: activation function, suchs as 'relu', 'sigmoid', 'leaky_relu', etc.
        :param bn: a flag to indicate whether or not to include a batch normalization after conv, default to True
        """
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad,
                              dilation=dilation, bias=not bn)
        if bn:
            self.bn = nn.BatchNorm3d(out_ch)
        else:
            self.bn = nn.Identity()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, input_vol):
        """
        Forward pass for the 3D conv block

        :param input_vol: input feature/volume
        :return: output feature/volume
        """
        output = self.conv(input_vol)
        output = self.bn(output)
        output = self.activation(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channel, kernel, activation, dilation=1):
        """
        A module for residual block with the same number of input and output channels, and same input and output feature
        size

        :param channel: number of input and output channels, assuming they are the same
        :param kernel: kernel size for the convolution layer
        :param activation: activation function, suchs as 'relu', 'sigmoid', 'leaky_relu', etc.
        :param dilation: dilation for the convolution layer, default to 1
        """
        super(ResidualBlock, self).__init__()
        assert kernel % 2 != 0, "Kernel size should be an odd number"
        pad = dilation * (kernel // 2)  # find padding required to maintain the same input/output feature size
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel, dilation=dilation,
                               stride=1, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel, dilation=dilation,
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, input_vol):
        """
        Forward pass for the residual block

        :param input_vol: input feature/volume
        :return: output feature/volume with the same size as the input
        """
        x = self.conv1(input_vol)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + input_vol
        output = self.activation(x)
        return output


class UpsampleBlock2D(nn.Module):
    def __init__(self, scale, in_ch, out_ch, activation, bn, mode):
        """
        A module to upsample an input volume and applies convolution to it after

        :param scale: the upsampling factor for the input volume
        :param in_ch: number of input channels for the input volume
        :param out_ch: desired number of output channels
        :param activation: activation function after convolution
        :param bn: a flag to indicate whether or not to include batch normalization after convolution
        :param mode: the mode used for upsampling
        """
        super(UpsampleBlock2D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)
        self.conv = ConvBlock2D(in_ch, out_ch, 3, 1, 1, 1, activation, bn)

    def forward(self, input_vol):
        """
        Forward pass for the 2D upsampling block

        :param input_vol: input feature/volume with size (BxCxHxW)
        :return: output feature/volume with size (BxCsxHsxWs) where s is the upsampling scale
        """
        output = self.upsample(input_vol)
        output = self.conv(output)
        return output


class Correlation(nn.Module):
    def __init__(self, kernel, disp_range, height, width):
        """
        A module to calculate correlation score between left and right features according to the given flow

        :param kernel: kernel size, i.e. patch size in correlation calculation
        :param disp_range: possible disparity range, which is centered at the predicted disparity, for consideration
        :param height: height of the feature image
        :param width: width of the feature image
        """
        super(Correlation, self).__init__()
        assert kernel > 0 and disp_range > 0, "Kernel size and disparity range must be greater than 0"
        assert kernel % 2 != 0 and disp_range % 2 != 0, "Kernel size and disparity range must be odd numbers"
        self.disp_window = disp_range // 2
        self.kernel_element = kernel * kernel
        pad = kernel // 2
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel, stride=1, padding=pad)

        # for bilinear sampling
        grid_y = torch.linspace(0, height - 1, height)
        grid_x = torch.linspace(0, width - 1, width)
        self.grid_ver, self.grid_hor = torch.meshgrid(grid_y, grid_x)
        self.grid_ver = torch.unsqueeze(self.grid_ver, dim=0)
        self.grid_hor = torch.unsqueeze(self.grid_hor, dim=0)
        # constants for normalizing pixel indices to prep for bilinear sampling
        # y = kx+b to map image pixel between 0 and height (or width) - 1 to between -1 and 1
        self.norm_k_height = 2 / (height - 1)
        self.norm_b_height = -1
        self.norm_k_width = 2 / (width - 1)
        self.norm_b_width = -1

    def to(self, *args, **kwargs):
        """
        Override the default to() function to make sure everything in the model can be put to the proper device

        ::param args:
        :param kwargs:
        :return: None
        """
        self.grid_hor = self.grid_hor.to(*args, **kwargs)
        self.grid_ver = self.grid_ver.to(*args, **kwargs)

    def _warp_im(self, src, disp, batch, occ=1, trade_off=0):
        """
        Warp the src feature according to the given disparity with the consideration of occlusion and trade-off

        :param src: src feature
        :param disp: disparity
        :param batch: batch size of the current batch
        :param occ: occlusion mask
        :param trade_off: trade off term, i.e. residual features from previous resolution scale
        :return: a reconstructed synthetic feature
        """
        grid_row = self.grid_ver.repeat(batch, 1, 1)
        grid_col = self.grid_hor.repeat(batch, 1, 1)
        grid_col = grid_col - torch.squeeze(disp, dim=1)
        # normalize grid so that they are between -1 and 1, which is a format accepted by F.grid_sample()
        grid_row = grid_row * self.norm_k_height + self.norm_b_height
        grid_col = grid_col * self.norm_k_width + self.norm_b_width
        grid = torch.stack([grid_col, grid_row], dim=3)
        synth = f.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)
        synth = occ * synth + trade_off
        return synth

    def _single_correlation(self, tgt, src):
        """
        Patch-based correlation between tgt and src image

        :param tgt: target feature image
        :param src: source feature image
        :return: patch-based correlation score
        """
        corr = tgt * src
        corr = torch.sum(corr, dim=1, keepdim=True)
        corr = self.kernel_element * self.avg_pool(corr)
        return corr

    def forward(self, left_feat, right_feat, disp, occ, trade_off):
        """
        Forward pass of the correlation module

        :param left_feat: left feature image
        :param right_feat: right feature image
        :param disp: predicted disparity
        :param occ: occlusion mask
        :param trade_off: trade off term, i.e. residual features from previous image scale
        :return: correlation score
        """
        batch, _, _, _ = left_feat.size()
        correlation = torch.Tensor().to(left_feat.device)
        for i in range(-self.disp_window, self.disp_window + 1, 1):
            curr_disp = disp + i
            synth_left_feat = self._warp_im(right_feat, curr_disp, batch, occ, trade_off)
            curr_corr = self._single_correlation(left_feat, synth_left_feat)
            correlation = torch.cat((correlation, curr_corr), dim=1)
        return correlation
