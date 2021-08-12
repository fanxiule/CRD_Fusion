import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import ResidualBlock, ConvBlock2D, Correlation, UpsampleBlock2D


class DispRefiner(nn.Module):
    def __init__(self, scale_list, img_h, img_w):
        """
        Disparity refinement module to refine prelim disparity with image features as guidance

        :param scale_list: list of exponents for all feature scales used in the network, e.g. [0, 3] or [0, 1, 2, 3]
        :param img_h: height of the input image at the original resolution
        :param img_w: width of the input image at the original resolution
        """
        super(DispRefiner, self).__init__()
        self.scale_list = scale_list
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        disp_range = 9

        self.correlation = nn.ModuleDict()  # calculate correlation between left and right features according to predicted disparity
        self.processor = nn.ModuleDict()  # obtain the residual feature
        self.disp_pred = nn.ModuleDict()  # predict disparity residual from residual feature
        self.mask_pred = nn.ModuleDict()  # predict occlusion mask from residual feature
        self.up_feat = nn.ModuleDict()  # upsample the residual feature
        self.feat_match = nn.ModuleDict()  # make sure the upsampled residual feature have the same channel number as the image feature

        # handle the input channel number and output number for each image scale
        in_ch_list = []  # input channel for the volume refiner
        out_ch_list = []  # output channel for the volume refiner, i.e. input for disp_pred and mask_pred
        feat_ch_list = []  # feature channel size at each resolution scale
        down_ch = 32
        for s in self.scale_list:
            if s == self.scale_list[0]:
                feat_ch_list.append(3)  # RGB image
                in_ch_list.append(disp_range + feat_ch_list[-1] + 16 + 1)
            elif s == self.scale_list[-1]:
                feat_ch_list.append(down_ch)  # high-level feature
                in_ch_list.append(disp_range + feat_ch_list[-1] + 1)  # no residual feature from previous scale
            else:
                feat_ch_list.append(down_ch)  # high-level feature
                in_ch_list.append(disp_range + feat_ch_list[-1] + 16 + 1)
            out_ch_list.append(in_ch_list[-1] + 3 * 32 + 2 * 16)

        for i in range(len(self.scale_list)):
            down_scale = 1 / (2 ** self.scale_list[i])
            h_at_scale = int(img_h * down_scale)
            w_at_scale = int(img_w * down_scale)
            self.correlation['corr%d' % self.scale_list[i]] = Correlation(kernel=3, disp_range=disp_range,
                                                                          height=h_at_scale, width=w_at_scale)
            self.processor['proc%d' % self.scale_list[i]] = RefinerVolProcessor(in_ch_list[i])
            self.disp_pred['disp%d' % self.scale_list[i]] = nn.Conv2d(in_channels=out_ch_list[i], out_channels=1,
                                                                      kernel_size=3, stride=1, padding=1)
            self.mask_pred['mask%d' % self.scale_list[i]] = nn.Conv2d(in_channels=out_ch_list[i], out_channels=1,
                                                                      kernel_size=3, stride=1, padding=1)
            if self.scale_list[i] != self.scale_list[-1]:
                up_factor = int(2 ** (self.scale_list[i + 1] - self.scale_list[i]))
                self.up_feat['up%d' % self.scale_list[i]] = UpsampleBlock2D(scale=up_factor, in_ch=out_ch_list[i + 1],
                                                                            out_ch=16, activation='leaky_relu', bn=True,
                                                                            mode='bilinear')
                self.feat_match['match%d' % self.scale_list[i]] = nn.Conv2d(in_channels=16,
                                                                            out_channels=feat_ch_list[i], kernel_size=3,
                                                                            stride=1, padding=1, dilation=1)

    def to(self, *args, **kwargs):
        """
        Override the default to() function to make sure everything in the model can be put to the proper device

        ::param args:
        :param kwargs:
        :return: None
        """
        for s in self.scale_list:
            self.correlation['corr%d' % s].to(*args, **kwargs)
            self.processor['proc%d' % s].to(*args, **kwargs)
            self.disp_pred['disp%d' % s].to(*args, **kwargs)
            self.mask_pred['mask%d' % s].to(*args, **kwargs)
            if s != self.scale_list[-1]:
                self.up_feat['up%d' % s].to(*args, **kwargs)
                self.feat_match['match%d' % s].to(*args, **kwargs)

    def forward(self, left_feat, right_feat, prelim_disp):
        """
        Forward pass of the disparity refinement module

        :param left_feat: stack of left features
        :param right_feat: stack of right features
        :param prelim_disp: preliminary disparity
        :return: stack of refined disparity and occlusion prediction at all scales of interest
        """
        prev_occ = 1
        prev_feat = None
        trade_off = 0
        prev_disp = prelim_disp
        prediction = {}
        for i in range(len(self.scale_list) - 1, -1, -1):
            if self.scale_list[i] != self.scale_list[-1]:
                up_factor = int(2 ** (self.scale_list[i + 1] - self.scale_list[i]))
                prev_disp = f.interpolate(pred_disp, scale_factor=up_factor, mode='bilinear', align_corners=False)
                prev_disp *= up_factor
                prev_occ = f.interpolate(pred_mask, scale_factor=up_factor, mode='bilinear', align_corners=False)
                prev_feat = self.up_feat['up%d' % self.scale_list[i]](vol)
                trade_off = self.feat_match['match%d' % self.scale_list[i]](prev_feat)

            corr = self.correlation['corr%d' % self.scale_list[i]](left_feat[self.scale_list[i]],
                                                                   right_feat[self.scale_list[i]], prev_disp, prev_occ,
                                                                   trade_off)
            vol = self.processor['proc%d' % self.scale_list[i]](corr, left_feat[self.scale_list[i]], prev_disp,
                                                                prev_feat)
            pred_disp = self.disp_pred['disp%d' % self.scale_list[i]](vol)
            pred_disp = self.relu(prev_disp + pred_disp)
            pred_mask = self.sigmoid(self.mask_pred['mask%d' % self.scale_list[i]](vol))
            prediction['refined_disp%d' % self.scale_list[i]] = pred_disp
            prediction['occ%d' % self.scale_list[i]] = pred_mask
        return prediction


class RefinerVolProcessor(nn.Module):
    def __init__(self, input_ch):
        """
        A module to process volume to obtain residual features

        :param input_ch: channel number of the inputs
        """
        super(RefinerVolProcessor, self).__init__()
        self.conv1 = ConvBlock2D(in_ch=input_ch, out_ch=32, kernel=3, stride=1, pad=1, dilation=1,
                                 activation='leaky_relu', bn=True)
        self.conv2 = ConvBlock2D(in_ch=(input_ch + 32), out_ch=32, kernel=3, stride=1, pad=2, dilation=2,
                                 activation='leaky_relu', bn=True)
        self.conv3 = ConvBlock2D(in_ch=(input_ch + 2 * 32), out_ch=32, kernel=3, stride=1, pad=4,
                                 dilation=4, activation='leaky_relu', bn=True)
        self.conv4 = ConvBlock2D(in_ch=(input_ch + 3 * 32), out_ch=16, kernel=3, stride=1, pad=1,
                                 dilation=1, activation='leaky_relu', bn=True)
        self.conv5 = ConvBlock2D(in_ch=(input_ch + 3 * 32 + 16), out_ch=16, kernel=3, stride=1, pad=1,
                                 dilation=1, activation='leaky_relu', bn=True)

    def forward(self, correlation, left_feat, prev_disp, prev_feat=None):
        """
        Forward pass for teh processor

        :param correlation: correlation scores between left and right features according to the previously predicted disparity
        :param left_feat: left feature
        :param prev_disp: predicted disparity at the previous scale or process
        :param prev_feat: previous residual feature
        :return:
        """
        if prev_feat is not None:
            cost = torch.cat((correlation, left_feat, prev_feat, prev_disp), dim=1)
        else:
            cost = torch.cat((correlation, left_feat, prev_disp), dim=1)
        x = self.conv1(cost)
        cost = torch.cat((cost, x), dim=1)
        x = self.conv2(cost)
        cost = torch.cat((cost, x), dim=1)
        x = self.conv3(cost)
        cost = torch.cat((cost, x), dim=1)
        x = self.conv4(cost)
        cost = torch.cat((cost, x), dim=1)
        x = self.conv5(cost)
        cost = torch.cat((cost, x), dim=1)
        return cost


class BaselineRefiner(nn.Module):
    def __init__(self, scale_list):
        """
        Baseline refinement module to calculate the residual based on coarse disparity and RGB image

        :param scale_list: list of exponents for all feature scales used in the network, e.g. [0, 3] or [0, 1, 2, 3]
        """
        super(BaselineRefiner, self).__init__()
        self.scale_list = scale_list
        self.conv_feature = nn.ModuleList()
        self.dilated_conv = nn.ModuleList()
        self.conv_out = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        for _ in self.scale_list:
            self.conv_feature.append(
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1))
            self.dilated_conv.append(DilConvBlock(32))
            self.conv_out.append(
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, left_rgb, _, prelim_disp):
        """
        Forward pass to the refinement module

        :param left_rgb: left RGB image
        :param _: place holder so that the forward pass input format is consistent with the proposed refiner
        :param prelim_disp: prelim disparity
        :return: refined disparity tensors at all image resolutions of interest
        """
        output = {}
        left_rgb = left_rgb[0]
        for i in range(len(self.scale_list) - 1, -1, -1):
            down_factor = 1 / (2 ** self.scale_list[i])
            rgb = f.interpolate(left_rgb, scale_factor=down_factor, mode='nearest', recompute_scale_factor=False)
            input_vol = torch.cat((prelim_disp, rgb), dim=1)
            x = self.conv_feature[i](input_vol)
            x = self.dilated_conv[i](x)
            x = self.conv_out[i](x)
            output['refined_disp%d' % self.scale_list[i]] = self.relu(x + prelim_disp)
            if self.scale_list[i] != 0:
                up_factor = int(2 ** (self.scale_list[i] - self.scale_list[i - 1]))
                prelim_disp = f.interpolate(output['refined_disp%d' % self.scale_list[i]], scale_factor=up_factor,
                                            mode='bilinear', align_corners=False)
                prelim_disp *= up_factor
        return output


class DilConvBlock(nn.Module):
    def __init__(self, channel):
        """
        A module to implement a series of dilated convolution layers

        :param channel: number of input/output channels
        """
        super(DilConvBlock, self).__init__()
        dilation_list = [1, 2, 4, 8, 1, 1]
        conv_list = []
        for i in range(len(dilation_list)):
            conv_list.append(ResidualBlock(channel, 3, "leaky_relu", dilation_list[i]))
        self.conv = nn.Sequential(*conv_list)

    def forward(self, input_vol):
        """
        Forward pass for the dilation block

        :param input_vol: input volume/feature
        :return: output volume/feature
        """
        x = self.conv(input_vol)
        return x
