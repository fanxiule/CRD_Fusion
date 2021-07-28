import torch.nn as nn

from .layers import ResidualBlock, ConvBlock2D


class FeatureExtractor(nn.Module):
    def __init__(self, feature_scale):
        """
        The feature extraction module to extract high level features from an RGB image

        :param feature_scale: exponent for the lowest feature scale used in the network, e.g. 3 means 1/2^3
        """
        super(FeatureExtractor, self).__init__()
        self.extractor = nn.ModuleList()
        in_ch = 3
        out_ch = 32
        for s in range(0, feature_scale):
            self.extractor.append(nn.Sequential(
                ConvBlock2D(in_ch=in_ch, out_ch=out_ch, kernel=5, stride=2, pad=2, dilation=1, activation='leaky_relu',
                            bn=True),
                ResidualBlock(channel=out_ch, kernel=3, activation='leaky_relu'),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)))
            in_ch = out_ch

    def forward(self, rgb):
        """
       Forward pass for the feature extraction module

       :param rgb: RGB image input
       :return: A stack of features from the original scale to the lowest scale
        """
        features = [rgb]
        feat = rgb
        for i in range(len(self.extractor)):
            feat = self.extractor[i](feat)
            features.append(feat)
        return features


class BaselineExtractor(nn.Module):
    def __init__(self, feature_scale):
        """
        The baseline feature extraction module to extract high level features from an RGB image

        :param feature_scale: exponent for the lowest feature scale used in the network, e.g. 3 means 1/2^3
        """
        super(BaselineExtractor, self).__init__()
        downsample_list = []
        for s in range(0, feature_scale):
            if s == 0:
                downsample_list.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2))
            else:
                downsample_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2))

        self.downsampling = nn.Sequential(*downsample_list)

        self.extractor = nn.Sequential(
            ResidualBlock(channel=32, kernel=3, activation='leaky_relu'),
            ResidualBlock(channel=32, kernel=3, activation='leaky_relu'),
            ResidualBlock(channel=32, kernel=3, activation='leaky_relu'),
            ResidualBlock(channel=32, kernel=3, activation='leaky_relu'),
            ResidualBlock(channel=32, kernel=3, activation='leaky_relu'),
            ResidualBlock(channel=32, kernel=3, activation='leaky_relu'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, rgb):
        """
        Forward pass for the baseline feature extraction module

        :param rgb: RGB image input
        :return: high level feature at the lowest scale
        """
        features = self.downsampling(rgb)
        features = self.extractor(features)
        features = [rgb, features]  # consistent with the output format from the proposed feature extraction module

        return features
