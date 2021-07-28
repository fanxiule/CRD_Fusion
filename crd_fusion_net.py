import os
import torch
import torch.nn as nn

import networks


class CRDFusionNet(nn.Module):
    def __init__(self, scale_list, max_disp, img_h, img_w, baseline=False, gen_fusion=True, reg_fusion=True):
        """
        The high-level module for the whole network

        :param scale_list: list of exponents for all feature scales used in the network, e.g. [0, 3] or [0, 1, 2, 3]
        :param max_disp: maximum number of disparities after image downscaling is applied
        :param img_h: height of the input image
        :param img_w: width of the input image
        :param baseline: if set to True, the baseline model is used. Default to False
        :param gen_fusion: if set to True, raw disparity fusion applied to cost generation module. Default to True
        :param reg_fusion: if set to True, raw disparity fusion applied to disparity regression module. Default to True
        """
        super(CRDFusionNet, self).__init__()
        self.models = nn.ModuleDict()
        down_scale = 1 / (2 ** scale_list[-1])
        max_disp_at_scale = int(max_disp * down_scale)

        if baseline:  # use baseline modules
            self.models['extractor'] = networks.BaselineExtractor(scale_list[-1])
            self.models['disp_refine'] = networks.BaselineRefiner(scale_list)
        else:  # use the proposed modules
            self.models['extractor'] = networks.FeatureExtractor(scale_list[-1])
            self.models['disp_refine'] = networks.DispRefiner(scale_list, img_h, img_w)

        # common modules
        self.models['disp_proc'] = networks.DispProcessor(max_disp_at_scale, down_scale)
        self.models['cost_gen'] = networks.CostGenerator(max_disp_at_scale, gen_fusion)
        self.models['disp_est'] = networks.DispEstimator()
        self.models['disp_reg'] = networks.DispRegressor(max_disp_at_scale, reg_fusion)

    def get_params(self):
        """
        Obtain number of parameters in the model

        :return: List of all model parameters
        """
        parameters_to_train = []
        for k in self.models:
            parameters_to_train += list(self.models[k].parameters())
        return parameters_to_train

    def to(self, *args, **kwargs):
        """
        Override the default to() function to make sure everything in the model can be put to the proper device

        ::param args:
        :param kwargs:
        :return: None
        """
        for k in self.models:
            self.models[k].to(*args, **kwargs)

    @staticmethod
    def _weight_init(m):
        """
        Initialize a layer's weight according to normal distribution and its bias terms with zeros

        :param m: a layer in tensor form
        :return: None
        """
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def init_model(self):
        """
        Initialize the model's weights and bias terms
        :return: None
        """
        for k in self.models:
            self.models[k].apply(self._weight_init)

    def forward(self, l_rgb, r_rgb, raw_disp, conf_mask):
        """
        Forward pass of the complete model

        :param l_rgb: left RGB tensor
        :param r_rgb: right RGB tensor
        :param raw_disp: raw disparity tensor normalized between [0, 1]
        :param conf_mask: confidence mask tensor
        :return: outputs containing prelim disp, refined disp, and occlusion mask (if applicable)
        """
        l_feature = self.models['extractor'](l_rgb)
        r_feature = self.models['extractor'](r_rgb)
        raw_disp_prob, conf = self.models['disp_proc'](raw_disp, conf_mask)
        cost = self.models['cost_gen'](l_feature[-1], r_feature[-1], raw_disp_prob, conf)
        cost = self.models['disp_est'](cost)
        prelim_disp = self.models['disp_reg'](cost, raw_disp_prob, conf)
        output = self.models['disp_refine'](l_feature, r_feature, prelim_disp)
        output['prelim_disp'] = prelim_disp
        return output

    def save_model(self, save_path):
        """
        Save weights of the model

        :param save_path: directory to the folder that will store the .pth checkpoint files
        :return: None
        """
        for model_name, m in self.models.items():
            weights_to_save = m.state_dict()
            if len(weights_to_save) != 0:
                checkpt_path = os.path.join(save_path, "%s.pth" % model_name)
                torch.save(weights_to_save, checkpt_path)

    def load_model(self, model_path):
        """
        Load pretrained weights to the model

        :param model_path: directory to the folder storing the pretrained .pth checkpoint files
        :return: None
        """
        checkpt_list = os.listdir(model_path)
        for checkpt in checkpt_list:
            if checkpt != "adam.pth":
                model_name, ext = checkpt.split(".")
                assert ext == "pth", "Non-checkpoint file(s) found in the provided directory"
                print("Loading pretrained weights for %s" % model_name)
                model_dict = self.models[model_name].state_dict()
                pretrained_dict = torch.load(os.path.join(model_path, checkpt))
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[model_name].load_state_dict(model_dict)
