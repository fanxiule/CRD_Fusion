import os
import torch
from .crd_fusion_dataset import CRDFusionDataset


class KITTI2012TestDataset(CRDFusionDataset):
    def __init__(self, data_path, max_disp, resized_h, resized_w, conf_thres, imgnet_norm=True, sanity=False):
        """
        Dataset to load and prepare data from KITTI 2012 test split

        :param data_path: directory to the dataset
        :param max_disp: maximum disparity
        :param resized_h: final image height after resizing
        :param resized_w: final image width after resizing
        :param conf_thres: threshold for confidence score
        :param imgnet_norm: if set to True, the RGB images will be normalized by ImageNet's statistics
        :param sanity: if set to True, only includes 1 data point. Mostly used to debug the model
        """
        super(KITTI2012TestDataset, self).__init__(data_path, max_disp, 1, resized_h, resized_w, conf_thres, False,
                                                   imgnet_norm, sanity)
        self.data_path = os.path.join(self.data_path, "testing")
        img_list_path = os.path.join(self.data_path, "colored_0")
        frame_list = os.listdir(img_list_path)
        self.data_list = []
        for f in frame_list:
            if "_10" in f:
                self.data_list.append(f)

        if self.sanity:
            self.data_list.sort()
            self.data_list = [self.data_list[0]]

    def __getitem__(self, index):
        """
        Get a data sample

        :param index: index for the data list
        :return: a stack of input data in tensor form including left rgb (normalized and non normalized), right rgb
                 (normalized and non normalized), raw disparity, and frame id
        """
        frame = self.data_list[index]
        raw_inputs = {}
        l_rgb_path = os.path.join(self.data_path, "colored_0", frame)
        r_rgb_path = os.path.join(self.data_path, "colored_1", frame)
        disp_path = os.path.join(self.data_path, "raw_disp", frame)

        raw_inputs['l_rgb'] = self._get_rgb(l_rgb_path)
        raw_inputs['r_rgb'] = self._get_rgb(r_rgb_path)
        raw_inputs['raw_disp'] = self._get_disp(disp_path)

        _, self.orig_height, self.orig_width = raw_inputs['l_rgb'].size()

        inputs = self._pad_inputs(raw_inputs)
        inputs['raw_disp'] = self._normalize_disp(inputs['raw_disp'])
        # non normalized stereo images for conf calculation
        inputs['l_rgb_non_norm'] = torch.clone(inputs['l_rgb'])
        inputs['r_rgb_non_norm'] = torch.clone(inputs['r_rgb'])
        if self.imgnet_norm:
            inputs['l_rgb'], inputs['r_rgb'] = self._normalize_rgb(inputs['l_rgb'], inputs['r_rgb'])

        # for saving the predicted disparity as the correct format
        inputs['frame_id'] = frame

        return inputs
