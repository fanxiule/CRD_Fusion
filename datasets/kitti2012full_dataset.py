import os
import random
import cv2
from .crd_fusion_dataset import CRDFusionDataset


class Kitti2012FullDataset(CRDFusionDataset):
    def __init__(self, data_path, max_disp, downscale, resized_height, resized_width, conf_thres, is_train,
                 imgnet_norm=True, sanity=False):
        """
        Dataset to load and prepare data from the full KITTI 2012 training set

        :param data_path: directory to the dataset
        :param max_disp: maximum disparity before downscaling
        :param downscale: downscaling factor
        :param resized_height: final image height after downscaling and resizing
        :param resized_width: final image width after downscaling and resizing
        :param conf_thres: threshold for confidence score
        :param is_train: flag to indicate if this dataset is for training or not (just a placeholder)
        :param imgnet_norm: if set to True, the RGB images will be normalized by ImageNet's statistics
        :param sanity: if set to True, only includes 1 data point. Mostly used to debug the model
        """
        super(Kitti2012FullDataset, self).__init__(data_path, max_disp, downscale, resized_height, resized_width,
                                                   conf_thres, is_train, imgnet_norm, sanity)
        self.data_path = self.data_path.replace("_full", "")
        self.data_path = os.path.join(self.data_path, "training")
        frame_list = os.listdir(os.path.join(self.data_path, "colored_0"))
        self.data_list = []
        for f in frame_list:
            if "_10" in f:
                self.data_list.append(f)

        if self.sanity:  # only keep the first entry for sanity check
            self.data_list.sort()
            self.data_list = [self.data_list[0]]

    def _get_gt_disp(self, disp_path):
        """
        Get the ground truth disparity

        :param disp_path: directory to the ground truth disparity
        :return: ground truth disparity as a PyTorch tensor
        """
        disp = cv2.imread(disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        disp = disp / 256.0
        disp = self.to_tensor(disp).float()
        disp[disp != disp] = 0  # set all pixels with NaN to zero
        disp[disp == float('inf')] = 0  # set all pixels with inf or -inf to zero
        disp[disp >= self.max_disp] = 0  # set all disparity larger than the preset maximum to 0
        disp /= self.downscale
        return disp

    def __getitem__(self, index):
        """
        Get a data sample

        :param index: index for the data list
        :return: a stack of input data in tensor form including left rgb, right rgb, raw disparity, confidence mask, \
                 frame id and ground truth disparity if the dataset is for validation
        """
        frame = self.data_list[index]
        do_color_aug = self.is_train and random.random() > 0.5 and (not self.sanity)
        raw_inputs = {}
        l_rgb_path = os.path.join(self.data_path, "colored_0", frame)
        r_rgb_path = os.path.join(self.data_path, "colored_1", frame)
        disp_path = os.path.join(self.data_path, "raw_disp", frame.replace(".png", ".npy"))
        conf_path = os.path.join(self.data_path, "conf", frame.replace(".png", ".npy"))

        raw_inputs['l_rgb'] = self._get_rgb(l_rgb_path)
        raw_inputs['r_rgb'] = self._get_rgb(r_rgb_path)
        raw_inputs['raw_disp'] = self._get_disp(disp_path)
        raw_inputs['mask'] = self._get_conf(conf_path)

        # Need to override orig_height and orig_width for KITTI since the image size may vary in the dataset
        _, self.orig_height, self.orig_width = raw_inputs['l_rgb'].size()
        assert self.orig_width % self.downscale == 0 and self.orig_height % self.downscale == 0, \
            "original image size not divisible by downscaling factor"

        # if not self.is_train:
        gt_occ_disp_path = os.path.join(self.data_path, "disp_occ", frame)
        gt_noc_disp_path = os.path.join(self.data_path, "disp_noc", frame)
        raw_inputs['gt_disp'] = self._get_gt_disp(gt_occ_disp_path)
        raw_inputs['noc_gt_disp'] = self._get_gt_disp(gt_noc_disp_path)

        if ((self.orig_width // self.downscale - self.resized_width) >= 0 and (
                self.orig_height // self.downscale - self.resized_height) > 0) or (
                (self.orig_width // self.downscale - self.resized_width) > 0 and (
                self.orig_height // self.downscale - self.resized_height) >= 0):
            inputs = self._crop_inputs(raw_inputs)
        elif ((self.orig_width // self.downscale - self.resized_width) <= 0 and (
                self.orig_height // self.downscale - self.resized_height) < 0) or (
                (self.orig_width // self.downscale - self.resized_width) < 0 and (
                self.orig_height // self.downscale - self.resized_height) <= 0):
            inputs = self._pad_inputs(raw_inputs)
        else:
            print("Inconsistent image resizing scheme")
            raise RuntimeError

        inputs['raw_disp'] = self._normalize_disp(inputs['raw_disp'])

        if do_color_aug:
            inputs['l_rgb'], inputs['r_rgb'] = self._data_augmentation(inputs['l_rgb'], inputs['r_rgb'])

        if self.imgnet_norm:
            inputs['l_rgb'], inputs['r_rgb'] = self._normalize_rgb(inputs['l_rgb'], inputs['r_rgb'])

        # for saving predicted disaprity
        inputs['frame_id'] = frame

        return inputs
