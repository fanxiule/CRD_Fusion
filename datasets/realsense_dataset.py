import os
import random
import cv2
import torch
from PIL import Image
from .crd_fusion_dataset import CRDFusionDataset
from utils import decode_disp


class RealSenseDataset(CRDFusionDataset):
    def __init__(self, data_path, max_disp, downscale, resized_height, resized_width, conf_thres, is_train,
                 imgnet_norm=True, sanity=False):
        """
        Dataset to load and prepare data for training/validation from a dataset collected by a RealSense camera

        :param data_path: directory to the dataset
        :param max_disp: maximum disparity before downscaling
        :param downscale: downscaling factor
        :param resized_height: final image height after downscaling and resizing
        :param resized_width: final image width after downscaling and resizing
        :param conf_thres: threshold for confidence score
        :param is_train: flag to indicate if this dataset is for training or not
        :param imgnet_norm: if set to True, the RGB images will be normalized by ImageNet's statistics
        :param sanity: if set to True, only includes 1 data point. Mostly used to debug the model
        """
        super(RealSenseDataset, self).__init__(data_path, max_disp, downscale, resized_height, resized_width,
                                               conf_thres, is_train, imgnet_norm, sanity)
        if self.is_train:
            with open(os.path.join(self.data_path, "train.txt")) as f:
                self.data_list = f.readlines()
        else:
            with open(os.path.join(self.data_path, "val.txt")) as f:
                self.data_list = f.readlines()
        self.data_list = [d.strip("\n") for d in self.data_list]

        if self.sanity:  # only keep the first entry for sanity check
            self.data_list.sort()
            self.data_list = [self.data_list[0]]

    def _get_rgb(self, rgb_path):
        """
        Open an RGB image and convert it to a torch tensor

        :param rgb_path: path to the RGB image
        :return: an RGB image tensor
        """
        img = Image.open(rgb_path).convert("RGB")
        img = self.to_tensor(img)
        return img

    def _get_disp(self, disp_path):
        """
        Open a disparity map and convert it to a torch tensor.

        :param disp_path: directory to the disparity map
        :return: a disparity tensor
        """
        disp = decode_disp(disp_path)
        disp = self.to_tensor(disp).to(torch.float32)
        return disp

    def _get_conf(self, conf_path):
        """
        Open a confidence map and convert it to a torch tensor

        :param conf_path: directory to the confidence map
        :return: a confidence tensor
        """
        conf = cv2.imread(conf_path, cv2.IMREAD_GRAYSCALE)
        conf = conf / 255.0
        conf = self.to_tensor(conf).to(torch.float32)
        conf[conf < self.conf_thres] = 0
        return conf

    def __getitem__(self, index):
        """
        Get a data sample

        :param index: index for the data list
        :return: a stack of input data in tensor form including left rgb, right rgb, raw disparity, confidence mask, \
                 and frame id
        """
        frame = self.data_list[index]
        do_color_aug = self.is_train and random.random() > 0.5 and (not self.sanity)
        raw_inputs = {}
        l_rgb_path = os.path.join(self.data_path, "left", frame)
        r_rgb_path = os.path.join(self.data_path, "right", frame)
        disp_path = os.path.join(self.data_path, "raw_disp", frame)
        conf_path = os.path.join(self.data_path, "conf", frame)
        gt_path = os.path.join(self.data_path, "gt", frame)

        raw_inputs['l_rgb'] = self._get_rgb(l_rgb_path)
        raw_inputs['r_rgb'] = self._get_rgb(r_rgb_path)
        raw_inputs['raw_disp'] = self._get_disp(disp_path)
        raw_inputs['mask'] = self._get_conf(conf_path)
        if os.path.exists(gt_path):
            raw_inputs['gt_disp'] = self._get_disp(gt_path)

        _, self.orig_height, self.orig_width = raw_inputs['l_rgb'].size()
        assert self.orig_width % self.downscale == 0 and self.orig_height % self.downscale == 0, \
            "original image size not divisible by downscaling factor"

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
        elif (self.orig_width // self.downscale == self.resized_width) and (
                self.orig_height // self.downscale == self.resized_height):
            inputs = raw_inputs
        else:
            print("Inconsistent image resizing scheme")
            raise RuntimeError

        if not self.is_train:
            inputs['raw_disp_non_norm'] = torch.clone(inputs['raw_disp'])
            inputs['l_rgb_non_norm'] = torch.clone(inputs['l_rgb'])
            inputs['r_rgb_non_norm'] = torch.clone(inputs['r_rgb'])

        inputs['raw_disp'] = self._normalize_disp(inputs['raw_disp'])

        if do_color_aug:
            inputs['l_rgb'], inputs['r_rgb'] = self._data_augmentation(inputs['l_rgb'], inputs['r_rgb'])

        if self.imgnet_norm:
            inputs['l_rgb'], inputs['r_rgb'] = self._normalize_rgb(inputs['l_rgb'], inputs['r_rgb'])

        # for saving predicted disaprity
        inputs['frame_id'] = frame.strip(".png")

        return inputs
