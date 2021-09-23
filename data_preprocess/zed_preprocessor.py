import os
import random
import numpy as np
import torch
from PIL import Image
from .data_preprocessor import DataPreprocessor

from utils import decode_disp, save_conf


class ZEDPreprocessor(DataPreprocessor):
    """
    Preprocessor for the dataset collected by a ZED camera
    """

    def __init__(self, dataset_path, device, full_ZSAD, split, seed):
        """
        Preprocessor for the dataset collected by a ZED camera
        """
        super(ZEDPreprocessor, self).__init__(dataset_path, None, None, None, device, full_ZSAD)
        # None as place holder for max_disp, block_sz, match_method since no stereo matching is needed to generate raw disparity
        self.split = split
        self.train_split = os.path.join(self.dataset_path, "train.txt")
        self.val_split = os.path.join(self.dataset_path, "val.txt")
        self.seed = seed
        self.l_im_path = os.path.join(self.dataset_path, "left")
        self.r_im_path = os.path.join(self.dataset_path, "right")
        self.disp_path = os.path.join(self.dataset_path, "raw_disp")
        self.conf_path = os.path.join(self.dataset_path, "conf")
        if not os.path.exists(self.conf_path):
            os.makedirs(self.conf_path)

    def _split_dataset(self, frame_list):
        """
        Split the dataset into a training set and a validation set

        :param frame_list: a list of all frames in the dataset
        :return: None
        """
        # open and close train and validation txt file to clear the contents
        open(self.train_split, 'w').close()
        open(self.val_split, 'w').close()

        random.seed(self.seed)
        random.shuffle(frame_list)
        frame_num = len(frame_list)
        train_num = int(frame_num * self.split)
        train_list = frame_list[:train_num]
        val_list = frame_list[train_num:]

        for f in train_list:
            with open(self.train_split, 'a') as tf:
                tf.write("%s\n" % f)
                tf.close()

        for f in val_list:
            with open(self.val_split, 'a') as vf:
                vf.write("%s\n" % f)
                vf.close()

    def _process_frame(self, l_path, r_path, disp_path, conf_path):
        """
        Process a frame by calculating the confidence mask

        :param l_path: path to the left image
        :param r_path: path to the right image
        :param disp_path: path to save the predicted disparity
        :param conf_path: path to save the confidence mask
        :return: None
        """
        l_im = Image.open(l_path).convert("RGB")
        r_im = Image.open(r_path).convert("RGB")
        disp = decode_disp(disp_path)
        l_im_tensor = self.to_tensor(l_im).unsqueeze(dim=0).to(self.device)
        r_im_tensor = self.to_tensor(r_im).unsqueeze(dim=0).to(self.device)
        disp_tensor = self.to_tensor(disp).unsqueeze(dim=0).to(torch.float32).to(self.device)
        conf_tensor = self.conf_gen.cal_confidence(l_im_tensor, r_im_tensor, disp_tensor)
        conf = conf_tensor.detach().cpu().numpy()
        conf = np.squeeze(conf)
        save_conf(conf, conf_path)

    def preprocess(self):
        """
        Preprocess the dataset

        :return: None
        """
        frame_list = os.listdir(self.l_im_path)
        frame_list.sort()
        self._split_dataset(frame_list)

        frame_count = 0
        for f in frame_list:
            frame_count += 1
            print("Processing frame %d/%d" % (frame_count, len(frame_list)))
            l_im_frame = os.path.join(self.l_im_path, f)
            r_im_frame = os.path.join(self.r_im_path, f)
            disp_frame = os.path.join(self.disp_path, f)
            conf_frame = os.path.join(self.conf_path, f)
            self._process_frame(l_im_frame, r_im_frame, disp_frame, conf_frame)
