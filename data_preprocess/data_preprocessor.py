import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from .conf_generation import ConfGeneration


class DataPreprocessor:
    """
    Superclass to preprocess a given dataset
    """

    def __init__(self, dataset_path, max_disp, block_sz, match_method, device, full_ZSAD):
        """
        Base class to preprocess datasets to generate raw disparity and confidence maps

        :param dataset_path: directory to the dataset
        :param max_disp: maximum disparity to check when matching
        :param block_sz: block size for the stereo algorithm
        :param match_method: choose between local_BM and SG_BM for local or semi-global block matching
        :param device: device to compute confidence measures, choose between cuda and cpu
        :param full_ZSAD: if set to True, ZSAD is performed on a 3x3 window. Otherwise, it is performed on a partial 3x3 window
        """
        self.dataset_path = dataset_path
        self.max_disp = max_disp
        self.match_method = match_method
        self.to_tensor = transforms.ToTensor()
        self.device = device
        self.conf_gen = ConfGeneration(self.device, full_ZSAD)

        if self.match_method == "localBM":
            self.stereo = cv2.StereoBM_create(numDisparities=self.max_disp, blockSize=block_sz)
        elif self.match_method == "SGBM":
            pre_filter_cap = 15
            p1 = block_sz * block_sz * 8
            p2 = block_sz * block_sz * 64
            uniqueness_ratio = 20
            speckle_window_size = 150
            speckle_range = 1
            disp_max_diff = 1
            full_dp = 1
            self.stereo = cv2.StereoSGBM_create(numDisparities=self.max_disp, blockSize=block_sz, P1=p1, P2=p2,
                                                disp12MaxDiff=disp_max_diff, preFilterCap=pre_filter_cap,
                                                uniquenessRatio=uniqueness_ratio, speckleWindowSize=speckle_window_size,
                                                speckleRange=speckle_range, mode=full_dp)
        else:
            self.stereo = None

    def _cal_disp(self, l_im, r_im):
        """
        Calculate the disparity and a validity mask for given left and right input images using stereo matching

        :param l_im: left image in the default OpenCV BGR format
        :param r_im: right image in the default OpenCV BGR format
        :return: disparity map & a binary validity mask to indicate if a pixel has been matched or not
        """
        if self.match_method == "localBM":  # local_BM and needs grayscale images
            l_im = cv2.cvtColor(l_im, cv2.COLOR_RGB2GRAY)  # PIL opens image in RGB
            r_im = cv2.cvtColor(r_im, cv2.COLOR_RGB2GRAY)
        disp = self.stereo.compute(l_im, r_im)
        disp = disp / 16.0

        # eliminate negative disparity and disparity very close to 0
        mask = np.copy(disp)
        mask[mask >= 0.001] = 1
        mask[mask < 0.001] = 0
        return disp, mask

    @staticmethod
    def _save_prediction(disp_dir, conf_dir, disp, conf):
        """
        Save predicted disparity and confidence maps

        :param disp_dir: file path to save the disparity
        :param conf_dir: file path to save the confidence map
        :param disp: predicted raw disparity
        :param conf: confidence map
        :return: None
        """
        np.save(disp_dir, disp)
        np.save(conf_dir, conf)

    def _process_frame(self, l_path, r_path, disp_path, conf_path):
        """
        Process a frame by calculating the raw disparity and confidence mask

        :param l_path: path to the left image
        :param r_path: path to the right image
        :param disp_path: path to save the predicted disparity
        :param conf_path: path to save the confidence mask
        :return: None
        """
        l_im = Image.open(l_path).convert("RGB")  # open images by PIL to be consistent with dataloader
        r_im = Image.open(r_path).convert("RGB")
        disp, mask = self._cal_disp(np.asarray(l_im), np.asarray(r_im))
        l_im_tensor = self.to_tensor(l_im).unsqueeze(dim=0).to(self.device)
        r_im_tensor = self.to_tensor(r_im).unsqueeze(dim=0).to(self.device)
        disp_tensor = self.to_tensor(disp).unsqueeze(dim=0).to(torch.float32).to(self.device)
        mask_tensor = self.to_tensor(mask).unsqueeze(dim=0).to(torch.float32).to(self.device)
        conf_tensor = self.conf_gen.cal_confidence(l_im_tensor, r_im_tensor, disp_tensor, mask_tensor)
        conf = conf_tensor.detach().cpu().numpy()
        conf = np.squeeze(conf)
        self._save_prediction(disp_path, conf_path, disp, conf)

    def preprocess(self):
        raise NotImplementedError
