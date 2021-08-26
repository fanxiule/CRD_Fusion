import random
import torch
import torch.utils.data as data
import torchvision.transforms.functional as f
import torch.nn.functional as nf
import numpy as np

from torchvision import transforms
from PIL import Image

random.seed(75)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CRDFusionDataset(data.Dataset):
    """
    Super class for datasets
    """

    def __init__(self, data_path, max_disp, downscale, resized_height, resized_width, conf_thres, is_train,
                 imgnet_norm=True, sanity=False):
        """
        Base class to prepare data for training and testing

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
        super(CRDFusionDataset, self).__init__()
        self.data_path = data_path
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.orig_height = None
        self.orig_width = None
        self.max_disp = max_disp
        self.downscale = downscale
        self.conf_thres = conf_thres
        self.is_train = is_train
        self.data_list = None
        self.imgnet_norm = imgnet_norm
        self.sanity = sanity

        # data augmentation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

        # common operation
        self.to_tensor = transforms.ToTensor()
        self.color_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def _get_rgb(self, rgb_path):
        """
        Open an RGB image and convert it to a torch tensor

        :param rgb_path: path to the RGB image
        :return: an RGB image tensor
        """
        img = Image.open(rgb_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')  # in case some png files are in RGBA format
        img = self.to_tensor(img)
        return img

    def _get_disp(self, disp_path):
        """
        Open a disparity map and convert it to a torch tensor.

        :param disp_path: directory to the disparity map
        :return: a disparity tensor
        """
        disp = np.load(disp_path)
        disp = self.to_tensor(disp).float()
        return disp

    def _get_conf(self, conf_path):
        """
        Open a confidence map and convert it to a torch tensor

        :param conf_path: directory to the confidence map
        :return: a confidence tensor
        """
        conf = np.load(conf_path)
        conf = self.to_tensor(conf)
        conf[conf < self.conf_thres] = 0
        return conf

    def _normalize_disp(self, disp):
        """
        Normalize disparity to the range of [0, 1] according to the max disparity

        :param disp: disparity tensor
        :return: normalized disparity tensor
        """
        norm_disp = disp / self.downscale
        max_disp = self.max_disp / self.downscale
        norm_disp = torch.clamp(norm_disp, min=0, max=max_disp)
        norm_disp = norm_disp / max_disp
        return norm_disp

    def _data_augmentation(self, l_img, r_img):
        """
        Data augmentation to the RGB images based on random color jittering

        :param l_img: left RGB tensor
        :param r_img: right RGB tensor
        :return: augmented left and right RGB tensors
        """
        aug_l_img = self.color_jitter(l_img)
        aug_r_img = self.color_jitter(r_img)
        return aug_l_img, aug_r_img

    def _normalize_rgb(self, l_img, r_img):
        """
        Normalize left and right images according to ImageNet mean and variance

        :param l_img: left RGB tensor
        :param r_img: right RGB tensor
        :return: normalized left and right RGB tensors
        """
        norm_l_img = self.color_norm(l_img)
        norm_r_img = self.color_norm(r_img)
        return norm_l_img, norm_r_img

    def _crop_inputs(self, inputs):
        """
        Crop input images according to downscaling factor and specified size.

        :param inputs: stack of input images including RGB, disparity, and confidence
        :return: a stack of cropped inputs
        """
        resized_inputs = {}
        if not self.is_train or self.sanity:
            # col_ind = 150
            # row_ind = 28
            col_ind = self.orig_width // self.downscale - self.resized_width
            row_ind = self.orig_height // self.downscale - self.resized_height
        else:
            col_ind = random.randint(0, self.orig_width // self.downscale - self.resized_width)
            row_ind = random.randint(0, self.orig_height // self.downscale - self.resized_height)
        for im_k, im_v in inputs.items():
            resized_inputs[im_k] = f.resize(im_v,
                                            [self.orig_height // self.downscale, self.orig_width // self.downscale],
                                            f.InterpolationMode.NEAREST)
            resized_inputs[im_k] = f.crop(resized_inputs[im_k], row_ind, col_ind, self.resized_height,
                                          self.resized_width)
        return resized_inputs

    def _pad_inputs(self, inputs):
        """
        Pad input images according to downscaling factor and predefined image size

        :param inputs: stack of input images including RGB, disparity, and confidence
        :return: a stack of padded inputs
        """
        resized_inputs = {'top_pad': self.resized_height - self.orig_height // self.downscale,
                          'left_pad': self.resized_width - self.orig_width // self.downscale}
        for im_k, im_v in inputs.items():
            resized_inputs[im_k] = f.resize(im_v,
                                            [self.orig_height // self.downscale, self.orig_width // self.downscale],
                                            f.InterpolationMode.NEAREST)
            resized_inputs[im_k] = nf.pad(torch.unsqueeze(resized_inputs[im_k], dim=0),
                                          (resized_inputs['left_pad'], 0, resized_inputs['top_pad'], 0),
                                          'replicate')
            resized_inputs[im_k] = torch.squeeze(resized_inputs[im_k], dim=0)

        return resized_inputs

    def __len__(self):
        """
        Total number of data samples

        :return: total number of data samples
        """
        return len(self.data_list)

    def __getitem__(self, index):
        raise NotImplementedError
