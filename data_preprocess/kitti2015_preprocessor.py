import os
import random
from .data_preprocessor import DataPreprocessor


class Kitti15Preprocessor(DataPreprocessor):
    """
    Preprocessor for the KITTI 2012 dataset
    """

    def __init__(self, dataset_path, max_disp, block_sz, match_method, device, full_ZSAD, split, seed):
        """
        Preprocessor for the Scene Flow dataset

        :param dataset_path: directory to the dataset
        :param max_disp: maximum disparity to check when matching
        :param block_sz: block size for the stereo algorithm
        :param match_method: choose between local_BM and SG_BM for local or semi-global block matching
        :param device: device to compute confidence measures, choose between cuda and cpu
        :param full_ZSAD: if set to True, ZSAD is performed on a 3x3 window. Otherwise, it is performed on a partial 3x3 window
        :param split: the ratio used to split data into train/val set
        :param seed: random seed for splitting the dataset
        """
        super(Kitti15Preprocessor, self).__init__(dataset_path, max_disp, block_sz, match_method, device, full_ZSAD)
        self.split = split
        self.train_split = os.path.join(self.dataset_path, "training", "train.txt")
        self.val_split = os.path.join(self.dataset_path, "training", "val.txt")
        self.seed = seed
        self.l_im_path = None
        self.r_im_path = None
        self.disp_path = None
        self.conf_path = None

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

    def preprocess(self):
        """
        Preprocess the dataset

        :return: None
        """
        scene_list = os.listdir(self.dataset_path)
        scene_count = 0
        for scene in scene_list:
            scene_count += 1
            self.l_im_path = os.path.join(self.dataset_path, scene, "image_2")
            self.r_im_path = os.path.join(self.dataset_path, scene, "image_3")
            self.disp_path = os.path.join(self.dataset_path, scene, "raw_disp")
            self.conf_path = os.path.join(self.dataset_path, scene, "conf")
            if not os.path.exists(self.disp_path):
                os.makedirs(self.disp_path)
            if not os.path.exists(self.conf_path):
                os.makedirs(self.conf_path)

            all_frames = os.listdir(self.l_im_path)
            all_frames.sort()
            frames = []
            for f in all_frames:
                if "_10" in f:
                    frames.append(f)
            if scene == "training":
                self._split_dataset(frames)

            frame_count = 0
            for f in frames:
                frame_count += 1
                print(
                    "Processing frame %d/%d in scene %d/%d" % (frame_count, len(frames), scene_count, len(scene_list)))
                l_im_frame = os.path.join(self.l_im_path, f)
                r_im_frame = os.path.join(self.r_im_path, f)
                disp_frame = os.path.join(self.disp_path, f.replace(".png", ".npy"))
                conf_frame = os.path.join(self.conf_path, f.replace(".png", ".npy"))
                self._process_frame(l_im_frame, r_im_frame, disp_frame, conf_frame)
