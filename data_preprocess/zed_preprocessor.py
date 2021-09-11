import os
import random
from .data_preprocessor import DataPreprocessor


class ZEDPreprocessor(DataPreprocessor):
    """
    Preprocessor for the dataset collected by a ZED camera
    """

    def __init__(self, dataset_path, split, seed):
        """
        Preprocessor for the dataset collected by a ZED camera
        """
        super(ZEDPreprocessor, self).__init__(dataset_path, None, None, None, None, False)
        # place folder for max_disp, block_sz, match_method, device, full_ZSAD since raw disparity and confidence are both available
        self.split = split
        self.train_split = os.path.join(self.dataset_path, "train.txt")
        self.val_split = os.path.join(self.dataset_path, "val.txt")
        self.seed = seed
        self.l_im_path = os.path.join(self.dataset_path, "left")

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
        frame_list = os.listdir(self.l_im_path)
        frame_list.sort()
        self._split_dataset(frame_list)
