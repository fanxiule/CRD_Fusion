import os
import argparse


class EvalOptions:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(description="CRD_Fusion Validation Options")
        # DIRECTORIES
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="directory where datasets are saved",
                                 default=os.getenv('data_path'))
        # default=os.path.expanduser("~/Documents/Datasets/")
        self.parser.add_argument("--checkpt",
                                 type=str,
                                 help="directory to pretrained checkpoint files",
                                 default="models/SceneFlow")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="directory to save Tensorboard event and optionally the predicted disparity map",
                                 default="models")

        # DATASET options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="SceneFlow",
                                 choices=["kitti2015", "kitti2012", "SceneFlow", "realsense", "zed"])
        self.parser.add_argument("--resized_height",
                                 type=int,
                                 help="image height after resizing",
                                 default=544)
        self.parser.add_argument("--resized_width",
                                 type=int,
                                 help="image width after resizing",
                                 default=960)
        self.parser.add_argument("--downscale",
                                 type=int,
                                 help="downscaling factor before image resizing",
                                 default=1)
        self.parser.add_argument("--max_disp",
                                 type=int,
                                 help="maximum disparity for prediction at the full spatial resolution. Must agree with "
                                      "the checkpoint files",
                                 default=192)

        # EVALUATION options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the folder where the tensorboard event and/or predicted disparity maps "
                                      "will be saved",
                                 default="crd_fusion_eval")
        self.parser.add_argument("--save_pred",
                                 action="store_true",
                                 help="if set, the predicted disparity maps and occlusion masks are saved in .npy format")
        self.parser.add_argument("--device",
                                 type=str,
                                 help="evaluation device",
                                 choices=["cpu", "cuda"],
                                 default="cuda")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="tensorboard logging frequency in number of batches",
                                 default=10)

        # For ablation
        # THEY SHOULD BE SET SUCH THAT THEY ARE CONSISTENT WITH THE CHECKPOINT
        self.parser.add_argument("--conf_threshold",
                                 type=float,
                                 help="if a confidence score is lower than the threshold, it will be replaced by 0",
                                 default=0.8)
        self.parser.add_argument("--imagenet_norm",
                                 action="store_true",
                                 help="if set, the RGB images are normalized by ImageNet mean and variance")
        self.parser.add_argument("--feature_downscale",
                                 type=int,
                                 help="downscaling factor during feature extraction. If set to 3, the image wil be "
                                      "downscaled to 1/(2^3)=1/8 of the original resolution",
                                 choices=range(1, 4),
                                 default=3)
        self.parser.add_argument("--multi_step_upsample",
                                 action="store_true",
                                 help="if set, the coarse disparity map is upsampled gradually during refinement")
        self.parser.add_argument("--fusion",
                                 action="store_true",
                                 help="if set, raw disparity fusion is applied to the model")
        self.parser.add_argument("--baseline",
                                 action="store_true",
                                 help="if set, the baseline model is used")
        self.parser.add_argument("--occ_detection",
                                 action="store_true",
                                 help="if set, occlusion mask is calculated and applied in loss function")
        self.parser.add_argument("--occ_threshold",
                                 type=float,
                                 help="threshold for occlusion mask",
                                 default=0.8)
        self.parser.add_argument("--post_processing",
                                 action="store_true",
                                 help="if set, post processing is applied")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
