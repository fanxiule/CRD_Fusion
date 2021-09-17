import os
import argparse


class TrainOptions:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(description="CRD_Fusion Training Options")

        # DIRECTORIES
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="directory where datasets are saved",
                                 default=os.getenv('data_path'))
        # default="/home/xfan/Documents/Datasets/")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="directory to save trained model and Tensorboard event",
                                 default="models")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the folder where the model will be saved in",
                                 default="crd_fusion")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="SceneFlow",
                                 choices=["kitti2015", "kitti2012", "kitti2015_full", "kitti2012_full", "SceneFlow"])
        self.parser.add_argument("--resized_height",
                                 type=int,
                                 help="image height after resizing",
                                 default=256)
        self.parser.add_argument("--resized_width",
                                 type=int,
                                 help="image width after resizing",
                                 default=512)
        self.parser.add_argument("--downscale",
                                 type=int,
                                 help="downscaling factor before image resizing",
                                 default=1)
        self.parser.add_argument("--max_disp",
                                 type=int,
                                 help="maximum disparity for prediction at the full spatial resolution",
                                 default=192)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=0.001)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of training epochs",
                                 default=15)
        self.parser.add_argument("--scheduler_step",
                                 type=int,
                                 help="step size in terms of epochs for the scheduler to change learning rate",
                                 default=10)
        self.parser.add_argument("--lr_change_rate",
                                 type=float,
                                 help="the multiplier to change the existing learning rate",
                                 default=0.1)

        # ABLATION options
        # input data
        self.parser.add_argument("--conf_threshold",
                                 type=float,
                                 help="if a confidence score is lower than the threshold, it will be replaced by 0",
                                 default=0.8)
        self.parser.add_argument("--imagenet_norm",
                                 action="store_true",
                                 help="if set, the RGB images are normalized by ImageNet mean and variance")
        # model structure
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
        self.parser.add_argument("--loss_conf",
                                 action="store_true",
                                 help="if set, confidence is applied to loss computation")
        self.parser.add_argument("--baseline",
                                 action="store_true",
                                 help="if set, the baseline model is used")
        # occlusion handling
        self.parser.add_argument("--occ_detection",
                                 action="store_true",
                                 help="if set, occlusion mask is calculated and applied in loss function")
        self.parser.add_argument("--occ_threshold",
                                 type=float,
                                 help="threshold for occlusion mask, used in post processing",
                                 default=0.8)
        self.parser.add_argument("--post_processing",
                                 action="store_true",
                                 help="if set, post processing is NOT applied")
        # loss
        self.parser.add_argument("--supervised",
                                 action="store_true",
                                 help="if set, the model is trained with supervised loss")
        self.parser.add_argument("--supervision_weight",
                                 type=float,
                                 help="weight for the supervision term in training loss calculation",
                                 default=0.7)
        self.parser.add_argument("--photo_weight",
                                 type=float,
                                 help="weight for the photometric loss in training loss calculation",
                                 default=3.0)
        self.parser.add_argument("--smooth_weight",
                                 type=float,
                                 help="weight for the smoothness loss in training loss calculation",
                                 default=0.45)  # 0.2 may work better for KITTI
        self.parser.add_argument("--occ_weight",
                                 type=float,
                                 help="weight for the cross entropy loss of the occlusion masks",
                                 default=0.75)  # may need higher > 0.5 for KITTI, 0.7 is too much
        self.parser.add_argument("--occ_epoch",
                                 type=int,
                                 help="after the specified epoch number, occlusion is applied to supervision and smoothness "
                                      "losses. Set it to negative to disable this action",
                                 default=-1)

        # SYSTEM options
        self.parser.add_argument("--device",
                                 type=str,
                                 help="training device",
                                 choices=["cpu", "cuda"],
                                 default="cuda")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)

        # LOADING Options
        self.parser.add_argument("--pretrained_model_path",
                                 type=str,
                                 help="path to the pretrained model to be loaded for fine tuning or evaluation")

        # LOGGING options
        self.parser.add_argument("--early_log_frequency",
                                 type=int,
                                 help="tensorboard logging frequency in number of batches in the early phase",
                                 default=200)
        self.parser.add_argument("--late_log_frequency",
                                 type=int,
                                 help="tensorboard logging frequency in number of batches in the late phase",
                                 default=2000)
        self.parser.add_argument("--early_late_split",
                                 type=int,
                                 help="logging is split into early and late phase at this batch id",
                                 default=4000)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="frequency in number of epochs to save a trained model",
                                 default=3)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
