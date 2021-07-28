import os
import argparse
import data_preprocess


def parse_arguments():
    # options for running initial stereo matching and generate confidence maps
    parser = argparse.ArgumentParser(description="Initial Stereo Matching")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="Directory to the dataset",
                        default=os.getenv('data_path'))
    # default="/home/xfan/Documents/Datasets/")
    parser.add_argument("--dataset_name",
                        type=str,
                        help="Name of the dataset",
                        choices=["kitti2015", "kitti2012", "SceneFlow"],
                        default="SceneFlow")
    parser.add_argument("--max_disp",
                        type=int,
                        help="Maximum disparity for stereo matching",
                        default=128)
    parser.add_argument("--block_size",
                        type=int,
                        help="Block size for stereo matching",
                        default=3)
    parser.add_argument("--match_method",
                        type=str,
                        help="Choose stereo matching methods to generate the disparity maps",
                        choices=["localBM", "SGBM"],
                        default="SGBM")
    parser.add_argument("--train_val_split_per",
                        type=float,
                        help="The ratio used to split data into train/val set. Set it to None if not needed. If set to "
                             "0.8, 80% of data is used for training and the rest for validation",
                        default=0.8)
    opt = parser.parse_args()
    return opt


def main(opt):
    preprocessor_dict = {"kitti2015": data_preprocess.Kitti15Preprocessor,
                         "kitti2012": data_preprocess.Kitti12Preprocessor,
                         "SceneFlow": data_preprocess.SceneFlowPreprocessor}
    preprocessor = preprocessor_dict[opt.dataset_name]
    dataset_path = os.path.join(opt.dataset_path, opt.dataset_name)
    if not os.path.exists(dataset_path):
        print("Dataset not found")
        raise RuntimeError
    if opt.dataset_name == "kitti2015" or opt.dataset_name == "kitti2012":
        assert opt.train_val_split_per is not None, "Need to specify the train and val split for this dataset"
        preprocessor = preprocessor(dataset_path, opt.max_disp, opt.block_size, opt.match_method,
                                    opt.train_val_split_per)
    else:
        preprocessor = preprocessor(dataset_path, opt.max_disp, opt.block_size, opt.match_method)
    print("Start preprocessing %s dataset" % opt.dataset_name)
    preprocessor.preprocess()


if __name__ == '__main__':
    options = parse_arguments()
    main(options)
