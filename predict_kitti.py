import os
import time
import argparse

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import KITTI2012TestDataset, KITTI2015TestDataset
from utils import post_process
from crd_fusion_net import CRDFusionNet
from data_preprocess import ConfGeneration


def parse_args():
    """
    Parse options for predicting KITTI stereo

    :return: options
    """
    parser = argparse.ArgumentParser(description="CRD_Fusion KITTI Test Options")
    parser.add_argument("--data_path", type=str, help="directory where datasets are saved",
                        default=os.getenv('data_path'))
    # default="/home/xfan/Documents/Datasets/")
    parser.add_argument("--checkpt", type=str, help="directory to pretrained checkpoint files",
                        default="/home/xfan/Documents/Avidbots/Experiments/CRD_Fusion/models/crd_fusion/checkpts/weights_100")
    parser.add_argument("--log_dir", type=str, help="directory to save prediction", default="models")
    parser.add_argument("--model_name", type=str, help="name of folder to save prediction",
                        default="crd_fusion_test")
    parser.add_argument("--device", type=str, help="test device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dataset", type=str, help="select a KITTI test set", default="kitti2015",
                        choices=["kitti2015", "kitti2012"])
    parser.add_argument("--max_disp", type=int, help="max disparity range according to the checkpt file", default=128)
    parser.add_argument("--resized_height", type=int, help="image height after resizing", default=376)
    parser.add_argument("--resized_width", type=int, help="image width after resizing", default=1248)
    parser.add_argument("--conf_threshold", type=float, help="confidence threshold for raw disparity", default=0.8)
    parser.add_argument("--occ_threshold", type=float, help="threshold for occlusion mask", default=0.8)
    parser.add_argument("--post_processing", action="store_true", help="if set, post processing is applied")
    parser.add_argument("--save_pred", action="store_true", help="if set, the predictions are saved")
    return parser.parse_args()


def save_pred(pred_disp, occ, conf, thrs, frame_id, log_path):
    """
    Save predictions

    :param pred_disp: predicted disparity map
    :param occ: occlusion mask
    :param conf: confidence mask
    :param thrs: threshold for confidence mask
    :param frame_id: frame id to name the files
    :param log_path: save directory
    :return: None
    """
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # save disp
    pred_disp = pred_disp.detach().cpu().numpy()
    pred_disp = np.squeeze(pred_disp)
    pred_disp = pred_disp * 256
    pred_disp[pred_disp == 0] = 1
    pred_disp[pred_disp < 0] = 0
    pred_disp[pred_disp > 65535] = 0
    pred_disp = pred_disp.astype(np.uint16)
    filename = os.path.join(log_path, frame_id)
    cv2.imwrite(filename, pred_disp)

    # save occ
    occ = occ.detach().cpu().numpy()
    occ = np.squeeze(occ)
    occ = (255 * occ).astype(np.uint8)
    filename = os.path.join(log_path, "occ_%s" % frame_id)
    cv2.imwrite(filename, occ)

    # save conf
    conf = conf.detach().cpu().numpy()
    conf = np.squeeze(conf)
    conf[conf < thrs] = 0
    conf = (255 * conf).astype(np.uint8)
    filename = os.path.join(log_path, "conf_%s" % frame_id)
    cv2.imwrite(filename, conf)


def predict(opts):
    """
    Predict KITTI stereo

    :param opts: options
    :return: None
    """
    log_path = os.path.join(opts.log_dir, opts.model_name)
    feature_scale_list = [0, 1, 2, 3]
    model = CRDFusionNet(feature_scale_list, opts.max_disp, opts.resized_height, opts.resized_width, False, True, True)
    if opts.checkpt is not None and os.path.isdir(opts.checkpt):
        model.load_model(opts.checkpt)
    else:
        model.init_model()
    model.to(opts.device)

    dataset_list = {'kitti2015': KITTI2015TestDataset, 'kitti2012': KITTI2012TestDataset}
    dataset = dataset_list[opts.dataset]
    data_path = os.path.join(opts.data_path, opts.dataset)
    predict_dataset = dataset(data_path, opts.max_disp, opts.resized_height, opts.resized_width, opts.conf_threshold,
                              True, False)
    predict_loader = DataLoader(predict_dataset, 1, False, num_workers=0, pin_memory=True, drop_last=False)
    conf_gen = ConfGeneration(opts.resized_height, opts.resized_width, opts.device, False)

    num_test_samples = len(predict_dataset)

    print("Begin predicting %s" % opts.model_name)
    print("Use checkpt in: %s" % opts.checkpt)
    print("Save predicted disparity maps in %s" % log_path)
    print("Save predictions: %r" % opts.save_pred)
    print("Dataset: %s" % opts.dataset)
    print("Input size: %d x %d" % (opts.resized_height, opts.resized_width))
    print("Total number of test samples %d" % num_test_samples)
    print("Max disp: %d" % opts.max_disp)
    print("Conf threshold: %.2f" % opts.conf_threshold)
    print("Post processing: %r" % opts.post_processing)

    print("-------------Start Prediction-------------")
    duration = 0
    model.eval()
    with torch.no_grad():
        for batch_id, inputs in enumerate(predict_loader):
            for k, v in inputs.items():
                if k != "frame_id" and k != "left_pad" and k != "top_pad":
                    inputs[k] = v.to(opts.device)
            batch_start_time = time.time()
            inputs['mask'] = conf_gen.cal_confidence(inputs['l_rgb_non_norm'], inputs['r_rgb_non_norm'],
                                                     opts.max_disp * inputs['raw_disp'])
            outputs = model(inputs['l_rgb'], inputs['r_rgb'], inputs['raw_disp'], inputs['mask'])
            # undo padding on prediction
            outputs['refined_disp0'] = outputs['refined_disp0'][:, :, inputs['top_pad']:, inputs['left_pad']:]
            outputs['occ0'] = outputs['occ0'][:, :, inputs['top_pad']:, inputs['left_pad']:]
            inputs['mask'] = inputs['mask'][:, :, inputs['top_pad']:, inputs['left_pad']:]
            if opts.post_processing:
                outputs['final_disp'] = post_process(outputs['refined_disp0'], outputs['occ0'], opts.occ_threshold)
            else:
                outputs['final_disp'] = outputs['refined_disp0']
            duration += (time.time() - batch_start_time)
            if opts.save_pred:
                save_pred(outputs['final_disp'], outputs['occ0'], inputs['mask'], opts.conf_threshold,
                          inputs['frame_id'][0], log_path)
    print("Frame rate: %.4f" % (num_test_samples / duration))


if __name__ == "__main__":
    args = parse_args()
    predict(args)
