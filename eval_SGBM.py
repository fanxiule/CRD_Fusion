import os
import argparse
import torch
from torch.utils.data import DataLoader

import datasets


def parse_arguments():
    """
    Parse arguments for evaluation on SGBM results

    :return: arguments
    """
    parser = argparse.ArgumentParser(description="Options to evaluate SGBM")
    parser.add_argument("--data_path",
                        type=str,
                        help="Directory to the dataset",
                        default=os.getenv('data_path'))
    # default="/home/xfan/Documents/Datasets/")
    parser.add_argument("--dataset",
                        type=str,
                        help="Name of the dataset",
                        choices=["kitti2015", "kitti2012", "SceneFlow"],
                        default="SceneFlow")
    parser.add_argument("--max_disp",
                        type=int,
                        help="Maximum disparity to generate the SGBM results",
                        default=128)
    # to be consistent with the dataset classes
    parser.add_argument("--resized_height",
                        type=int,
                        help="image height after resizing",
                        default=544)
    parser.add_argument("--resized_width",
                        type=int,
                        help="image width after resizing",
                        default=960)

    args = parser.parse_args()
    return args


def cal_error(sgbm, gt, noc_gt=None):
    """
    Calculate errors for SGBM results

    :param sgbm: disparity map obtained by SGBM
    :param gt: ground truth disparity map for all pixels
    :param noc_gt: ground truth disparity map for noc pixels (for KITTI)
    :return: error metrics
    """
    error = {'validity': True}
    gt_valid = gt > 0
    gt_valid_pixel = torch.sum(gt_valid)
    if noc_gt is not None:
        noc_gt_valid = noc_gt > 0
        noc_gt_valid_pixel = torch.sum(noc_gt_valid)
    else:
        noc_gt_valid_pixel = None
    if gt_valid_pixel == 0 or (noc_gt_valid_pixel is not None and noc_gt_valid_pixel == 0):
        error['validity'] = False
        return error

    sgbm_valid = sgbm > 0
    sgbm_valid_pixels = torch.sum(sgbm_valid)
    total_pixels = torch.numel(sgbm)
    error['coverage'] = sgbm_valid_pixels / total_pixels * 100.0

    gt_diff = torch.abs(sgbm - gt)
    # For EPE, it only makes sense to consider pixels that are valid in both SGBM and GT
    # For Bad3, a strict case (pixels valid in both SGBM and GT) and a relaxed case (pixels valid only in GT).
    # The second case is more accurate since the SGBM algorithm also attempts to solve those invalid pixels
    strict_gt_valid = torch.logical_and(gt_valid, sgbm_valid)
    strict_gt_valid_pixel = torch.sum(strict_gt_valid)
    strict_gt_diff = torch.mul(gt_diff, strict_gt_valid)
    relaxed_gt_diff = torch.mul(gt_diff, gt_valid)
    error['gt_epe'] = torch.sum(strict_gt_diff) / strict_gt_valid_pixel
    strict_gt_bad3 = strict_gt_diff > 3.0
    error['strict_gt_bad3'] = torch.sum(strict_gt_bad3) / strict_gt_valid_pixel * 100.0
    relaxed_gt_bad3 = relaxed_gt_diff > 3.0
    error['relaxed_gt_bad3'] = torch.sum(relaxed_gt_bad3) / gt_valid_pixel * 100.0

    if noc_gt is not None:
        noc_gt_diff = torch.abs(sgbm - noc_gt)
        strict_noc_gt_valid = torch.logical_and(noc_gt_valid, sgbm_valid)
        strict_noc_gt_valid_pixel = torch.sum(strict_noc_gt_valid)
        strict_noc_gt_diff = torch.mul(noc_gt_diff, strict_noc_gt_valid)
        relaxed_noc_gt_diff = torch.mul(noc_gt_diff, noc_gt_valid)
        error['noc_gt_epe'] = torch.sum(strict_noc_gt_diff) / strict_noc_gt_valid_pixel
        strict_noc_gt_bad3 = strict_noc_gt_diff > 3.0
        error['strict_noc_gt_bad3'] = torch.sum(strict_noc_gt_bad3) / strict_noc_gt_valid_pixel * 100.0
        relaxed_noc_gt_bad3 = relaxed_noc_gt_diff > 3.0
        error['relaxed_noc_gt_bad3'] = torch.sum(relaxed_noc_gt_bad3) / noc_gt_valid_pixel * 100.0
    return error


def eval_sgbm(opts):
    """
    Main function to evaluate SGBM results

    :param opts: options
    :return: None
    """
    dataset_list = {'kitti2015': datasets.Kitti2015Dataset,
                    'kitti2012': datasets.Kitti2012Dataset,
                    'SceneFlow': datasets.SceneFlowDataset}
    dataset = dataset_list[opts.dataset]
    data_path = os.path.join(opts.data_path, opts.dataset)
    dataset = dataset(data_path, opts.max_disp, 1, opts.resized_height, opts.resized_width, 0.8, False, False)
    loader = DataLoader(dataset, 1, False, num_workers=0, pin_memory=True, drop_last=False)

    num_samples = len(dataset)
    num_valid_samples = num_samples

    print("Begin evaluating SGBM")
    print("Dataset: %s" % opts.dataset)
    print("Number of leftmost columns to be ignored: %d" % opts.max_disp)

    coverage = 0.0
    gt_epe = 0.0
    strict_gt_bad3 = 0.0
    relaxed_gt_bad3 = 0.0
    noc_gt_epe = 0.0
    strict_noc_gt_bad3 = 0.0
    relaxed_noc_gt_bad3 = 0.0

    for batch_id, inputs in enumerate(loader):
        sgbm_disp = opts.max_disp * inputs['raw_disp']
        gt_disp = inputs['gt_disp']
        if "noc_gt_disp" in inputs:
            noc_gt_disp = inputs['noc_gt_disp']
        else:
            noc_gt_disp = None

        # undo padding
        if "top_pad" in inputs:
            sgbm_disp = sgbm_disp[:, :, inputs['top_pad']:, inputs['left_pad']:]
            gt_disp = gt_disp[:, :, inputs['top_pad']:, inputs['left_pad']:]
            if noc_gt_disp is not None:
                noc_gt_disp = noc_gt_disp[:, :, inputs['top_pad']:, inputs['left_pad']:]

        # disable error computation for leftmost columns since the algorithms does not attempt to solve them
        sgbm_disp = sgbm_disp[:, :, :, opts.max_disp:]
        gt_disp = gt_disp[:, :, :, opts.max_disp:]
        if noc_gt_disp is not None:
            noc_gt_disp = noc_gt_disp[:, :, :, opts.max_disp:]

        batch_err = cal_error(sgbm_disp, gt_disp, noc_gt_disp)

        if not batch_err['validity']:
            num_valid_samples -= 1
        else:
            coverage += batch_err['coverage']
            gt_epe += batch_err['gt_epe']
            strict_gt_bad3 += batch_err['strict_gt_bad3']
            relaxed_gt_bad3 += batch_err['relaxed_gt_bad3']
            if "noc_gt_disp" in inputs:
                noc_gt_epe += batch_err['noc_gt_epe']
                strict_noc_gt_bad3 += batch_err['strict_noc_gt_bad3']
                relaxed_noc_gt_bad3 += batch_err['relaxed_noc_gt_bad3']

    coverage /= num_valid_samples
    gt_epe /= num_valid_samples
    strict_gt_bad3 /= num_valid_samples
    relaxed_gt_bad3 /= num_valid_samples
    noc_gt_epe /= num_valid_samples
    strict_noc_gt_bad3 /= num_valid_samples
    relaxed_noc_gt_bad3 /= num_valid_samples

    print("Average converage: %.4f" % coverage)
    print("For all pixels: ")
    print("EPE = %.4f | Strict Bad3 = %.4f | Relaxed Bad3 = %.4f" % (gt_epe, strict_gt_bad3, relaxed_gt_bad3))
    print("For noc pixels (KITTI only): ")
    print(
        "EPE = %.4f | Strict Bad3 = %.4f | Relaxed Bad3 = %.4f" % (noc_gt_epe, strict_noc_gt_bad3, relaxed_noc_gt_bad3))


if __name__ == "__main__":
    eval_opts = parse_arguments()
    eval_sgbm(eval_opts)
