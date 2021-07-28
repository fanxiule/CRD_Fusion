import re
import torch
import numpy as np


def sec_to_hms(t):
    """
    Convert time seconds to three ints representing hours, minutes and seconds

    :param t: time in seconds only
    :return: three ints for hours, minutes, and seconds, respectively
    """
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s)


def sec_to_hms_str(t):
    """
    Convert time in seconds to a string consisting of hours, minutes and seconds

    :param t: time in seconds only
    :return: a string in the format of #h#m#s, e.g. 5h10m24s
    """
    h, m, s = sec_to_hms(t)
    time_str = "%dh%dm%ds" % (h, m, s)
    return time_str


def compute_raw_disp_err(raw_disp, gt_disp, max_disp_ds):
    """
    Compute bad3 for raw disparity

    :param raw_disp: raw disparity map normalized between 0 and 1
    :param gt_disp: groudn truth disparity map
    :param max_disp_ds: maximum disparity range after the image is downscaled
    :return: bad3 for raw disparity
    """
    valid_mask = gt_disp > 0
    valid_pixels = torch.sum(valid_mask)
    rescale_raw_disp = raw_disp * max_disp_ds
    rescale_raw_disp[rescale_raw_disp == 0] = -5.0  # make sure invalid pixels are counted as outliers
    diff = torch.abs(rescale_raw_disp - gt_disp)
    diff = torch.mul(valid_mask, diff)
    bad3 = diff > 3.0
    bad3 = torch.sum(bad3) / valid_pixels * 100.0
    return bad3


def compute_disp_error(pred_disp, gt_disp):
    """
    Calculate disparity error metrics for the whole batch

    :param pred_disp: predicted disparity
    :param gt_disp: ground truth disparity
    :return: total EPE, total bad3 and error maps for the whole batch
    """
    valid_mask = gt_disp > 0
    valid_pixels = torch.sum(valid_mask)
    diff = torch.abs(pred_disp - gt_disp)
    diff = torch.mul(valid_mask, diff)
    total_epe = torch.sum(diff) / valid_pixels
    total_bad3 = diff > 3.0
    total_bad3 = torch.sum(total_bad3) / valid_pixels * 100.0
    return total_epe, total_bad3, diff


def readPFM(file):
    """
    A method to read PFM file into a numpy array. Taken directly from the SceneFlow website
    (https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/code/python_pfm.py)

    :param file: directory to the PFM file
    :return: a decoded numpy array based on the PFM file
    """
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def post_process(pred_disp, occlusion, thres):
    """
    Post process the predicted disparity maps

    :param pred_disp: predicted disparity map
    :param occlusion: soft occlusion mask
    :param thres: threshold to filter the occlusion mask
    :return: disparity map after post processing
    """
    batch, _, height, width = pred_disp.size()
    window = 10
    validity = torch.clone(occlusion).detach()
    validity[validity < thres] = 0
    validity[validity > 0] = 1
    final_disp = torch.clone(pred_disp).detach()
    for i in range(1, width, 1):
        if i < window:
            local_window = i
        else:
            local_window = window
        prev_col = final_disp[:, :, :, i - local_window:i]
        avg = torch.mean(prev_col, dim=3)
        final_disp[:, :, :, i] = validity[:, :, :, i] * final_disp[:, :, :, i] + (1 - validity[:, :, :, i]) * avg
    return final_disp


def unpad_imgs(inputs, outputs):
    """
    Undo padding on images

    :param inputs: inputs to the model
    :param outputs: prediction from the model
    :return:
    """
    for k, v in inputs.items():
        if k != "frame_id" and k != "left_pad" and k != "top_pad":
            inputs[k] = v[:, :, inputs['top_pad']:, inputs['left_pad']:]
    outputs['refined_disp0'] = outputs['refined_disp0'][:, :, inputs['top_pad']:, inputs['left_pad']:]
    if "occ0" in outputs:
        outputs['occ0'] = outputs['occ0'][:, :, inputs['top_pad']:, inputs['left_pad']:]
