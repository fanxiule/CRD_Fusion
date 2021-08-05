import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
from utils import sec_to_hms_str, compute_disp_error, post_process, unpad_imgs
from crd_fusion_net import CRDFusionNet
from eval_options import EvalOptions

options = EvalOptions()
eval_opts = options.parse()


def save_disp(pred_disp, frame_id, log_path):
    """
    Save predicted disparity in .npy format

    :param pred_disp: predicted disparity
    :param frame_id: id or filename for the disparity map
    :param log_path: path to save the prediction
    :return: None
    """
    pred_path = os.path.join(log_path, "pred")
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    for i in range(len(frame_id)):
        if "/" in frame_id[i]:
            scene, f_id = frame_id[i].rsplit('/', 1)
            save_path = os.path.join(pred_path, scene)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            f_id = frame_id[i]
            save_path = pred_path
        disp = torch.squeeze(pred_disp[i]).detach().cpu().numpy()
        disp_path = os.path.join(save_path, f_id) + ".npy"
        np.save(disp_path, disp)


def log_time(epe, bad3, duration, batch_sz, start_time, current_step, total_steps):
    """
    Print interim results including error metrics, time elapsed, and estimated time left for the current training

    :param epe: endpoint error of the current batch
    :param bad3: percentage of pixels with err > 3px of the current batch
    :param duration: time spent to process the current batch
    :param batch_sz: current batch size
    :param start_time: starting time of the whole evaluation process
    :param current_step: current step number
    :param total_steps: total steps needed to complete evaluation
    :return: None
    """
    total_time_elapsed = time.time() - start_time
    sample_proc_rate = batch_sz / duration
    time_left = (total_steps / current_step - 1.0) * total_time_elapsed
    print("Avg EPE: %.2f | Avg Bad3: %.2f | sample/s: %.2f | time elapsed: %s | est time left: %s" % (
        epe, bad3, sample_proc_rate, sec_to_hms_str(total_time_elapsed), sec_to_hms_str(time_left)))


def log_event(writer, inputs, outputs, final_err, refined_err, detect_occ, max_disp, scale_list, step):
    """
    Log interim results as an instance in a tensorboard event

    :param writer: tensorboard writer
    :param inputs: inputs to the model
    :param outputs: outputs of the model
    :param final_err: error metrics based on final_disp for the current batch
    :param refined_err: error metrics based on refined_disp0 for the current batch
    :param detect_occ: if set to True, the model has been trained to predict occlusion mask
    :param max_disp: maximum number of disparities after image downscaling is applied
    :param scale_list: list of exponents for all feature scales used in the network, e.g. [0, 3] or [0, 1, 2, 3]
    :param step: current step number
    :return: None
    """
    writer.add_scalar("Final EPE", final_err['epe'], step)
    writer.add_scalar("Final Bad3", final_err['bad3'], step)
    if final_err['err_map'] is not None:
        writer.add_image("Final Error Map", final_err['err_map'][0] / max_disp, step)

    writer.add_scalar("Refined EPE", refined_err['epe'], step)
    writer.add_scalar("Refined Bad3", refined_err['bad3'], step)
    if refined_err['err_map'] is not None:
        writer.add_image("Refined Error Map", refined_err['err_map'][0] / max_disp, step)

    for k, v in inputs.items():
        if k == "gt_disp" or k == "noc_gt_disp":
            writer.add_image("input_%s" % k, v[0] / max_disp, step)
        elif k != "frame_id" and k != "top_pad" and k != "left_pad":
            writer.add_image("input_%s" % k, v[0], step)

    for s in scale_list:
        max_disp_at_scale = max_disp / (2 ** s)
        writer.add_image("refined_disp%d" % s, outputs['refined_disp%d' % s][0] / max_disp_at_scale, step)
        if detect_occ:
            writer.add_image("occ%d" % s, outputs['occ%d' % s][0], step)
        if s == scale_list[-1]:
            writer.add_image("prelim_disp", outputs['prelim_disp'][0] / max_disp_at_scale, step)
        if s == 0:
            writer.add_image("final_disp", outputs['final_disp'][0] / max_disp_at_scale, step)


def handle_nan_err(err_metric):
    """
    Handle NaN in error metric. Occurs when no valid pixels found in gt disp

    :param err_metric: error metric
    :return: None
    """
    err_metric['epe'] = 0
    err_metric['bad3'] = 0


def evaluate(opts):
    """
    Evaluate the model

    :param opts: evaluation options
    :return: None
    """
    log_path = os.path.join(opts.log_dir, opts.model_name)

    # checking
    assert opts.resized_height % (2 ** opts.feature_downscale) == 0, \
        "resized_height not divisible by the given lowest feature scale"
    assert opts.resized_width % (2 ** opts.feature_downscale) == 0, \
        "resized_width not divisible by the given lowest feature scale"
    assert opts.max_disp % (opts.downscale * (2 ** opts.feature_downscale)) == 0, \
        "maximum disparity range not divisible by downscaling factor and lowest feature scale"
    assert not (opts.baseline and opts.occ_detection), \
        "Baseline and occlusion detection cannot be used at the same time"

    feature_scale_list = [0]
    if opts.multi_step_upsample:
        for s in range(1, opts.feature_downscale + 1):
            feature_scale_list.append(s)  # scale list for gradual upsampling in refinement
    else:
        feature_scale_list.append(opts.feature_downscale)  # scale list for direct upsampling in refinement

    model = CRDFusionNet(feature_scale_list, opts.max_disp / opts.downscale, opts.resized_height, opts.resized_width,
                         opts.baseline, opts.gen_fusion, opts.reg_fusion)
    if opts.checkpt is not None and os.path.isdir(opts.checkpt):
        model.load_model(opts.checkpt)
    else:
        print("Cannot find checkpoint path. Use randomly initialized weights")
        model.init_model()
    model.to(opts.device)

    dataset_list = {'kitti2015': datasets.Kitti2015Dataset,
                    'kitti2012': datasets.Kitti2012Dataset,
                    'SceneFlow': datasets.SceneFlowDataset}
    dataset = dataset_list[opts.dataset]
    data_path = os.path.join(opts.data_path, opts.dataset)
    eval_dataset = dataset(data_path, opts.max_disp, opts.downscale, opts.resized_height, opts.resized_width,
                           opts.conf_threshold, False, opts.imagenet_norm)
    eval_loader = DataLoader(eval_dataset, 1, False, num_workers=opts.num_workers, pin_memory=True,
                             drop_last=False)
    num_eval_samples = len(eval_dataset)
    num_valid_samples = num_eval_samples
    num_total_steps = num_eval_samples

    print("Begin evalutating %s" % opts.model_name)
    print("Use checkpt in: %s" % opts.checkpt)
    print("Log event and/or predicted disparity maps in %s" % log_path)
    print("Log frequency: %d" % opts.log_frequency)
    print("Save disp: %r" % opts.save_disp)
    print("-------------Input Data Info-------------")
    print("Dataset: %s" % opts.dataset)
    print("Input size: %d x %d" % (opts.resized_height, opts.resized_width))
    print("Downscaling: %d" % opts.downscale)
    print("Max disp: %d" % opts.max_disp)
    print("Total number of evaluation samples %d" % num_eval_samples)
    print("Total number of iterations: %d" % num_total_steps)
    print("-------------Ablation Info-------------")
    print("Conf threshold: %.2f" % opts.conf_threshold)
    print("ImageNet norm: %r" % opts.imagenet_norm)
    print("Scale list: %s" % ', '.join(str(s) for s in feature_scale_list))
    print("Raw disp fusion in generator: %r" % opts.gen_fusion)
    print("Raw disp fusion in regressor: %r" % opts.reg_fusion)
    print("Using baseline model: %r" % opts.baseline)
    print("Occlusion detection: %r" % opts.occ_detection)
    print("Occlusion threshold used in post processing: %.2f" % opts.occ_threshold)
    print("Post processing: %r" % opts.post_processing)

    writer = SummaryWriter(os.path.join(log_path, 'eval'))

    current_step = 0
    final_err = {'epe': 0, 'bad3': 0, 'err_map': None}
    refined_err = {'epe': 0, 'bad3': 0, 'err_map': None}
    # for KITTI
    final_noc_err = {'epe': 0, 'bad3': 0}
    refined_noc_err = {'epe': 0, 'bad3': 0}

    print("-------------Start Evaluation-------------")
    start_time = time.time()
    total_time = 0

    model.eval()
    with torch.no_grad():
        for batch_id, inputs in enumerate(eval_loader):
            current_step += 1
            for k, v in inputs.items():
                if k != "frame_id" and k != "left_pad" and k != "top_pad":
                    inputs[k] = v.to(opts.device)

            batch_start_time = time.time()
            outputs = model(inputs['l_rgb'], inputs['r_rgb'], inputs['raw_disp'], inputs['mask'])
            if "top_pad" in inputs:
                unpad_imgs(inputs, outputs)
            if opts.occ_detection and opts.post_processing:
                outputs['final_disp'] = post_process(outputs['refined_disp0'], outputs['occ0'], opts.occ_threshold)
            else:
                outputs['final_disp'] = outputs['refined_disp0']
            duration = time.time() - batch_start_time
            total_time += duration

            batch_num = inputs['l_rgb'].size()[0]
            avg_final = {}
            avg_refined = {}

            avg_final['epe'], avg_final['bad3'], avg_final['err_map'] = compute_disp_error(outputs['final_disp'],
                                                                                           inputs['gt_disp'])
            avg_refined['epe'], avg_refined['bad3'], avg_refined['err_map'] = compute_disp_error(
                outputs['refined_disp0'], inputs['gt_disp'])

            if torch.isnan(avg_final['epe']) or torch.isnan(avg_final['bad3']) or torch.isnan(
                    avg_refined['epe']) or torch.isnan(avg_refined['bad3']):
                # Mostly for SceneFlow where several Test images cause NaN error
                handle_nan_err(avg_final)
                handle_nan_err(avg_refined)
                num_valid_samples -= batch_num

            final_err['epe'] += batch_num * avg_final['epe']
            final_err['bad3'] += batch_num * avg_final['bad3']
            refined_err['epe'] += batch_num * avg_refined['epe']
            refined_err['bad3'] += batch_num * avg_refined['bad3']

            if 'noc_gt_disp' in inputs:  # for KITTI
                noc_final_avg_epe, noc_final_avg_bad3, _ = compute_disp_error(outputs['final_disp'],
                                                                              inputs['noc_gt_disp'])
                noc_refined_avg_epe, noc_refined_avg_bad3, _ = compute_disp_error(outputs['refined_disp0'],
                                                                                  inputs['noc_gt_disp'])
                final_noc_err['epe'] += batch_num * noc_final_avg_epe
                final_noc_err['bad3'] += batch_num * noc_final_avg_bad3
                refined_noc_err['epe'] += batch_num * noc_refined_avg_epe
                refined_noc_err['bad3'] += batch_num * noc_refined_avg_bad3

            if opts.save_disp:
                save_disp(outputs['final_disp'], inputs['frame_id'], log_path)

            if current_step % opts.log_frequency == 0:
                log_time(avg_final['epe'], avg_final['bad3'], duration, batch_num, start_time, current_step,
                         num_total_steps)
                log_event(writer, inputs, outputs, avg_final, avg_refined, opts.occ_detection,
                          opts.max_disp / opts.downscale, feature_scale_list, current_step)

    final_epe = final_err['epe'] / num_valid_samples
    final_bad3 = final_err['bad3'] / num_valid_samples
    refined_epe = refined_err['epe'] / num_valid_samples
    refined_bad3 = refined_err['bad3'] / num_valid_samples
    final_noc_epe = final_noc_err['epe'] / num_valid_samples
    final_noc_bad3 = final_noc_err['bad3'] / num_valid_samples
    refined_noc_epe = refined_noc_err['epe'] / num_valid_samples
    refined_noc_bad3 = refined_noc_err['bad3'] / num_valid_samples
    frame_rate = num_valid_samples / total_time

    print("Refined disparity | average EPE: %.4f | average Bad3: %.4f" % (refined_epe, refined_bad3))
    print("Final disparity | average EPE: %.4f | average Bad3: %.4f" % (final_epe, final_bad3))
    print("Number of valid samples: %d" % num_valid_samples)
    print("Overall framerate (for reference only): %.4f" % frame_rate)
    print("-------------For KITTI only-------------")
    print("Refined disparity (noc) | average EPE: %.4f | average Bad3: %.4f" % (refined_noc_epe, refined_noc_bad3))
    print("Final disparity (noc) | average EPE: %.4f | average Bad3: %.4f" % (final_noc_epe, final_noc_bad3))


if __name__ == "__main__":
    evaluate(eval_opts)
