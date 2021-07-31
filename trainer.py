import os
import time
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
from utils import sec_to_hms_str, compute_disp_error, post_process, unpad_imgs
from crd_fusion_net import CRDFusionNet
from loss import SelfSupLoss, SupLoss


class CRDFusionTrainer:
    def __init__(self, options):
        """
        Initialize an object to train the network

        :param options: Training options
        """
        torch.manual_seed(75)
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.start_time = 0

        # checking
        assert self.opt.resized_height % (2 ** self.opt.feature_downscale) == 0, \
            "resized_height not divisible by the given lowest feature scale"
        assert self.opt.resized_width % (2 ** self.opt.feature_downscale) == 0, \
            "resized_width not divisible by the given lowest feature scale"
        assert self.opt.max_disp % (self.opt.downscale * (2 ** self.opt.feature_downscale)) == 0, \
            "maximum disparity range not divisible by downscaling factor and lowest feature scale"
        assert not (self.opt.baseline and self.opt.occ_detection), \
            "Baseline and occlusion detection cannot be used at the same time"

        # models and parameters
        self.current_epoch = 0
        self.current_step = 0
        self.feature_scale_list = [0]
        if self.opt.multi_step_upsample:
            for s in range(1, self.opt.feature_downscale + 1):
                self.feature_scale_list.append(s)  # scale list for gradual upsampling in refinement
        else:
            self.feature_scale_list.append(self.opt.feature_downscale)  # scale list for direct upsampling in refinement

        self.model = CRDFusionNet(self.feature_scale_list, self.opt.max_disp / self.opt.downscale,
                                  self.opt.resized_height, self.opt.resized_width, self.opt.baseline,
                                  self.opt.gen_fusion, self.opt.reg_fusion)
        self.model.to(self.opt.device)
        parameters_to_train = self.model.get_params()
        total_params = sum(p.numel() for p in parameters_to_train if p.requires_grad)

        # loss function
        if self.opt.supervised:
            self.loss = SupLoss(self.feature_scale_list)
        else:
            self.loss = SelfSupLoss(self.opt.supervision_weight, self.opt.photo_weight, self.opt.smooth_weight,
                                    self.opt.left_weight, self.opt.occ_weight, self.opt.max_disp / self.opt.downscale,
                                    self.feature_scale_list, self.opt.resized_height, self.opt.resized_width,
                                    self.opt.occ_detection, self.opt.occ_epoch, self.opt.loss_fusion)
        self.loss.to(self.opt.device)

        # optimization
        self.optimizer = optim.Adam(parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.opt.scheduler_step,
                                                            self.opt.lr_change_rate)

        # load pretrained weights and optimizer state if specified
        if self.opt.pretrained_model_path is not None:
            self.load_model()
        else:
            self.model.init_model()

        # dataset
        dataset_list = {'kitti2015': datasets.Kitti2015Dataset,
                        'kitti2012': datasets.Kitti2012Dataset,
                        'SceneFlow': datasets.SceneFlowDataset}
        self.dataset = dataset_list[self.opt.dataset]
        data_path = os.path.join(self.opt.data_path, self.opt.dataset)

        train_dataset = self.dataset(data_path, self.opt.max_disp, self.opt.downscale, self.opt.resized_height,
                                     self.opt.resized_width, self.opt.conf_threshold, True, self.opt.imagenet_norm,
                                     False)
        val_dataset = self.dataset(data_path, self.opt.max_disp, self.opt.downscale, self.opt.resized_height,
                                   self.opt.resized_width, self.opt.conf_threshold, False, self.opt.imagenet_norm)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers,
                                       pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, False, num_workers=self.opt.num_workers,
                                     pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)

        # higher level information about training
        num_train_samples = len(train_dataset)
        num_val_samples = len(val_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        print("Begin training %s" % self.opt.model_name)
        print("-------------Model Info-------------")
        print("Pretrained model: %s" % self.opt.pretrained_model_path)
        print("Total number of model parameters: %d" % total_params)
        print("-------------Logging Info-------------")
        print("Checkpoints and log are saved in %s" % self.log_path)
        print("Checkpoint save frequency: %d" % self.opt.save_frequency)
        print("Logging frequency: every %d steps in the first %d steps, every %d steps after" % (
            self.opt.early_log_frequency, self.opt.early_late_split, self.opt.late_log_frequency))
        print("-------------Input Data Info-------------")
        print("Dataset: %s" % self.opt.dataset)
        print("Input size: %d x %d" % (self.opt.resized_height, self.opt.resized_width))
        print("Downscaling: %d" % self.opt.downscale)
        print("Max disp: %d" % self.opt.max_disp)
        print("-------------Optimization Info-------------")
        print("Total number of training samples: %d" % num_train_samples)
        print("Total number of evaluation samples: %d" % num_val_samples)
        print("Total number of iterations: %d" % self.num_total_steps)
        print("Total number of epochs: %d" % self.opt.num_epochs)
        print("Batch size: %d" % self.opt.batch_size)
        print("Initial learning rate: %.5f" % self.opt.learning_rate)
        print("Scheduler step: %d" % self.opt.scheduler_step)
        print("Scheduler change rate: %.2f" % self.opt.lr_change_rate)
        print("-------------Ablation Study Info-------------")
        print("Conf threshold: %.2f" % self.opt.conf_threshold)
        print("ImageNet norm: %r" % self.opt.imagenet_norm)
        print("Scale list: %s" % ', '.join(str(s) for s in self.feature_scale_list))
        print("Raw disp fusion in generator: %r" % self.opt.gen_fusion)
        print("Raw disp fusion in regressor: %r" % self.opt.reg_fusion)
        print("Raw disp fusion in loss: %r" % self.opt.loss_fusion)
        print("Using baseline model: %r" % self.opt.baseline)
        print("Occlusion detection: %r" % self.opt.occ_detection)
        print("Occlusion threshold used in post processing: %.2f" % self.opt.occ_threshold)
        print("Post processing: %r" % self.opt.post_processing)
        print("Supervised training: %r" % self.opt.supervised)
        print("Loss function weighting (if applicable): %.2f, %.2f, %.2f, %.2f, %.2f" % (
            self.opt.supervision_weight, self.opt.photo_weight, self.opt.smooth_weight, self.opt.left_weight,
            self.opt.occ_weight))
        print("Apply occlusion mask in supervision/left loss from epoch: %d" % self.opt.occ_epoch)
        # Note that when occ_detection is False or loss_fusion is False, their corresponding weights would become 0 in
        # the loss function, even if they are shown as non-zero here

        # logging
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        self.save_opts()

    def train(self):
        """
        Train the model

        :return: None
        """
        print("-------------Start Training-------------")
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        for self.current_epoch in range(self.opt.num_epochs):
            self.train_epoch()
            self.model_lr_scheduler.step()
            if (self.current_epoch + 1) % self.opt.save_frequency == 0:
                self.val()
                self.save_model()

    def train_epoch(self):
        """
        Train the model for one epoch

        :return: None
        """
        print("Training epoch %d" % (self.current_epoch + 1))
        self.model.train()
        for batch_id, inputs in enumerate(self.train_loader):
            self.current_step += 1
            batch_start_time = time.time()
            outputs, losses = self.process_batch(inputs)

            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()

            duration = time.time() - batch_start_time

            # logging
            if self.current_step <= self.opt.early_late_split:  # log more frequently at the beginning of training
                log_flag = (self.current_step % self.opt.early_log_frequency == 0)
            else:
                log_flag = (self.current_step % self.opt.late_log_frequency == 0)

            if log_flag:
                self.log_time(batch_id, duration, losses['total_loss'])

                if 'gt_disp' in inputs:
                    refined_errors = self.compute_disp_err(inputs['gt_disp'], outputs['refined_disp0'])
                    final_errors = self.compute_disp_err(inputs['gt_disp'], outputs['final_disp'])
                else:
                    refined_errors = None
                    final_errors = None

                self.log("train", inputs, outputs, losses, refined_errors, final_errors)
                self.val()

            # print("Training loss at epoch %d step %d: %.4f" % (
            #     self.current_epoch + 1, self.current_step, losses["total_loss"].item()))

    def process_batch(self, inputs):
        """
        Train the model with one batch

        :param inputs: input stack from the dataloader
        :return: outputs stack with predicted disparity, and
                 training losses containing total loss, supervision loss, photometric loss, smoothness loss, etc.
        """
        for k, v in inputs.items():
            if k != "frame_id" and k != "left_pad" and k != "top_pad":
                inputs[k] = v.to(self.opt.device)

        outputs = self.model(inputs['l_rgb'], inputs['r_rgb'], inputs['raw_disp'], inputs['mask'])
        if self.opt.supervised:
            losses = self.loss(outputs, inputs['gt_disp'])
        else:
            losses = self.loss(inputs['l_rgb'], inputs['r_rgb'], inputs['raw_disp'], inputs['mask'], outputs,
                               self.current_epoch + 1)
        if "top_pad" in inputs:
            unpad_imgs(inputs, outputs)
        if self.opt.occ_detection and self.opt.post_processing and (not self.opt.supervised):
            outputs['final_disp'] = post_process(outputs['refined_disp0'], outputs['occ0'], self.opt.occ_threshold)
        else:
            outputs['final_disp'] = outputs['refined_disp0']
        return outputs, losses

    def val(self):
        """
        Validate the data with a mini batch to have a glimpse on the model's validation results. Note that this is not
        to replace the complete validation step

        :return: None
        """
        self.model.eval()  # set the model to evaluation mode
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)  # in case the end of the validation dataset is reached
            inputs = self.val_iter.next()
        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if 'gt_disp' in inputs:
                refined_errors = self.compute_disp_err(inputs['gt_disp'], outputs['refined_disp0'])
                final_errors = self.compute_disp_err(inputs['gt_disp'], outputs['final_disp'])
            else:
                refined_errors = None
                final_errors = None
            self.log("val", inputs, outputs, losses, refined_errors, final_errors)
        self.model.train()  # reset the model to training mode

    def save_opts(self):
        """
        Save the current training options to disk

        :return: None
        """
        opt_dir = os.path.join(self.log_path, "model_opt")
        if not os.path.exists(opt_dir):
            os.makedirs(opt_dir)
        opts = self.opt.__dict__.copy()
        with open(os.path.join(opt_dir, "opt.json"), "w") as f:
            json.dump(opts, f, indent=2)

    def log_time(self, batch_id, duration, loss):
        """
        Print interim results including training loss, time elapsed, and estimated time left for the current training

        :param batch_id: id for the batch being logged
        :param duration: time spent to process one batch
        :param loss: training loss
        :return: None
        """
        total_time_elapsed = time.time() - self.start_time
        # not always accurate when the current batch is the last batch in the dataset
        # acceptable for information only
        sample_proc_rate = self.opt.batch_size / duration
        time_left = (self.num_total_steps / self.current_step - 1.0) * total_time_elapsed
        print("Epoch %d | batch_id %d | sample/s: %.2f | training loss: %.4f | time elapsed: %s | est time left: %s" % (
            (self.current_epoch + 1), batch_id, sample_proc_rate, loss.item(), sec_to_hms_str(total_time_elapsed),
            sec_to_hms_str(time_left)))

    def log(self, mode, inputs, outputs, losses, refined_errors, final_errors):
        """
        Log interim results as an instance in a tensorboard event

        :param mode: mode for tensorboard event, train or val
        :param inputs: stacks of input tensors, including left and right RGB, confidence mask, raw disaprity,
               gt disparity if available
        :param outputs: all output estimated disparity maps
        :param losses: all training losses
        :param refined_errors: error metrics for 'refined_disp0'
        :param final_errors: error metrics for 'final_disp'. Same as refined errors if there is no post processing
        :return: None
        """
        writer = self.writers[mode]
        for k, v in losses.items():
            writer.add_scalar(k, v, self.current_step)

        if refined_errors is not None:
            for k, v in refined_errors.items():
                if k != "err_map":
                    writer.add_scalar("refined_%s" % k, v, self.current_step)
                else:
                    writer.add_image("refined_error_map", (v[0] / (self.opt.max_disp / self.opt.downscale)),
                                     self.current_step)

        if final_errors is not None:
            for k, v in final_errors.items():
                if k != "err_map":
                    writer.add_scalar("final_%s" % k, v, self.current_step)
                else:
                    writer.add_image("final_error_map", (v[0] / (self.opt.max_disp / self.opt.downscale)),
                                     self.current_step)

        for k, v in inputs.items():
            if k == "gt_disp" or k == "noc_gt_disp":
                writer.add_image("input_%s" % k, (v[0] / (self.opt.max_disp / self.opt.downscale)), self.current_step)
            elif k != "frame_id":
                writer.add_image("input_%s" % k, v[0], self.current_step)

        for s in self.feature_scale_list:
            max_disp_at_scale = self.opt.max_disp / (self.opt.downscale * (2 ** s))
            writer.add_image("refined_output_%d" % s, outputs['refined_disp%d' % s][0] / max_disp_at_scale,
                             self.current_step)
            if not self.opt.baseline:
                writer.add_image("occ_%d" % s, outputs['occ%d' % s][0], self.current_step)
            if s == self.feature_scale_list[-1]:
                writer.add_image("prelim_output", outputs['prelim_disp'][0] / max_disp_at_scale, self.current_step)
            if s == 0:
                writer.add_image("final_output", outputs['final_disp'][0] / max_disp_at_scale, self.current_step)

    @staticmethod
    def compute_disp_err(gt_disp, disp_pred):
        """
        Calculate error metrics for the predicted disparity

        :param gt_disp: ground truth disparity tensor
        :param disp_pred: predicted disparity tensor
        :return: error metrics consisting of EPE, bad3, and error map
        """
        epe, bad3, diff = compute_disp_error(disp_pred, gt_disp)
        errors = {'epe': epe, 'bad3': bad3, 'err_map': diff}
        return errors

    def load_model(self):
        """
        Load pretrained checkpoints

        :return: None
        """
        assert os.path.isdir(
            self.opt.pretrained_model_path), "Cannot find pretrained model %s" % self.opt.pretrained_model_path
        print("Loading pretrained model from %s" % self.opt.pretrained_model_path)
        self.model.load_model(self.opt.pretrained_model_path)

        # load optimizer state
        optimizer_path = os.path.join(self.opt.pretrained_model_path, "adam.pth")
        if os.path.isfile(optimizer_path):
            print("Loading Adam weights")
            pretrained_dict = torch.load(optimizer_path)
            optimizer_dict = self.optimizer.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in optimizer_dict}
            optimizer_dict.update(pretrained_dict)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("No pretrained Adam weights found. Adam is initialized randomly")

    def save_model(self):
        """
        Save the model as checkpoints files

        :return: None
        """
        save_dir = os.path.join(self.log_path, "checkpts", "weights_%d" % (self.current_epoch + 1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_model(save_dir)
        optim_path = os.path.join(save_dir, "adam.pth")
        torch.save(self.optimizer.state_dict(), optim_path)
