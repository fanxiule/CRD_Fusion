import torch
import torch.nn as nn
import torch.nn.functional as f


class SSIM(nn.Module):
    def __init__(self, c1, c2):
        """
        A simplified implementation of SSIM

        :param c1: constant c1
        :param c2: constant c2
        """
        super(SSIM, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.avg_pool = nn.AvgPool2d(3, 1, 1)  # for computing mean and variance

    def forward(self, x, y):
        """
        Forward pass to calculate simplified SSIM between two input images

        :param x: input image tensor 1
        :param y: input image tensor 2
        :return: SSIM score between two inputs
        """
        avg_x = self.avg_pool(x)
        avg_y = self.avg_pool(y)

        var_x = self.avg_pool(x ** 2) - avg_x ** 2
        var_y = self.avg_pool(y ** 2) - avg_y ** 2
        cov_xy = self.avg_pool(x * y) - avg_x * avg_y

        SSIM_num = (2 * avg_x * avg_y + self.c1) * (2 * cov_xy + self.c2)
        SSIM_den = (avg_x ** 2 + avg_y ** 2 + self.c1) * (var_x + var_y + self.c2)

        SSIM = torch.clamp(SSIM_num / SSIM_den, 0, 1)
        return SSIM


class SelfSupLoss(nn.Module):
    def __init__(self, alpha_disp, alpha_warp, alpha_smooth, alpha_left, alpha_occ, max_disp, scale_list, resized_h,
                 resized_w, detect_occ, occ_epoch, fusion=True):
        """
        A module to compute the self-supervised training loss

        :param alpha_disp: weight for the raw disparity supervision loss
        :param alpha_warp: weight for the photometric reconstruction loss
        :param alpha_smooth: weight for the predicted disparity smoothness loss
        :param alpha_left: weight for the left smoothness loss
        :param alpha_occ: weight for the predicted occlusion mask cross entropy loss
        :param max_disp: maximum number of disparities after downscaling is applied
        :param scale_list: list of exponents for all feature scales used in the network, e.g. [0, 3] or [0, 1, 2, 3]
        :param resized_h: image height after downscaling and resizing
        :param resized_w: image width after downscaling and resizing
        :param detect_occ: if set to True, the occlusion mask is generated and applied to training loss
        :param occ_epoch: a preset epoch number. When current epoch is greater than the preset one, apply left loss and occlusion mask in supervision loss
        :param fusion: if set to True, raw disparity fusion is applied to loss computation
        """
        super(SelfSupLoss, self).__init__()
        self.alpha_disp = alpha_disp
        self.alpha_warp = alpha_warp
        self.alpha_smooth = alpha_smooth
        self.alpha_left = alpha_left
        self.alpha_occ = alpha_occ
        self.bce = nn.BCELoss(reduction='sum')

        self.max_disp = max_disp
        self.scale_list = scale_list
        self.detect_occ = detect_occ
        self.occ_epoch = occ_epoch
        self.current_epoch = 0
        self.fusion = fusion
        if not self.fusion:
            self.alpha_disp = 0  # disable raw disparity supervision with no fusion
        if not self.detect_occ:
            self.alpha_occ = 0  # disable occlusion cross entropy loss when detect occlusion is disabled
            self.alpha_left = 0  # no occlusion mask detection means no left smoothness loss can be computed

        self.upsample = {}
        for s in self.scale_list:
            upsample_scale = 2 ** s
            self.upsample['up%d' % s] = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        self.smoothL1 = nn.SmoothL1Loss(reduction='none', beta=1.0)
        self.SSIM = SSIM(0.01 ** 2, 0.03 ** 2)

        # filters to approximate gradient
        self.img_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(3, 1, 3, 3)
        self.img_sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(3, 1, 3, 3)
        self.disp_grad_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(1, 1, 3, 3)
        self.disp_grad_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(1, 1, 3, 3)
        self.left_smooth = torch.tensor([[-1, 0, 0], [-2, 4, 0], [-1, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)

        # for bilinear sampling
        grid_y = torch.linspace(0, resized_h - 1, resized_h)
        grid_x = torch.linspace(0, resized_w - 1, resized_w)
        self.grid_ver, self.grid_hor = torch.meshgrid(grid_y, grid_x)
        self.grid_ver = torch.unsqueeze(self.grid_ver, dim=0)
        self.grid_hor = torch.unsqueeze(self.grid_hor, dim=0)

        # constants for normalizing pixel indices to prep for bilinear sampling
        # y = kx+b to map image pixel between 0 and (height or width - 1) to between -1 and 1
        self.norm_k_height = 2 / (resized_h - 1 - 0)
        self.norm_b_height = -1
        self.norm_k_width = 2 / (resized_w - 1 - 0)
        self.norm_b_width = -1

    def to(self, *args, **kwargs):
        """
        Override the default to() function so that some tensors can be put to the proper device

        ::param args:
        :param kwargs:
        :return: None
        """
        self.grid_ver = self.grid_ver.to(*args, **kwargs)
        self.grid_hor = self.grid_hor.to(*args, **kwargs)
        self.img_sobel_x = self.img_sobel_x.to(*args, **kwargs)
        self.img_sobel_y = self.img_sobel_y.to(*args, **kwargs)
        self.disp_grad_x = self.disp_grad_x.to(*args, **kwargs)
        self.disp_grad_y = self.disp_grad_y.to(*args, **kwargs)
        self.left_smooth = self.left_smooth.to(*args, **kwargs)

    @staticmethod
    def _cal_img_smoothness(img, kernel):
        """
        Approximate image smoothness with the provided filter

        :param img: image for gradient computation
        :param kernel: kernel for gradient approximation
        :return: approximated image gradient
        """
        group, _, sz, _ = kernel.size()
        pad = sz // 2
        grad = f.conv2d(img, kernel, stride=1, padding=pad, groups=group)
        grad = torch.linalg.norm(grad, ord=1, dim=1, keepdim=True)
        return grad

    def _cal_left_smooth(self, disp, occlusion):
        """
        Enforce disparity smoothness according to disparity on the left for occluded region

        :param disp: disparity map
        :param occlusion: occlusion mask
        :return: left smoothness loss
        """
        left_smooth = self._cal_img_smoothness(disp, self.left_smooth)
        left_smooth = torch.mul(left_smooth, 1 - occlusion)
        left_smooth = torch.sum(left_smooth)
        return left_smooth

    def _reconstruct_image(self, src, disp, batch):
        """
        Reconstruct a synthetic left view given a right view (src) and left disparity

        :param src: source tensor
        :param disp: disparity tensor
        :param batch: batch number
        :return: reconstructed synthetic left RGB
        """
        grid_row = self.grid_ver.repeat(batch, 1, 1)
        grid_col = self.grid_hor.repeat(batch, 1, 1)
        grid_col = grid_col - torch.squeeze(disp, dim=1)
        # normalize grid so that they are between -1 and 1, which is a format accepted by f.grid_sample()
        grid_row = grid_row * self.norm_k_height + self.norm_b_height
        grid_col = grid_col * self.norm_k_width + self.norm_b_width
        grid = torch.stack([grid_col, grid_row], dim=3)
        synth = f.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return synth

    def _cal_reconstruct_loss(self, left_rgb, right_rgb, pred_disp, batch, conf_mask, occlusion):
        """
        Calculate photometric reconstruction loss

        :param left_rgb: left RGB tensor
        :param right_rgb: right RGB tensor
        :param pred_disp: predicted disparity
        :param batch: batch size
        :param conf_mask: confidence mask
        :param occlusion: occlusion mask
        :return: photometric reconstruction loss
        """
        alpha = 0.85
        synthetic_rgb = self._reconstruct_image(right_rgb, pred_disp, batch)
        l1_loss = left_rgb - synthetic_rgb
        l1_loss = torch.linalg.norm(l1_loss, ord=1, dim=1, keepdim=True)

        SSIM_loss = self.SSIM(left_rgb, synthetic_rgb)
        SSIM_loss = 0.5 * (1 - SSIM_loss)
        SSIM_loss = torch.linalg.norm(SSIM_loss, ord=1, dim=1, keepdim=True)

        recon_loss = alpha * SSIM_loss + (1 - alpha) * l1_loss
        recon_loss = torch.mul(recon_loss, 1 - conf_mask)
        recon_loss = torch.mul(recon_loss, occlusion)
        recon_loss = torch.sum(recon_loss)
        return recon_loss

    def _cal_supervision_loss(self, raw_disp, conf_mask, pred_disp, occlusion):
        """
        Calculate disparity supervision loss where supervision is from the raw disparity

        :param raw_disp: disparity generated by traditional stereo matching or sensor
        :param conf_mask: confidence mask tensor for the raw disparity
        :param pred_disp: predicted disparity tensor:
        :param occlusion: occlusion mask
        :return: supervision loss
        """
        supervision_loss = self.smoothL1(raw_disp, pred_disp)
        supervision_loss = torch.mul(supervision_loss, conf_mask)
        if 0 <= self.occ_epoch < self.current_epoch:
            supervision_loss = torch.mul(supervision_loss, occlusion)
        supervision_loss = torch.sum(supervision_loss)
        return supervision_loss

    def _cal_smoothness_loss(self, pred_disp, img_x_grad, img_y_grad, occlusion):
        """
        Calculate edge-aware predicted disparity smoothness loss

        :param pred_disp: predicted disparity tensor
        :param img_x_grad: image gradient in x direction
        :param img_y_grad: image gradient in y direction
        :param occlusion: occlusion mask
        :return: smoothness loss
        """
        disp_smooth_x = self._cal_img_smoothness(pred_disp, self.disp_grad_x)
        disp_smooth_y = self._cal_img_smoothness(pred_disp, self.disp_grad_y)

        disp_smooth_x *= torch.exp(-img_x_grad)
        disp_smooth_y *= torch.exp(-img_y_grad)
        smoothness_loss = disp_smooth_x + disp_smooth_y
        if 0 <= self.occ_epoch < self.current_epoch:
            smoothness_loss = torch.mul(smoothness_loss, 1 - occlusion)
        smoothness_loss = torch.sum(smoothness_loss)
        return smoothness_loss

    def _cal_mask_loss(self, occlusion):
        """
        Calculate binary cross entropy loss between the occlusion mask and an all-1 mask

        :param occlusion: predicted occlusion mask
        :return: binary cross entropy loss
        """
        target = torch.ones_like(occlusion)
        mask_loss = self.bce(occlusion, target)
        return mask_loss

    def forward(self, l_rgb, r_rgb, raw_disp, conf, pred, epoch):
        """
        Forward pass to calculate training loss for a batch

        :param l_rgb: left RGB tensor
        :param r_rgb: right RGB tensor
        :param raw_disp: raw disparity tensor from traditional stereo matching/sensor
        :param conf: confidence mask tensor
        :param pred: model prediction at all image resolutions of interest
        :param epoch: current epoch number
        :return: training losses including total loss, supervision loss, photometric loss and smoothness loss
        """
        losses = {}
        batch_num = l_rgb.size()[0]
        self.current_epoch = epoch

        img_grad_x = self._cal_img_smoothness(l_rgb, self.img_sobel_x)
        img_grad_y = self._cal_img_smoothness(l_rgb, self.img_sobel_y)
        occlusion_mask = torch.ones_like(pred['refined_disp0'])
        if not self.fusion:
            conf_mask = torch.ones_like(conf)
        else:
            conf_mask = conf

        disp_sup_loss = 0
        reconstruct_loss = 0
        smoothness_loss = 0
        left_smooth_loss = 0
        bce_loss = 0
        for s in self.scale_list:
            up_pred_disp = self.upsample['up%d' % s](pred['refined_disp%d' % s])
            up_pred_disp = up_pred_disp * (2 ** s)
            weight = 1 / (2 ** s)
            if self.detect_occ:
                occlusion_mask = self.upsample['up%d' % s](pred['occ%d' % s])
                bce_loss += weight * self._cal_mask_loss(occlusion_mask)
                if 0 <= self.occ_epoch < self.current_epoch:
                    left_smooth_loss += weight * self._cal_left_smooth(up_pred_disp, occlusion_mask)
            disp_sup_loss += weight * self._cal_supervision_loss(self.max_disp * raw_disp, conf_mask, up_pred_disp,
                                                                 occlusion_mask)
            reconstruct_loss += weight * self._cal_reconstruct_loss(l_rgb, r_rgb, up_pred_disp, batch_num, conf_mask,
                                                                    occlusion_mask)
            smoothness_loss += weight * self._cal_smoothness_loss(up_pred_disp, img_grad_x, img_grad_y, occlusion_mask)

        total_px = torch.numel(pred['refined_disp0']) * len(self.scale_list)
        losses['disp_loss'] = self.alpha_disp * disp_sup_loss
        losses['photo_loss'] = self.alpha_warp * reconstruct_loss
        losses['smooth_loss'] = self.alpha_smooth * smoothness_loss
        losses['left_loss'] = self.alpha_left * left_smooth_loss
        losses['occ_loss'] = self.alpha_occ * bce_loss
        losses['total_loss'] = losses['disp_loss'] + losses['photo_loss'] + losses['smooth_loss'] + losses[
            'left_loss'] + losses['occ_loss']
        losses['total_loss'] /= total_px
        return losses


class SupLoss(nn.Module):
    def __init__(self, scale_list):
        """
        A module to compute the supervised training loss

        :param scale_list: list of exponents for all feature scales used in the network, e.g. [0, 3] or [0, 1, 2, 3]
        """
        super(SupLoss, self).__init__()
        self.scale_list = scale_list
        self.smoothL1 = nn.SmoothL1Loss(reduction='none', beta=1.0)
        self.upsample = {}
        for s in self.scale_list:
            upsample_scale = 2 ** s
            self.upsample['up%d' % s] = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)

    def forward(self, pred, gt_disp):
        """
        Forward pass to calculate training loss for a batch

        :param pred: model prediction at all image resolutions of interest
        :param gt_disp: ground truth disparity
        :return: training loss
        """
        loss = {'total_loss': 0}
        validity_mask = gt_disp > 0
        for s in self.scale_list:
            up_pred_disp = self.upsample['up%d' % s](pred['refined_disp%d' % s])
            up_pred_disp = up_pred_disp * (2 ** s)
            sup_loss = self.smoothL1(up_pred_disp, gt_disp)
            sup_loss = torch.mul(validity_mask, sup_loss)
            sup_loss = torch.sum(sup_loss)
            loss['total_loss'] += sup_loss

        total_px = torch.numel(pred['refined_disp0']) * len(self.scale_list)
        loss['total_loss'] = loss['total_loss'] / total_px
        return loss
