import torch
import torch.nn.functional as f


class ConfGeneration:
    def __init__(self, device, full_ZSAD):
        """
        A class to generate confidence map

        :param device: device to compute confidence measures, choose between cuda and cpu
        :param full_ZSAD: if set to True, ZSAD is performed on a 3x3 window. Otherwise, it is performed on a partial 3x3 window
        """
        self.img_height = None
        self.img_width = None
        self.device = device

        # for img gradient
        self.img_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(3, 1, 3, 3)
        self.img_sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(3, 1, 3, 3)
        self.img_sobel_x = self.img_sobel_x.to(self.device)
        self.img_sobel_y = self.img_sobel_y.to(self.device)

        # for image warping
        self.grid_ver = None
        self.grid_hor = None
        self.norm_k_height = None
        self.norm_b_height = None
        self.norm_k_width = None
        self.norm_b_width = None
        if self.img_width is not None and self.img_height is not None:
            self._gen_grid()

        # for confidence computation
        self.ZSAD_conf_scaling = 0.24
        self.MND_conf_scaling = 2
        self.smooth_scaling = 0.01
        self.MND_window = 5

        filer_tl = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_t = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_tr = torch.tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_l = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_m = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_r = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_bl = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_b = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).expand(1, 1, 3, 3)
        filer_br = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float32).expand(1, 1, 3, 3)
        if full_ZSAD:
            self.element_selector = torch.cat(
                [filer_tl, filer_t, filer_tr, filer_l, filer_m, filer_r, filer_bl, filer_b, filer_br], dim=0).to(
                self.device)
            self.group = 9
        else:
            self.element_selector = torch.cat(
                [filer_tl, filer_tr, filer_m, filer_bl, filer_br], dim=0).to(self.device)
            self.group = 5

    def _gen_grid(self):
        """
        Generate grid for image warping

        :return: None
        """
        grid_y = torch.linspace(0, self.img_height - 1, self.img_height)
        grid_x = torch.linspace(0, self.img_width - 1, self.img_width)
        self.grid_ver, self.grid_hor = torch.meshgrid(grid_y, grid_x)
        self.grid_ver = torch.unsqueeze(self.grid_ver, dim=0).to(self.device)
        self.grid_hor = torch.unsqueeze(self.grid_hor, dim=0).to(self.device)
        # for warping
        self.norm_k_height = 2 / (self.img_height - 1 - 0)
        self.norm_b_height = -1
        self.norm_k_width = 2 / (self.img_width - 1 - 0)
        self.norm_b_width = -1

    @staticmethod
    def _gen_mask(disp):
        """
        Generate a validity mask

        :param disp: raw disparity map
        :return: validity mask
        """
        mask = torch.clone(disp)
        mask[mask >= 0.001] = 1
        mask[mask < 0.001] = 0
        return mask

    def _warp_img(self, src, disp):
        """
        Bilinearly warp an image

        :param src: source image
        :param disp: raw disaprity
        :return: synthetic image
        """
        grid_row = self.grid_ver.repeat(1, 1, 1)
        grid_col = self.grid_hor.repeat(1, 1, 1)
        grid_col = grid_col - torch.squeeze(disp, dim=1)
        # normalize grid so that they are between -1 and 1, which is a format accepted by f.grid_sample()
        grid_row = grid_row * self.norm_k_height + self.norm_b_height
        grid_col = grid_col * self.norm_k_width + self.norm_b_width
        grid = torch.stack([grid_col, grid_row], dim=3)
        synth = f.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return synth

    def _cal_ZSAD(self, l_im, synth_l_im):
        """
        Calculate ZSAD

        :param l_im: left image
        :param synth_l_im: synthetic left image
        :return: ZSAD map
        """
        l_img = torch.unsqueeze(l_im, dim=1).repeat(1, self.group, 1, 1)
        synth_l_img = torch.unsqueeze(synth_l_im, dim=1).repeat(1, self.group, 1, 1)
        l_img = f.conv2d(l_img, self.element_selector, stride=1, padding=1, groups=self.group)
        synth_l_img = f.conv2d(synth_l_img, self.element_selector, stride=1, padding=1, groups=self.group)
        l_img_mean = torch.mean(l_img, dim=1, keepdim=True)
        synth_l_img_mean = torch.mean(synth_l_img, dim=1, keepdim=True)
        ZSAD = torch.abs(l_img - l_img_mean - synth_l_img + synth_l_img_mean)
        ZSAD = torch.mean(ZSAD, dim=1, keepdim=True)
        ZSAD = torch.linalg.norm(ZSAD, ord=2, dim=0, keepdim=True)
        ZSAD /= torch.mean(ZSAD)
        return ZSAD

    def _cal_MND(self, disp):
        """
        Calculate MND

        :param disp: raw disparity
        :return: MND map
        """
        mean_disp = f.avg_pool2d(disp, self.MND_window, stride=1, padding=self.MND_window // 2)
        MND = -torch.abs(disp - mean_disp)
        return MND

    def _cal_img_grad(self, img):
        """
        Calculate image gradient by using sobel filters

        :param img: RGB image normalized between 0 and 1
        :return: image gradient
        """
        img_x_grad = f.conv2d(255 * img, self.img_sobel_x, stride=1, padding=1, groups=3)  # undo scaling from ToTensor
        img_y_grad = f.conv2d(255 * img, self.img_sobel_y, stride=1, padding=1, groups=3)
        grad = torch.sqrt(img_x_grad ** 2 + img_y_grad ** 2)
        grad = torch.linalg.norm(grad, ord=2, axis=1, keepdim=True)
        return grad

    def cal_confidence(self, l_im, r_im, disp, mask=None):
        """
        Calculate confidence map for images

        :param l_im: left image
        :param r_im: right image
        :param disp: raw disparity
        :param mask: validity mask. Default to None
        :return: confidence map
        """
        batch, _, img_h, img_w = l_im.size()
        assert batch == 1, "this module can only handle batch size = 1"
        self.img_height = img_h
        self.img_width = img_w
        self._gen_grid()
        if mask is None:
            mask = self._gen_mask(disp)
        synth_l_im = self._warp_img(r_im, disp)
        ZSAD = self._cal_ZSAD(l_im[0], synth_l_im[0])
        ZSAD_conf = torch.exp(-self.ZSAD_conf_scaling * ZSAD)
        MND = self._cal_MND(disp)
        MND_conf = torch.exp(self.MND_conf_scaling * MND)
        l_im_grad = self._cal_img_grad(l_im)
        smooth_weight = torch.exp(-self.smooth_scaling * l_im_grad)
        edge_weight = 1 - smooth_weight
        confidence = smooth_weight * MND_conf + edge_weight * ZSAD_conf
        confidence *= mask
        return confidence
