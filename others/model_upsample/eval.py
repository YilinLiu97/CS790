import numpy as np
from skimage.metrics import structural_similarity as ssim
from model_trans.image_processor import restore_img


def compute_psnr(x, y):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return 100
    max_pixel = max(x.max(), y.max())
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(x, y, orig_size):
    a = restore_img(x, orig_size)
    b = restore_img(y, orig_size)
    ssim_v = ssim(b, a, data_range=a.max() - a.min(), multichannel=True)
    return ssim_v


class Metric:
    def __init__(self, patch_size, orig_size):
        self.psnr_list = []
        self.ssim_list = []
        self.loss_list = []
        self.patch_size = patch_size
        self.orig_size = orig_size

    def evaluate(self, x, y, loss):
        self.psnr_list.append(compute_psnr(x, y))
        self.ssim_list.append(compute_ssim(x, y, self.orig_size))
        self.loss_list.append(loss.item())
