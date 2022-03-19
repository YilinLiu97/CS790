import torch
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision.transforms as T
from ConvDecoder.demo_helper.helpers import apply_mask, MaskFunc, ifft2, channels2imgs, root_sum_of_squares, crop_center


def show_image(imgs, titles):
    fig = plt.figure(figsize=(12, 5))
    for i in range(len(imgs)):
        ax = fig.add_subplot(121 + i)
        plt.imshow(imgs[i], 'gray')
        ax.set_title(titles[i])
        plt.axis('off')
        plt.show()


def add_noise(img):
    # convert complex numpy to torch tensor
    # data = img.copy()
    # data = np.stack((data.real, data.imag), axis=-1)
    # slice_ksp = torch.FloatTensor(data)
    #
    # # Create masks for retrospectively under-sampling. (By default: 4X)
    # masked_kspace, mask = apply_mask(slice_ksp,
    #                                  mask_func=MaskFunc(center_fractions=[0.07], accelerations=[4]))
    #
    # multi_img = ifft2(masked_kspace)
    # img_dirt = []
    # for img in multi_img.detach().cpu():
    #     img_dirt += [img[:, :, 0].numpy(), img[:, :, 1].numpy()]
    # img_dirt = channels2imgs(np.array(img_dirt))
    # img_dirt = root_sum_of_squares(torch.from_numpy(img_dirt))
    # special crop
    # if img_dirt.shape[0] > 320:
    #     img_dirt = crop_center(img_dirt, 320, 320)
    img = torch.FloatTensor(img)
    img_dirt = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))(img)
    img_dirt += torch.randn(img_dirt.shape)

    return img_dirt


def normalize_img(img):
    max_v = img.max()
    min_v = img.min()
    return (img - min_v) / (max_v - min_v)


def pad_image(img, patch_size):
    img = normalize_img(img)
    if len(img.shape) == 3 and img.shape[0] > 3:
        img = img.transpose(0, 2)
    if len(img.shape) == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    h_size = img.shape[1]
    w_size = img.shape[2]
    size = max(h_size, w_size)
    # h_pad = (h_size % patch_size) // 2
    # w_pad = (w_size % patch_size) // 2
    # T.Pad(padding=(h_pad, w_pad, h_size - h_pad if h_pad > 0 else 0, w_size - w_pad if w_pad > 0 else 0))(img)
    pad = (size % patch_size)
    h_pad = (patch_size - pad if pad > 0 else 0) + size - h_size
    w_pad = (patch_size - pad if pad > 0 else 0) + size - w_size
    img = T.Pad(padding=(0, 0, w_pad, h_pad))(img)

    return img.unsqueeze(0), [h_size, w_size]


def tokenize_img(img, patch_size):
    x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    return x


def restore_img(seq, patch_size, orig_size):
    h_size, w_size = orig_size[0], orig_size[1]
    size = max(h_size, w_size)
    pad = (size % patch_size)
    h_pad = (patch_size - pad if pad > 0 else 0) + size - h_size
    h = w = (h_size + h_pad) // patch_size
    x = rearrange(seq, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, c=3, p1=patch_size, p2=patch_size)

    x = torch.squeeze(x, 0)
    x = x[:, :orig_size[0], :orig_size[1]]
    x = x.transpose(0, 2)
    return x.detach().cpu().numpy()


def store_img(img, img_dir):
    img = np.asarray(normalize_img(img) * 255, dtype=np.uint8)
    img = np.asarray([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
    img = img.transpose((1, 2, 0))
    image = Image.fromarray(img, 'RGB')
    image.save(img_dir)


def store_fig(psnr_list, ssim_list, loss_list, fig_dir, args):

    iter_nums = np.linspace(1, args.num_epoch, args.num_epoch)
    plots = plt.figure(figsize=(15, 4))
    ax = plots.add_subplot(141)
    ax.plot(iter_nums, psnr_list)
    ax.set_ylabel('PSNR')
    ax.set_xlabel('Epochs')

    ax = plots.add_subplot(142)
    ax.plot(iter_nums, ssim_list)
    ax.set_ylabel('SSIM')
    ax.set_xlabel('Epochs')

    ax = plots.add_subplot(143)
    ax.plot(iter_nums, loss_list)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')

    plt.savefig(fig_dir)
