import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision.transforms as T
import seaborn as sns

THRESH = 0.5


def show_image(imgs, titles):
    fig = plt.figure(figsize=(12, 5))
    for i in range(len(imgs)):
        ax = fig.add_subplot(121 + i)
        plt.imshow(imgs[i], 'gray')
        ax.set_title(titles[i])
        plt.axis('off')
        plt.show()


def add_mask(mask, value, size, start, end):
    mask[:, start:start + size, end:end + size] = 0
    value[:, start:start + size, end:end + size] = 0


def add_noise(img):
    # img = normalize_img(img)
    # noise = np.random.normal(scale=25/255., size=img.shape)
    # img = np.clip(img + noise, 0, 1).astype(np.float32)
    img_dirt = torch.FloatTensor(img)
    # img_dirt = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))(img)

    mask = torch.ones_like(img_dirt)
    # value = torch.zeros_like(img_dirt)
    # add_mask(mask, value, 50, 240, 220)
    # add_mask(mask, value, 80, 380, 400)
    # add_mask(mask, value, 70, 400, 100)
    # add_mask(mask, value, 70, 80, 320)
    #
    # img_dirt = img_dirt * mask + value

    # mask = cv2.imread("./data/kate_mask.jpg")
    # mask = mask.transpose((2, 1, 0))
    # mask = torch.FloatTensor(mask)
    # img_dirt *= mask

    img_dirt = normalize_img(img_dirt)
    noise = np.random.normal(0, 0.1 ** 0.5, img_dirt.shape)
    # noise = torch.rand(img_dirt.shape)
    img_dirt += noise

    return img_dirt, mask


def normalize_img(img):
    max_v = img.max()
    min_v = img.min()
    if max_v - min_v <= 0:
        return img
    return (img - min_v) / (max_v - min_v)


def pad_image(img, img_size):
    img = normalize_img(img)
    if len(img.shape) == 3 and img.shape[0] > 3:
        img = img.transpose(0, 2)
    if len(img.shape) == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    h_size = img.shape[1]
    w_size = img.shape[2]
    size = max(h_size, w_size)
    pad = img_size - size
    h_pad = pad + size - h_size
    w_pad = pad + size - w_size
    img = T.Pad(padding=(0, 0, w_pad, h_pad))(img)

    return img.unsqueeze(0), [h_size, w_size]


def restore_img(x, orig_size):
    x = torch.squeeze(x, 0)
    x = x[:, :orig_size[0], :orig_size[1]]
    x = x.transpose(0, 2)
    if x.is_cuda:
        return x.detach().cpu().numpy()
    else:
        return x.numpy()


def store_img(img, img_dir):
    img = np.asarray(normalize_img(img) * 255, dtype=np.uint8)
    img = np.asarray([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
    img = img.transpose((1, 2, 0))
    image = Image.fromarray(img, 'RGB')
    image.save(img_dir)


def draw_stats(x, seq, patch_size, orig_size, file_dir, file_dir2):
    x = x[-1].detach().cpu().numpy()
    uniform_data = x[0, 0, :, :]
    a_map = np.zeros(seq.shape)
    for i in range(uniform_data.shape[0]):
        for j in range(uniform_data.shape[1]):
            v = uniform_data[i, j]
            if v > THRESH:
                a_map[0, i, :] = 1
                a_map[0, j, :] = 1

    a_map = restore_img(torch.from_numpy(a_map), patch_size, orig_size)
    a_map[:, :, 1] = 0
    a_map[:, :, 0] = 0
    store_img(a_map, file_dir2)
    sns.set()
    heatmap = sns.heatmap(uniform_data)
    fig = heatmap.get_figure()
    fig.savefig(file_dir)
    plt.close(fig)


def draw_l1(x, y, file_dir):
    x = x[-1].detach().cpu().numpy()
    y = y[-1].detach().cpu().numpy()

    l_one = np.mean(np.abs(x - y), axis=1)
    l_one = l_one.reshape((4, 4))
    sns.set()
    heatmap = sns.heatmap(l_one)
    fig = heatmap.get_figure()
    fig.savefig(file_dir)
    plt.close(fig)


def store_fig(psnr_list, ssim_list, loss_list, fig_dir, num):

    iter_nums = np.linspace(1, num, num)
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
    plt.close(plots)
