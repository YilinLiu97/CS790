from __future__ import print_function
import os
import cv2
import torch
import numpy as np
import urllib.request as urll
from model_upsample.image_processor import add_noise, pad_image


def get_data_list(args):
    f_list = os.listdir(args.input_dir)
    x_list, y_list, y_dirt_list, size_list, mask_list = [], [], [], [], []
    for f in f_list:
        if "kate" in f:
            continue
        x, y, y_dirt, orig_size, mask = load_data(args.input_dir + f, args)
        x_list.append(x)
        y_list.append(y)
        y_dirt_list.append(y_dirt)
        size_list.append(orig_size)
        mask_list.append(mask)

    x = torch.FloatTensor(np.concatenate(x_list, axis=0)).to(args.device)
    y = torch.FloatTensor(np.concatenate(y_list, axis=0)).to(args.device)
    y_dirt = torch.FloatTensor(np.concatenate(y_dirt_list, axis=0)).to(args.device)
    orig_size = np.stack(size_list, axis=0)
    mask = torch.FloatTensor(np.concatenate(mask_list, axis=0)).to(args.device)
    return x, y, y_dirt, orig_size, mask


def load_data(file_name, args):
    img = cv2.imread(file_name)
    img = cv2.resize(img, [args.img_size, args.img_size])
    img = img.transpose((2, 1, 0))
    y = img

    y = torch.FloatTensor(y)

    y_dirt, mask = add_noise(img, args)

    y_dirt, orig_size = pad_image(y_dirt, args.patch_size)
    y, _ = pad_image(y, args.patch_size)

    x = torch.randn(y_dirt.shape)

    return x, y, y_dirt, orig_size, mask
