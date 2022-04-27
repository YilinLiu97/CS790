from __future__ import print_function
import cv2
import torch
import urllib.request as urll
from model.image_processor import add_noise, pad_image, tokenize_img, show_image


def download_data(url, file_name):
    proxy_handler = urll.ProxyHandler({})
    opener = urll.build_opener(proxy_handler)
    urll.install_opener(opener)
    req = urll.Request(url)
    r = opener.open(req)
    result = r.read()

    with open(file_name, 'wb') as f:
        f.write(result)


def load_data(args, filename='file1339.h5'):
    # f = h5py.File(filename, 'r')
    # slicenu = f["kspace"].shape[0] // 2 + 2
    # img = f['kspace'][slicenu]
    # y = f["reconstruction_rss"][slicenu]
    img = cv2.imread(args.input_dir)
    img = img.transpose((2, 1, 0))
    y = img

    y_dirt, mask = add_noise(img)
    orig_shape = y_dirt.shape
    if args.use_noise:
        x = torch.randn(orig_shape)
    else:
        x = torch.FloatTensor(y_dirt)
    y = torch.FloatTensor(y)

    x, orig_size = pad_image(x, args.patch_size)
    y_dirt, _ = pad_image(y_dirt, args.patch_size)
    y, _ = pad_image(y, args.patch_size)
    num_patch = int((x.shape[2] // args.patch_size) * (x.shape[3] // args.patch_size))
    patch_len = 3 * args.patch_size ** 2
    voc_size = max(num_patch + 1, patch_len)
    batch_size = 1
    weights = torch.ones(voc_size).expand(batch_size, -1)
    seq = torch.multinomial(weights, num_samples=num_patch, replacement=False)
    # seq = torch.zeros((1, num_patch)).int()

    y = tokenize_img(y, args.patch_size)
    y_dirt = tokenize_img(y_dirt, args.patch_size)
    mask, _ = pad_image(mask, args.patch_size)
    mask = tokenize_img(mask, args.patch_size)
    y_dirt = y_dirt.float()

    return x.to(args.device), seq.to(args.device), y.to(args.device), y_dirt.to(
        args.device), orig_size, num_patch, voc_size, patch_len, mask.to(args.device)
