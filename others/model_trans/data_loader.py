from __future__ import print_function
import cv2
import torch
import urllib.request as urll
from model_trans.image_processor import add_noise, pad_image


def download_data(url, file_name):
    proxy_handler = urll.ProxyHandler({})
    opener = urll.build_opener(proxy_handler)
    urll.install_opener(opener)
    req = urll.Request(url)
    r = opener.open(req)
    result = r.read()

    with open(file_name, 'wb') as f:
        f.write(result)


def load_data(args):
    img = cv2.imread(args.input_dir)
    img = cv2.resize(img, [img.shape[0] // 8, img.shape[1] // 8])
    print(img.shape)
    img = img.transpose((2, 1, 0))
    y = img

    y = torch.FloatTensor(y)

    y_dirt, mask = add_noise(img)

    x = torch.randn(args.latent_dim)
    y_dirt, orig_size = pad_image(y_dirt, args.patch_size)
    y, _ = pad_image(y, args.patch_size)

    return x.to(args.device), y.to(args.device), y_dirt.to(
        args.device), orig_size, mask.to(args.device)
