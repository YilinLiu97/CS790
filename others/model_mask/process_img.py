import numpy as np
from PIL import Image


def store_img(imgs, img_dir, epoch):
    for i in range(0, len(imgs), 2):
        img = np.asarray(imgs[i] * 255, dtype=np.uint8)
        # img = np.asarray([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
        image = Image.fromarray(img, 'RGB')
        image.save(img_dir + "/{}_epoch.jpg".format(i))
