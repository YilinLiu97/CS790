import model_upsample.config as config
from model_upsample.swin_transformer import SwinUnet
from model_upsample.image_processor import store_img, restore_img, store_fig
from model_upsample.data_loader import get_data_list
from model_upsample.eval import Metric
import torchvision.transforms as T
import torch
import random

default_index = 0
delta = 0.1


def get_data(x_list, y_list, y_dirt_list, size_list, mask_list, index=-1):
    if index < 0:
        index = random.randint(0, x_list.shape[0] - 1)
    x = torch.unsqueeze(x_list[index], 0)
    y = torch.unsqueeze(y_list[index], 0)
    y_dirt = torch.unsqueeze(y_dirt_list[index], 0)
    orig_size = size_list[index]
    mask = torch.unsqueeze(mask_list[index], 0)

    return x, y, y_dirt, orig_size, mask, index


def main():
    args = config.parse_args()

    x_list, y_list, y_dirt_list, size_list, mask_list = get_data_list(args)
    _, y, y_dirt, orig_size, mask, _ = get_data(x_list, y_list, y_dirt_list, size_list, mask_list, default_index)
    
    store_img(restore_img(y, orig_size), args.img_dir + "{}.jpg".format("origin"))
    store_img(restore_img(y_dirt * mask, orig_size), args.img_dir + "{}.jpg".format("dirt"))
    print("save image")
    model = SwinUnet(args)
    model.to(args.device)
    metric = Metric(args.patch_size, orig_size)
    print("start train")

    for i in range(args.num_epoch):

        if i % args.display_epoch == 0 or i == args.num_epoch - 1:
            x, y, y_dirt, orig_size, _, idx = get_data(x_list, y_list, y_dirt_list, size_list, mask_list, default_index)
            model.eval()
            out = model.forward(x)
            out = restore_img(out, orig_size)
            store_img(out, args.img_dir + "{}_epoch.jpg".format(i))
            model.train()

        x, y, y_dirt, orig_size, _, idx = get_data(x_list, y_list, y_dirt_list, size_list, mask_list)
        model.train()

        out = model.forward(x)
        model.update(x, y_dirt, mask, i)
        metric.evaluate(out, y, model.loss)

        if (i + 1) % 500 == 0:
            print("epoch: " + str(i + 1) + " " + str(model.loss.item()))
            store_fig(metric.psnr_list, metric.ssim_list, metric.loss_list, args.img_dir + "fig.jpg", i + 1)


if __name__ == '__main__':
    main()
