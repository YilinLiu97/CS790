import model.config2 as config
from model.network import GeneratorDecoder, GeneratorFull
from model.image_processor import store_img, restore_img, store_fig
from model.data_loader import load_data
from model.eval import Metric
import torch


def main():
    args = config.parse_args()

    x, seq, y, y_dirt, origin_shape, num_tokens, voc_size, num_patch, mask = load_data(args)
    print(seq.shape, y_dirt.shape)

    store_img(restore_img(y, args.patch_size, origin_shape), args.img_dir + "{}.jpg".format("origin"))
    store_img(restore_img(y_dirt, args.patch_size, origin_shape), args.img_dir + "{}.jpg".format("dirt"))
    if args.full_mode:
        model = GeneratorFull(y.shape[2], voc_size, num_tokens, num_patch, args)
    else:
        model = GeneratorDecoder(voc_size, num_tokens, num_patch, args)
    model.to(args.device)
    metric = Metric(args.patch_size, origin_shape)

    for i in range(args.num_epoch):
        if i % args.display_epoch == 0 or i == args.num_epoch - 1:
            model.eval()
            out = model.forward(x, seq)
            out = restore_img(out, args.patch_size, origin_shape)
            store_img(out, args.img_dir + "{}_epoch.jpg".format(i))
            model.train()

        model.train()
        model.optimize_parameters(x, seq, y_dirt, mask)
        metric.evaluate(model.out, y, model.loss)

        if (i + 1) % 500 == 0:
            print("epoch: " + str(i + 1) + " " + str(model.loss.item()))
            store_fig(metric.psnr_list, metric.ssim_list, metric.loss_list, args.img_dir + "fig.jpg", i + 1)


if __name__ == '__main__':
    main()
