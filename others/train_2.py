import model_trans.config as config
from model_trans.transgan import Generator, get_optimizer, optimize_parameters, weights_init, set_init
from model_trans.image_processor import store_img, restore_img, store_fig
from model_trans.data_loader import load_data
from model.eval import Metric


def main():
    args = config.parse_args()

    x, y, y_dirt, orig_size, mask = load_data(args)
    print(x.shape, y_dirt.shape, mask.shape)

    store_img(restore_img(y, orig_size), args.img_dir + "{}.jpg".format("origin"))
    store_img(restore_img(y_dirt, orig_size), args.img_dir + "{}.jpg".format("dirt"))
    set_init(args.init_type)
    model = Generator(args)
    model.apply(weights_init)
    model.to(args.device)
    optimizer, scheduler = get_optimizer(args, model)
    metric = Metric(args.patch_size, orig_size)

    for i in range(args.num_epoch):
        if i % args.display_epoch == 0 or i == args.num_epoch - 1:
            model.eval()
            out = model.forward(x)
            out = restore_img(out, orig_size)
            store_img(out, args.img_dir + "{}_epoch.jpg".format(i))
            model.train()

        loss = optimize_parameters(optimizer, scheduler, model, x, y_dirt, mask)
        out = model.forward(x)
        metric.evaluate(out, y, loss)

        if (i + 1) % 500 == 0:
            print("epoch: " + str(i + 1) + " " + str(loss.item()))
            store_fig(metric.psnr_list, metric.ssim_list, metric.loss_list, args.img_dir + "fig.jpg", i + 1)


if __name__ == '__main__':
    main()
