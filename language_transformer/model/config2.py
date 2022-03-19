import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_epoch', default=500, type=int, help='node rank for distributed training')
    parser.add_argument('--num_epoch', default=10000, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='distributed backend')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--full_mode', default='true', type=str2bool, help='full AE or decoder only')
    parser.add_argument('--input_dir', default='./data/1.jpg', type=str, help='full AE or decoder only')
    parser.add_argument('--img_dir', default='./img/', type=str, help='full AE or decoder only')
    parser.add_argument('--use_noise', default='false', type=str2bool, help='full AE or decoder only')

    opt = parser.parse_args(args=[])

    return opt
