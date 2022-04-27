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
    parser.add_argument('--num_epoch', default=5000, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='distributed backend')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--full_mode', default='false', type=str2bool, help='full AE or decoder only')
    parser.add_argument('--input_dir', default='./data/test/', type=str, help='full AE or decoder only')
    parser.add_argument('--img_dir', default='./img/', type=str, help='full AE or decoder only')
    parser.add_argument('--use_noise', default='true', type=str2bool, help='full AE or decoder only')

    parser.add_argument('--img_size', default=256, type=int, help='patch size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--in_chans', default=3, type=int, help='patch size')
    parser.add_argument('--embd_dim', default=96, type=int, help='patch size')
    parser.add_argument('--DEPTHS', default=[1, 1, 1, 1], type=list, help='patch size')
    parser.add_argument('--NUM_HEADS', default=[1, 1, 1, 1], type=list, help='patch size')
    parser.add_argument('--WINDOW_SIZE', default=8, type=int, help='patch size')
    parser.add_argument('--MLP_RATIO', default=2., type=float, help='patch size')
    parser.add_argument('--QKV_BIAS', default='true', type=str2bool, help='patch size')
    parser.add_argument('--QK_SCALE', default=None, help='patch size')
    parser.add_argument('--DROP_RATE', default=0.0, type=float, help='patch size')
    parser.add_argument('--DROP_PATH_RATE', default=0.0, type=float, help='patch size')
    parser.add_argument('--APE', default='false', type=str2bool, help='patch size')
    parser.add_argument('--PATCH_NORM', default='true', type=str2bool, help='patch size')

    opt = parser.parse_args(args=[])

    return opt
