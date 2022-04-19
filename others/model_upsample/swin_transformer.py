from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
import random
import torchvision.transforms as T


from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, args, num_classes=1, zero_head=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.args = args

        self.swin_unet = SwinTransformerSys(img_size=args.img_size,
                                            patch_size=args.patch_size,
                                            in_chans=args.in_chans,
                                            num_classes=self.num_classes,
                                            embed_dim=args.embd_dim,
                                            depths=args.DEPTHS,
                                            num_heads=args.NUM_HEADS,
                                            window_size=args.WINDOW_SIZE,
                                            mlp_ratio=args.MLP_RATIO,
                                            qkv_bias=args.QKV_BIAS,
                                            qk_scale=args.QK_SCALE,
                                            drop_rate=args.DROP_RATE,
                                            drop_path_rate=args.DROP_PATH_RATE,
                                            ape=args.APE,
                                            patch_norm=args.PATCH_NORM,
                                            use_checkpoint=False)

        self.optimizer = torch.optim.Adam(self.swin_unet.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", threshold=1e-4)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def compute_loss(self, out, y):
        trans = T.GaussianBlur(random.randint(50, 100) * 2 + 1, sigma=(random.random() * 5 + 0.5))
        x_d = trans(out)
        y_d = trans(y)
        loss = torch.nn.L1Loss(reduction='mean')(x_d, y_d)
        return loss

    def update(self, x, y, mask, e):
        self.optimizer.zero_grad()
        out = self.forward(x)
        self.loss = self.compute_loss(out * mask, y * mask)
        self.loss.backward()
        self.optimizer.step()
        lr_ = self.args.lr * (1.0 - e / self.args.num_epoch) ** 0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_
        # self.scheduler.step(self.loss)

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
