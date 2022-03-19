import torch
import torch.nn as nn
from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder
from ConvDecoder.DIP_UNET_models.skip import *
from ConvDecoder.DIP_UNET_models.unet_and_tv.unet_model import UnetModel

## Transformer
class TransEncoder(nn.Module):
   def __init__(self, voc_size, num_tokens, num_patch):
        super(TransDecoder).__init__()
        self.num_patch = num_patch
        self.out = None
        self.encoder = ViTransformerWrapper(image_size=image_size, patch_size=patch_size,
                                            attn_layers=Encoder(dim=512, depth=6, heads=8))

    def forward(self, seq):
        mask = torch.ones_like(seq).bool().to(self.device)
        out = self.encoder(seq)
        self.out = out[:, :, :self.num_patch]
        return self.out


class TransDecoder(nn.Module):
    def __init__(self, voc_size, num_tokens, num_patch):
        super(TransDecoder).__init__()
        self.num_patch = num_patch
        self.out = None
        self.decoder = TransformerWrapper(num_tokens=voc_size, max_seq_len=num_tokens,
                                          attn_layers=Decoder(dim=512, depth=6, heads=8))

    def forward(self, seq):
        mask = torch.ones_like(seq).bool().to(self.device)
        out = self.decoder(seq)
        self.out = out[:, :, :self.num_patch]
        return self.out

x = torch.rand(1,64,238)
model = TransEncoder()


## The architecture used in the original DIP paper
class orig_DIP(nn.Module): 
    def __init__(self, in_size, num_channels, output_depth, short=False):
        super(orig_DIP, self).__init__()
        if short is False:
            self.net = skip(in_size,num_channels, output_depth, 
               num_channels_down = [num_channels] * 8,
               num_channels_up =   [num_channels] * 8,
               num_channels_skip =    [num_channels*0] * 6 + [4,4],  
               filter_size_up = 3, filter_size_down = 5, 
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=False, need_bias=True, pad="zero", act_fun='ReLU').type(dtype)
        else:
            self.net = skip(in_size,num_channels, output_depth, 
               num_channels_down = [num_channels] * 2,
               num_channels_up =   [num_channels] * 2,
               num_channels_skip =    [num_channels*0] * 0 + [4,4],  
               filter_size_up = 3, filter_size_down = 5, 
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=False, need_bias=True, pad="zero", act_fun='ReLU').type(dtype)
        
    def forward(self, x):
        return self.net(x)

## A lightweight decoder-like CNN
class conv_model(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size, need_dropout=False, need_sigmoid=False):
        super(conv_model, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1]*(num_layers-1)
        
        ### compute up-sampling factor from one layer to another
        scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
        hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
        
        ### hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers-1):
            
            self.net.add(nn.Upsample(size=hidden_size[i], mode='nearest'))
            if use_partial:
                conv = PartialConv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True, multi_channel=True, return_mask=True)
            else:
                conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True)
            self.net.add(conv)
            self.net.add(nn.ReLU())
            self.net.add(nn.BatchNorm2d( num_channels, affine=True))
            if need_dropout:
              self.net.add(nn.Dropout2d(0.3))
        ### final layer
        self.net.add(nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True) )
        #self.net.add(nn.ReLU())
        self.net.add(nn.BatchNorm2d( num_channels, affine=True))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        if need_sigmoid:
          self.net.add(nn.Sigmoid())
        
        
    def forward(self, x, scale_out=1):
        return self.net(x)*scale_out

