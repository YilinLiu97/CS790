import itertools
import torch
import torch.nn as nn
from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder


class GeneratorDecoder(nn.Module):
    def __init__(self, voc_size, num_tokens, num_patch, args):
        super().__init__()
        self.device = args.device
        self.num_patch = num_patch
        self.out = None
        self.decoder = TransformerWrapper(num_tokens=voc_size, max_seq_len=num_tokens,
                                          attn_layers=Decoder(dim=512, depth=2, heads=2))
        self.optimizer = torch.optim.Adam(itertools.chain(self.decoder.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", threshold=1e-4)

    def forward(self, x, seq):
        mask = torch.ones_like(seq).bool().to(self.device)
        out = self.decoder(seq, mask=mask)
        self.out = out[:, :, :self.num_patch]
        return self.out

    def compute_loss(self, out, y):
        loss = torch.nn.L1Loss(reduction='mean')(out, y)
        return loss

    def compute_l1_r(self):
        r = 0
        for p in self.decoder.parameters():
            r += torch.sum(torch.abs(p))
        return r

    def compute_l2_r(self):
        r = 0
        for p in self.decoder.parameters():
            r += torch.sum(p ** 2)
        return r

    def optimize_parameters(self, x, seq, y, mask):
        # forward
        self.optimizer.zero_grad()
        out = self.forward(x, seq)
        self.loss = self.compute_loss(out * mask, y * mask) + 1e-4 * self.compute_l2_r()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step(self.loss)


class GeneratorFull(nn.Module):
    def __init__(self, image_size, voc_size, num_tokens, num_patch, args):
        super().__init__()
        self.device = args.device
        self.num_patch = num_patch
        self.out = None
        self.encoder = ViTransformerWrapper(image_size=image_size, patch_size=args.patch_size,
                                            attn_layers=Encoder(dim=512, depth=6, heads=8))

        self.decoder = TransformerWrapper(num_tokens=voc_size, max_seq_len=num_tokens,
                                          attn_layers=Decoder(dim=512, depth=6, heads=8, cross_attend=True))
        self.optimizer = torch.optim.Adam(itertools.chain(self.decoder.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", threshold=1e-5)

    def forward(self, x, seq):
        encoded = self.encoder(x, return_embeddings=True)
        out = self.decoder(seq, context=encoded)
        self.out = out[:, :, :self.num_patch]
        return self.out

    def compute_loss(self, out, y):
        loss = torch.nn.MSELoss(reduction='mean')(out, y)
        return loss

    def optimize_parameters(self, x, seq, y, mask):
        # forward
        self.optimizer.zero_grad()
        out = self.forward(x, seq)
        self.loss = self.compute_loss(out * mask, y * mask)
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step(self.loss)
