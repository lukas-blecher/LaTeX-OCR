import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import *
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import rearrange, repeat


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AutoregressiveWrapper, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x: torch.Tensor):
        return self.decoder.generate(torch.LongTensor([self.args.bos_token]*len(x)).to(x.device), self.args.max_seq_len, eos_token=self.args.eos_token, context=self.encoder(x))


class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = 16

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h//self.patch_size, w//self.patch_size
        pos_emb_ind = repeat(torch.arange(h)*(self.width//self.patch_size-w), 'h -> (h w)', w=w)+torch.arange(h*w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()
        x += self.pos_embed[:, pos_emb_ind]
        #x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class CNNBackbone(nn.Module):
    def __init__(self, feature_dim=512, channels=1, out_dim=128, depth=5, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        dims = [channels]+[feature_dim]*(depth-1)+[out_dim]
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.model(x)


def get_model(args):
    #backbone = CNNBackbone(args.backbone_dim, depth=args.backbone_depth, channels=args.channels)
    backbone = ResNetV2(
        layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=args.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    encoder = CustomVisionTransformer(img_size=(args.max_height, args.max_width),
                                      patch_size=args.patch_size,
                                      in_chans=args.channels,
                                      num_classes=0,
                                      embed_dim=args.dim,
                                      depth=args.encoder_depth,
                                      num_heads=args.heads,
                                      hybrid_backbone=backbone
                                      ).to(args.device)

    decoder = AutoregressiveWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                cross_attend=True
            )),
        pad_value=args.pad_token
    ).to(args.device)
    if args.wandb:
        import wandb
        wandb.watch((encoder.attn_layers, decoder.net.attn_layers))
    return Model(encoder, decoder, args)
