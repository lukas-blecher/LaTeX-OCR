import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import repeat

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size

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


def get_encoder(args):
    backbone = ResNetV2(
        layers=args.backbone_layers, num_classes=0, global_pool='', in_chans=args.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    min_patch_size = 2**(len(args.backbone_layers)+1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps//min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(img_size=(args.max_height, args.max_width),
                                      patch_size=args.patch_size,
                                      in_chans=args.channels,
                                      num_classes=0,
                                      embed_dim=args.dim,
                                      depth=args.encoder_depth,
                                      num_heads=args.heads,
                                      embed_layer=embed_layer
                                      )
    return encoder