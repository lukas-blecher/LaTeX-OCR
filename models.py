import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import *
from x_transformers.autoregressive_wrapper import *
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import rearrange, repeat



class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out


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


class Model(nn.Module):
    def __init__(self, encoder: CustomVisionTransformer, decoder: CustomARWrapper, args, temp: float = .333):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_token = args.bos_token
        self.eos_token = args.eos_token
        self.max_seq_len = args.max_seq_len
        self.temperature = temp

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        device = x.device
        encoded = self.encoder(x.to(device))
        dec = self.decoder.generate(torch.LongTensor([self.bos_token]*len(x))[:, None].to(device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec


def get_model(args):
    backbone = ResNetV2(
        layers=args.backbone_layers, num_classes=0, global_pool='', in_chans=args.channels,
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

    decoder = CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token
    ).to(args.device)
    if 'wandb' in args and args.wandb:
        import wandb
        wandb.watch((encoder, decoder.net.attn_layers))
    return Model(encoder, decoder, args)
