import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import *
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from einops import rearrange, repeat


class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_width,
        max_height,
        patch_size,
        attn_layers,
        channels=1,
        num_classes=None,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert max_width % patch_size == 0 and max_height % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (max_width // patch_size)*(max_height // patch_size)
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.max_width = max_width
        self.max_height = max_height

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        #self.mlp_head = FeedForward(dim, dim_out = num_classes, dropout = dropout) if exists(num_classes) else None

    def forward(self, img, **kwargs):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = torch.tensor(img.shape[2:])//p
        pos_emb_ind = repeat(torch.arange(h)*(self.max_width//p-w), 'h -> (h w)', w=w)+torch.arange(h*w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()
        x += self.pos_embedding[:, pos_emb_ind]
        x = self.dropout(x)

        x = self.attn_layers(x, **kwargs)
        x = self.norm(x)

        return x


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AutoregressiveWrapper, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x: torch.Tensor):
        return self.decoder.generate(torch.LongTensor([self.args.bos_token]*len(x)).to(x.device), self.args.max_seq_len, eos_token=self.args.eos_token, context=self.encoder(x))


def get_model(args):
    encoder = ViTransformerWrapper(
        max_width=args.max_width,
        max_height=args.max_height,
        channels=args.channels,
        patch_size=args.patch_size,
        attn_layers=Encoder(
            dim=args.dim,
            depth=args.num_layers,
            heads=args.heads,
        )
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
