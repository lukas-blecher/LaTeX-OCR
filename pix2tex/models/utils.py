import torch
import torch.nn as nn

from x_transformers import TransformerWrapper, Encoder, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

from . import hybrid
from . import vit


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AutoregressiveWrapper, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x: torch.Tensor):
        return self.decoder.generate(torch.LongTensor([self.args.bos_token]*len(x)).to(x.device), self.args.max_seq_len, eos_token=self.args.eos_token, context=self.encoder(x))


def get_model(args, training=False):
    if args.encoder_structure.lower() == 'vit':
        encoder = vit.get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)
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
    )
    available_gpus = torch.cuda.device_count()
    if available_gpus > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    if args.wandb:
        import wandb
        wandb.watch(model)
    if training:
        # check if largest batch can be handled by system
        batchsize = args.batchsize if args.get(
            'micro_batchsize', -1) == -1 else args.micro_batchsize
        im = torch.empty(batchsize, args.channels, args.max_height,
                         args.min_height, device=args.device).float()
        seq = torch.randint(0, args.num_tokens, (batchsize,
                                                 args.max_seq_len), device=args.device).long()
        decoder(seq, context=encoder(im)).sum().backward()
        model.zero_grad()
        torch.cuda.empty_cache()
        del im, seq
    return model
