import torch
import torch.nn as nn

from . import hybrid
from . import vit
from . import transformer


class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x: torch.Tensor):
        return self.decoder.generate(torch.LongTensor([self.args.bos_token]*len(x)).to(x.device),
                                     self.args.max_seq_len, eos_token=self.args.eos_token, context=self.encoder(x))


def get_model(args):
    if args.encoder_structure.lower() == 'vit':
        encoder = vit.get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)
    decoder = transformer.get_decoder(args)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    if args.wandb:
        import wandb
        wandb.watch(model)

    return model
