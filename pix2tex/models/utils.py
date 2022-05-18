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


def get_model(args, training=False):
    if args.encoder_structure.lower() == 'vit':
        encoder = vit.get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)
    decoder = transformer.get_decoder(args)
    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus > 1:
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
        try:
            batchsize = args.batchsize if args.get('micro_batchsize', -1) == -1 else args.micro_batchsize
            for _ in range(5):
                im = torch.empty(batchsize, args.channels, args.max_height, args.min_height, device=args.device).float()
                seq = torch.randint(0, args.num_tokens, (batchsize, args.max_seq_len), device=args.device).long()
                decoder(seq, context=encoder(im)).sum().backward()
        except RuntimeError:
            raise RuntimeError("The system cannot handle a batch size of %i for the maximum image size (%i, %i). Try to use a smaller micro batchsize."%(batchsize, args.max_height, args.max_width))
        model.zero_grad()
        torch.cuda.empty_cache()
        del im, seq
    return model
