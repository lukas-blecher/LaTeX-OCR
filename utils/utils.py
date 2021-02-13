import random
import os
import numpy as np
import torch
from munch import Munch
from inspect import isfunction

# helper functions from lucidrains


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args(args, **kwargs):
    args = Munch({'epoch': 0}, **args)
    kwargs = Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = 'cuda' if torch.cuda.is_available() and not kwargs.no_cuda else 'cpu'
    args.max_dimensions = [args.max_width, args.max_height]
    args.out_path = os.path.join(args.model_path, args.name)
    os.makedirs(args.out_path, exist_ok=True)
    return args


def token2str(tokens, tokenizer):
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]


def get_optimizer(optimizer):
    return getattr(torch.optim, optimizer)


def get_scheduler(scheduler):
    return getattr(torch.optim.lr_scheduler, scheduler)
