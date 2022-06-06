import random
import os
import cv2
import re
from PIL import Image
import numpy as np
import torch
from munch import Munch
from inspect import isfunction
import contextlib

operators = '|'.join(['arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot', 'coth', 'csc', 'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf',
                      'injlim', 'ker', 'lg', 'lim', 'liminf', 'limsup', 'ln', 'log', 'max', 'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh'])
ops = re.compile(r'\\operatorname{(%s)}' % operators)


class EmptyStepper:
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

# helper functions from lucidrains


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def seed_everything(seed: int):
    """Seed all RNGs

    Args:
        seed (int): seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args(args, **kwargs) -> Munch:
    args = Munch({'epoch': 0}, **args)
    kwargs = Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.update(kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = get_device(args, kwargs.no_cuda)
    args.encoder_structure = args.get('encoder_structure', 'hybrid')
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
    if 'decoder_args' not in args or args.decoder_args is None:
        args.decoder_args = {}
    return args


def get_device(args, no_cuda=False):
    device = 'cpu'
    available_gpus = torch.cuda.device_count()
    args.gpu_devices = args.gpu_devices if args.get('gpu_devices', False) else list(range(available_gpus))
    if available_gpus > 0 and not no_cuda:
        device = 'cuda:%d' % args.gpu_devices[0] if args.gpu_devices else 0
        assert available_gpus >= len(args.gpu_devices), "Available %d gpu, but specified gpu %s." % (available_gpus, ','.join(map(str, args.gpu_devices)))
        assert max(args.gpu_devices) < available_gpus, "legal gpu_devices should in [%s], received [%s]" % (','.join(map(str, range(available_gpus))), ','.join(map(str, args.gpu_devices)))
    return device


def gpu_memory_check(model, args):
    # check if largest batch can be handled by system
    try:
        batchsize = args.batchsize if args.get('micro_batchsize', -1) == -1 else args.micro_batchsize
        for _ in range(5):
            im = torch.empty(batchsize, args.channels, args.max_height, args.min_height, device=args.device).float()
            seq = torch.randint(0, args.num_tokens, (batchsize, args.max_seq_len), device=args.device).long()
            loss = model.data_parallel(im, device_ids=args.gpu_devices, tgt_seq=seq)
            loss.sum().backward()
    except RuntimeError:
        raise RuntimeError("The system cannot handle a batch size of %i for the maximum image size (%i, %i). Try to use a smaller micro batchsize." % (batchsize, args.max_height, args.max_width))
    model.zero_grad()
    with torch.cuda.device(args.device):
        torch.cuda.empty_cache()
    del im, seq


def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]


def pad(img: Image, divable: int = 32) -> Image:
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    threshold = 128
    data = np.array(img.convert('LA'))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(np.uint8)
    else:
        data = (255-data[..., -1]).astype(np.uint8)
    data = (data-data.min())/(data.max()-data.min())*255
    if data.mean() > threshold:
        # To invert the text to white
        gray = 255*(data < threshold).astype(np.uint8)
    else:
        gray = 255*(data > threshold).astype(np.uint8)
        data = 255-data

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    im = Image.fromarray(rect).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s


def alternatives(s):
    # TODO takes list of list of tokens
    # try to generate equivalent code eg \ne \neq or \to \rightarrow
    # alts = [s]
    # names = ['\\'+x for x in re.findall(ops, s)]
    # alts.append(re.sub(ops, lambda match: str(names.pop(0)), s))

    # return alts
    return [s]


def get_optimizer(optimizer):
    return getattr(torch.optim, optimizer)


def get_scheduler(scheduler):
    if scheduler is None:
        return EmptyStepper
    return getattr(torch.optim.lr_scheduler, scheduler)


def num_model_params(model):
    return sum([p.numel() for p in model.parameters()])


@contextlib.contextmanager
def in_model_path():
    import pix2tex
    model_path = os.path.join(os.path.dirname(pix2tex.__file__), 'model')
    saved = os.getcwd()
    os.chdir(model_path)
    try:
        yield
    finally:
        os.chdir(saved)
