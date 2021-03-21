from dataset.dataset import test_transform
import cv2
import pandas as pd
from PIL import ImageGrab
from PIL import Image
import os
import sys
import argparse
import logging
import yaml

import numpy as np
import torch
from torchvision import transforms
from munch import Munch
from transformers import PreTrainedTokenizerFast


from dataset.latex2png import tex2pil
from models import get_model
from utils import *


def initialize(arguments):
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = Munch(params)
    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    return args, model, tokenizer


def call_model(args, model, tokenizer):
    encoder, decoder = model.encoder, model.decoder
    img = ImageGrab.grabclipboard()
    if img is None:
        print('Copy an image into the clipboard.')
        return
    ratios = [a/b for a, b in zip(img.size, args.max_dimensions)]
    if any([r > 1 for r in ratios]):
        size = np.array(img.size)//max(ratios)
        img = img.resize(size.astype(int), Image.BILINEAR)
    img = np.array(pad(img, args.patch_size).convert('RGB'))
    t = test_transform(image=img)['image'][:1].unsqueeze(0)
    im = t.to(args.device)

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, None].to(device), args.max_seq_len,
                               eos_token=args.eos_token, context=encoded.detach(), temperature=args.temperature)
        pred = post_process(token2str(dec, tokenizer)[0])
    print(pred, '\n')
    df = pd.DataFrame([pred])
    df.to_clipboard(index=False, header=False)
    if args.show:
        try:
            tex2pil([f'$${pred}$$'])[0].show()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use model', add_help=False)
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml')
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth')
    parser.add_argument('-s', '--show', action='store_true', help='Show the rendered predicted latex code')
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    latexocr_path = os.path.dirname(sys.argv[0])
    if latexocr_path != '':
        sys.path.insert(0, latexocr_path)
        os.chdir(latexocr_path)

    args, model, tokenizer = initialize(args)
    while True:
        instructions = input('Press ENTER to predict the LaTeX code for the image in the memory. Type "x" to stop the program. ')
        if instructions.strip().lower() == 'x':
            break
        call_model(args, model, tokenizer)
