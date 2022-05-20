from pix2tex.dataset.transforms import test_transform
import pandas.io.clipboard as clipboard
from PIL import ImageGrab
from PIL import Image
import os
from typing import Tuple
import argparse
import logging
import yaml
import re

import numpy as np
import torch
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from pix2tex.dataset.latex2png import tex2pil
from pix2tex.models import get_model
from pix2tex.utils import *
from pix2tex.model.checkpoints.get_latest_checkpoint import download_checkpoints


def minmax_size(img: Image, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


class LatexOCR:
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None
    last_pic = None

    @in_model_path()
    def __init__(self, arguments=None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        if arguments is None:
            arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        if not os.path.exists(self.args.checkpoint):
            download_checkpoints()
        self.model = get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.model.eval()

        if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
            self.image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), map_location=self.args.device))
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @in_model_path()
    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                print('Provide an image.')
                return ''
            else:
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = minmax_size(pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad(minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS), self.args.max_dimensions, self.args.min_dimensions))
                    t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item()+1)*32
                    logging.info(r, img.size, (w, int(input_image.size[1]*r)))
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(im.to(self.args.device), temperature=self.args.get('temperature', .25))
        pred = post_process(token2str(dec, self.tokenizer)[0])
        try:
            clipboard.copy(pred)
        except:
            pass
        return pred


def output_prediction(pred, args):
    print(pred, '\n')
    if args.show or args.katex:
        try:
            if args.katex:
                raise ValueError
            tex2pil([f'$${pred}$$'])[0].show()
        except Exception as e:
            # render using katex
            import webbrowser
            from urllib.parse import quote
            url = 'https://katex.org/?data=' + \
                quote('{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000",\
"strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}' % pred.replace('\\', '\\\\'))
            webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description='Use model')
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml')
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth')
    parser.add_argument('-s', '--show', action='store_true', help='Show the rendered predicted latex code')
    parser.add_argument('-f', '--file', type=str, default=None, help='Predict LaTeX code from image file instead of clipboard')
    parser.add_argument('-k', '--katex', action='store_true', help='Render the latex code in the browser')
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    parser.add_argument('--no-resize', action='store_true', help='Resize the image beforehand')
    arguments = parser.parse_args()
    with in_model_path():
        model = LatexOCR(arguments)
        file = None
        while True:
            instructions = input('Predict LaTeX code for image ("?"/"h" for help). ')
            possible_file = instructions.strip()
            ins = possible_file.lower()
            if ins == 'x':
                break
            elif ins in ['?', 'h', 'help']:
                print('''pix2tex help:

    Usage:
        On Windows and macOS you can copy the image into memory and just press ENTER to get a prediction.
        Alternatively you can paste the image file path here and submit.

        You might get a different prediction every time you submit the same image. If the result you got was close you
        can just predict the same image by pressing ENTER again. If that still does not work you can change the temperature
        or you have to take another picture with another resolution (e.g. zoom out and take a screenshot with lower resolution). 

        Press "x" to close the program.
        You can interrupt the model if it takes too long by pressing Ctrl+C.

    Visualization:
        You can either render the code into a png using XeLaTeX (see README) to get an image file back.
        This is slow and requires a working installation of XeLaTeX. To activate type 'show' or set the flag --show
        Alternatively you can render the expression in the browser using katex.org. Type 'katex' or set --katex

    Settings:
        to toggle one of these settings: 'show', 'katex', 'no_resize' just type it into the console
        Change the temperature (default=0.333) type: "t=0.XX" to set a new temperature.
                    ''')
                continue
            elif ins in ['show', 'katex', 'no_resize']:
                setattr(arguments, ins, not getattr(arguments, ins, False))
                print('set %s to %s' % (ins, getattr(arguments, ins)))
                continue
            elif os.path.isfile(os.path.realpath(possible_file)):
                file = possible_file
            else:
                t = re.match(r't=([\.\d]+)', ins)
                if t is not None:
                    t = t.groups()[0]
                    model.args.temperature = float(t)+1e-8
                    print('new temperature: T=%.3f' % model.args.temperature)
                    continue
            try:
                img = None
                if file:
                    img = Image.open(file)
                else:
                    try:
                        img = ImageGrab.grabclipboard()
                    except:
                        pass
                pred = model(img)
                output_prediction(pred, arguments)
            except KeyboardInterrupt:
                pass
            file = None


if __name__ == "__main__":
    main()
