import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
import numpy as np
from PIL import Image
import cv2
import imagesize
import yaml
from tqdm.auto import tqdm
from pix2tex.utils import *
from pix2tex.dataset.dataset import *
from munch import Munch
import argparse
from typing import Tuple


def prepare_data(dataloader: Im2LatexDataset) -> Tuple[torch.tensor, torch.tensor]:
    """Use the data from a dataloader to train a image resizer model. 
    Randomly resize the images of one batch in the dataset and return the original resolution along side with the new images.

    Args:
        dataloader (Im2LatexDataset): The dataset in question

    Returns:
        Tuple[torch.tensor, torch.tensor]: One batch of resized images and labels
    """
    _, ims = dataloader.pairs[dataloader.i-1].T
    images = []
    scale = None
    c = 0
    width, height = imagesize.get(ims[0])
    while True:
        c += 1
        s = np.array([width, height])
        scale = 5*(np.random.random()+.02)
        if all((s*scale) <= dataloader.max_dimensions[0]) and all((s*scale) >= 16):
            break
        if c > 25:
            return None, None
    x, y = 0, 0
    for path in list(ims):
        im = Image.open(path)
        modes = [Image.Resampling.BICUBIC,
                 Image.Resampling.BILINEAR]
        if scale < 1:
            modes.append(Image.Resampling.LANCZOS)
        m = modes[int(len(modes)*np.random.random())]
        im = im.resize((int(width*scale), int(height*scale)), m)
        try:
            im = pad(im)
        except:
            return None, None
        if im is None:
            print(path, 'not found!')
            continue
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(dataloader.transform(image=im)['image'][:1])
        if images[-1].shape[-1] > x:
            x = images[-1].shape[-1]
        if images[-1].shape[-2] > y:
            y = images[-1].shape[-2]
    if x > dataloader.max_dimensions[0] or y > dataloader.max_dimensions[1]:
        return None, None
    for i in range(len(images)):
        h, w = images[i].shape[1:]
        images[i] = F.pad(images[i], (0, x-w, 0, y-h), value=0)
    try:
        images = torch.cat(images).float().unsqueeze(1)
    except RuntimeError as e:
        #print(e, 'Images not working: %s' % (' '.join(list(ims))))
        return None, None
    dataloader.i += 1
    labels = torch.tensor(width//32-1).repeat(len(ims)).long()
    return images, labels


def val(val: Im2LatexDataset, model: ResNetV2, num_samples=400, device='cuda') -> float:
    """Evaluate the model on a dataset

    Args:
        val (Im2LatexDataset): Validation dataset
        model (ResNetV2): Model to evaluate
        num_samples (int, optional): Number of samples to evaluate on. Defaults to 400.
        device (str, optional): Torch device. Defaults to 'cuda'.

    Returns:
        float: Accuracy
    """
    model.eval()
    c, t = 0, 0
    iter(val)
    with torch.no_grad():
        for i in range(num_samples):
            im, l = prepare_data(val)
            if im is None:
                continue
            p = model(im.to(device)).argmax(-1).detach().cpu().numpy()
            c += (p == l[0].item()).sum()
            t += len(im)
    model.train()
    return c/t


def main(args):
    """Train a image resizer model.

    Args:
        args (Munch): Object with properties `data`, `batchsize`, `max_dimensions`, 
        `valdata`, `channels`, `device`, `resume`, `lr`, `num_epochs`, `valbatches`, `sample_freq`, `out`
    """
    # data
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(batchsize=args.batchsize, test=False, max_dimensions=args.max_dimensions, keep_smaller_batches=True, device=args.device)
    valloader = Im2LatexDataset().load(args.valdata)
    valloader.update(batchsize=args.batchsize, test=True, max_dimensions=args.max_dimensions, keep_smaller_batches=True, device=args.device)

    # model
    model = ResNetV2(layers=[2, 3, 3], num_classes=int(max(args.max_dimensions)//32), global_pool='avg', in_chans=args.channels, drop_rate=.05,
                     preact=True, stem_type='same', conv_layer=StdConv2dSame).to(args.device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    opt = Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    sched = OneCycleLR(opt, .005, total_steps=args.num_epochs*len(dataloader))
    global bestacc
    bestacc = val(valloader, model, args.valbatches, args.device)

    def train_epoch(sched=None):
        iter(dataloader)
        dset = tqdm(range(len(dataloader)))
        for i in dset:
            im, label = prepare_data(dataloader)
            if im is not None:
                if im.shape[-1] > dataloader.max_dimensions[0] or im.shape[-2] > dataloader.max_dimensions[1]:
                    continue
                opt.zero_grad()
                label = label.to(args.device)

                pred = model(im.to(args.device))
                loss = crit(pred, label)
                if i % 2 == 0:
                    dset.set_description('Loss: %.4f' % loss.item())
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
            if (i+1) % args.sample_freq == 0 or i+1 == len(dset):
                acc = val(valloader, model, args.valbatches, args.device)
                print('Accuracy %.2f' % (100*acc), '%')
                global bestacc
                if acc > bestacc:
                    torch.save(model.state_dict(), args.out)
                    bestacc = acc
    for _ in range(args.num_epochs):
        train_epoch(sched)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train size classification model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--resume', help='path to checkpoint folder', type=str, default='')
    parser.add_argument('--out', type=str, default='checkpoints/image_resizer.pth', help='output destination for trained model')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=10)
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/debug.yaml')
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    args.update(**vars(parsed_args))
    main(args)
