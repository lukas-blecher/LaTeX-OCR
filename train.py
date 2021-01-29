import os
import sys
import argparse
import logging
import yaml

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from munch import Munch
from tqdm.auto import tqdm
import wandb

from dataset.dataset import Im2LatexDataset
from models import get_model
from utils import *


def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args)
    device = args.device
    os.makedirs(args.model_path, exist_ok=True)

    model = get_model(args)
    encoder, decoder = model.encoder, model.decoder
    opt = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.005, steps_per_epoch=len(dataloader), epochs=args.epochs)

    for e in range(args.epoch, args.epochs):
        args.epoch = e
        dset = tqdm(iter(dataloader))
        for i, (seq, im) in enumerate(dset):
            opt.zero_grad()
            tgt_seq, tgt_mask = seq['input_ids'].to(device), seq['attention_mask'].bool().to(device)
            encoded = encoder(im.to(device))
            loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            scheduler.step()

            dset.set_description('Loss: %.4f' % loss.item())
            if args.wandb:
                wandb.log({'train/loss': loss.item()})
            if (i+1) % args.sample_freq == 0:
                pred = ''.join(dataloader.tokenizer.decode(decoder.generate(torch.LongTensor([args.bos_token]).to(
                    device), args.max_seq_len, eos_token=args.eos_token, context=encoded[:1].detach())[:-1]).split(' ')).replace('Ġ', ' ').strip()
                s = seq['input_ids'][0]
                truth = ''.join(dataloader.tokenizer.decode(s[1:list(s).index(args.eos_token)]).split(' ')).replace('Ġ', ' ').strip()
                if args.wandb:
                    table = wandb.Table(columns=["Truth", "Prediction"])
                    table.add_data(truth, pred)
                    wandb.log({"test/examples": table})
                else:
                    print('\n%s\n%s' % (truth, pred))
        if (e+1) % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.model_path, '%s_e%02d.pth' % (args.name, e+1)))
            yaml.dump(dict(args), open(os.path.join(args.model_path, 'config.yaml'), 'w+'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default='settings/default.yaml', help='path to yaml config file', type=argparse.FileType('r'))
    parser.add_argument('-d', '--data', default='dataset/data/dataset.pkl', type=str, help='Path to Dataset pkl file')
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')

    parsed_args = parser.parse_args()
    with parsed_args.config as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
    train(args)
