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
    dataloader.update(args)
    device = args.device
    os.makedirs(args.model_path, exist_ok=True)

    model = get_model(args)
    encoder, decoder = model.encoder, model.decoder
    opt = optim.Adam(model.parameters(), args.lr)

    for e in range(args.epochs):
        dset = tqdm(dataloader)
        for i, (seq, im) in enumerate(dset):
            opt.zero_grad()
            tgt_seq, tgt_mask = seq['input_ids'].to(device), seq['attention_mask'].bool().to(device)
            encoded = encoder(im.to(device))
            loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            dset.set_description('Loss: %.4f' % loss.item())
            if args.wandb:
                wandb.log({'train/loss': loss.item()})
            if i % args.sample_freq == 0:
                pred = ''.join(dataloader.tokenizer.decode(decoder.generate(torch.LongTensor([dataloader.bos_token_id]).to(
                    device), args.max_seq_len, eos_token=dataloader.eos_token_id, context=encoded[:1])[:-1]).split(' ')).replace('Ġ', ' ').strip()
                truth = dataloader.pairs[dataloader.i][0][0]
                if args.wandb:
                    table = wandb.Table(columns=["Truth", "Prediction"])
                    table.add_data(tuth, pred)
                    wandb.log({"test/examples": table})
        if (e+1) % args.save_freq == 0:
            torch.save(model.parameters(), os.path.join(args.model_path, '%s_e%02d' % (args.name, e+1)))


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
    args = Munch(params)
    args.wandb = not parsed_args.debug and not args.debug
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    args.device = torch.device('cuda' if torch.cuda.is_available() and not parsed_args.no_cuda else 'cpu')
    seed_everything(args.seed)
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
    train(args)
