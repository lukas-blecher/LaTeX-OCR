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


def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(args)
    device = args.device
    args.pad_token_id = dataloader.pad_token_id
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
            if i % 15 == 0:
                print(''.join(dataloader.tokenizer.decode(decoder.generate(torch.LongTensor([dataloader.bos_token_id]).to(
                    device), args.max_seq_len, eos_token=dataloader.eos_token_id, context=encoded[:1])[:-1]).split(' ')).replace('Ġ', ' ').strip())
                print(dataloader.pairs[dataloader.i][0][0])


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
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    args.device = torch.device('cuda' if torch.cuda.is_available() and not parsed_args.no_cuda else 'cpu')
    train(args)
