from pix2tex.dataset.dataset import Im2LatexDataset
import argparse
import logging
import yaml

import numpy as np
import torch
from torchtext.data import metrics
from munch import Munch
from tqdm.auto import tqdm
import wandb
from Levenshtein import distance

from pix2tex.models import get_model, Model
from pix2tex.utils import *


def detokenize(tokens, tokenizer):
    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
    for b in range(len(toks)):
        for i in reversed(range(len(toks[b]))):
            if toks[b][i] is None:
                toks[b][i] = ''
            toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
            if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                del toks[b][i]
    return toks


@torch.no_grad()
def evaluate(model: Model, dataset: Im2LatexDataset, args: Munch, num_batches: int = None, name: str = 'test'):
    """evaluates the model. Returns bleu score on the dataset

    Args:
        model (torch.nn.Module): the model
        dataset (Im2LatexDataset): test dataset
        args (Munch): arguments
        num_batches (int): How many batches to evaluate on. Defaults to None (all batches).
        name (str, optional): name of the test e.g. val or test for wandb. Defaults to 'test'.

    Returns:
        Tuple[float, float, float]: BLEU score of validation set, normed edit distance, token accuracy
    """
    assert len(dataset) > 0
    device = args.device
    log = {}
    bleus, edit_dists, token_acc = [], [], []
    bleu_score, edit_distance, token_accuracy = 0, 1, 0
    pbar = tqdm(enumerate(iter(dataset)), total=len(dataset))
    for i, (seq, im) in pbar:
        if seq is None or im is None:
            continue
        #loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
        dec = model.generate(im.to(device), temperature=args.get('temperature', .2))
        pred = detokenize(dec, dataset.tokenizer)
        truth = detokenize(seq['input_ids'], dataset.tokenizer)
        bleus.append(metrics.bleu_score(pred, [alternatives(x) for x in truth]))
        for predi, truthi in zip(token2str(dec, dataset.tokenizer), token2str(seq['input_ids'], dataset.tokenizer)):
            ts = post_process(truthi)
            if len(ts) > 0:
                edit_dists.append(distance(post_process(predi), ts)/len(ts))
        dec = dec.cpu()
        tgt_seq = seq['input_ids'][:, 1:]
        shape_diff = dec.shape[1]-tgt_seq.shape[1]
        if shape_diff < 0:
            dec = torch.nn.functional.pad(dec, (0, -shape_diff), "constant", args.pad_token)
        elif shape_diff > 0:
            tgt_seq = torch.nn.functional.pad(tgt_seq, (0, shape_diff), "constant", args.pad_token)
        mask = torch.logical_or(tgt_seq != args.pad_token, dec != args.pad_token)
        tok_acc = (dec == tgt_seq)[mask].float().mean().item()
        token_acc.append(tok_acc)
        pbar.set_description('BLEU: %.3f, ED: %.2e, ACC: %.3f' % (np.mean(bleus), np.mean(edit_dists), np.mean(token_acc)))
        if num_batches is not None and i >= num_batches:
            break
    if len(bleus) > 0:
        bleu_score = np.mean(bleus)
        log[name+'/bleu'] = bleu_score
    if len(edit_dists) > 0:
        edit_distance = np.mean(edit_dists)
        log[name+'/edit_distance'] = edit_distance
    if len(token_acc) > 0:
        token_accuracy = np.mean(token_acc)
        log[name+'/token_acc'] = token_accuracy
    if args.wandb:
        # samples
        pred = token2str(dec, dataset.tokenizer)
        truth = token2str(seq['input_ids'], dataset.tokenizer)
        table = wandb.Table(columns=["Truth", "Prediction"])
        for k in range(min([len(pred), args.test_samples])):
            table.add_data(post_process(truth[k]), post_process(pred[k]))
        log[name+'/examples'] = table
        wandb.log(log)
    else:
        print('\n%s\n%s' % (truth, pred))
        print('BLEU: %.2f' % bleu_score)
    return bleu_score, edit_distance, token_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('-c', '--checkpoint', default=None, type=str, help='path to model checkpoint')
    parser.add_argument('-d', '--data', default='dataset/data/val.pkl', type=str, help='Path to Dataset pkl file')
    parser.add_argument('--no-cuda', action='store_true', help='Use CPU')
    parser.add_argument('-b', '--batchsize', type=int, default=10, help='Batch size')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='sampling emperature')
    parser.add_argument('-n', '--num-batches', type=int, default=None, help='how many batches to evaluate on. Defaults to None (all)')

    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/config.yaml')
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.testbatchsize = parsed_args.batchsize
    args.wandb = False
    args.temperature = parsed_args.temperature
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed if 'seed' in args else 42)
    model = get_model(args)
    if parsed_args.checkpoint is None:
        with in_model_path():
            parsed_args.checkpoint = os.path.realpath('checkpoints/weights.pth')
    model.load_state_dict(torch.load(parsed_args.checkpoint, args.device))
    dataset = Im2LatexDataset().load(parsed_args.data)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    dataset.update(**valargs)
    evaluate(model, dataset, args, num_batches=parsed_args.num_batches)
