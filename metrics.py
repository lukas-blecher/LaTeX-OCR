from Levenshtein import distance
import pix2tex.utils as p2t_utils
import pix2tex.models as p2t_models
import yaml
from munch import Munch
import torch
from pix2tex.dataset.dataset import Im2LatexDataset
import numpy as np
from collections import defaultdict
from pix2tex.eval import detokenize
from torchtext.data import metrics

from pix2tex.utils.utils import alternatives, post_process, token2str
from tqdm import tqdm


# CONFIG_PATH = "pix2tex/model/settings/config.yaml"
# BATCHSIZE = 1
# TEMPERATURE = .9
# CHECKPOINT_PATH = "hw_checkpoints/handwritten_training/handwritten_training_e19_step63.pth"
# DEVICE = "cpu"
# DATA_PATH = "pix2tex/dataset/handwritten/test.pkl"

def get_model_and_data(config_path, batch_size, temperature, checkpoint_path, device, data_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = p2t_utils.parse_args(Munch(config))

    args.testbatchsize = batch_size
    args.wandb = False
    args.temperature = temperature

    model = p2t_models.get_model(args)
    model.load_state_dict(torch.load(checkpoint_path, device))

    dataset = Im2LatexDataset(pad=True).load(data_path)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    dataset.update(**valargs)

    return model, dataset


def evaluate(model, dataset: Im2LatexDataset, args: Munch, num_batches: int = None, name: str = 'test'):
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
    bleus, edit_dists, token_acc = [], [], []
    bleu_score, edit_distance, token_accuracy = 0, 1, 0
    pbar = tqdm(enumerate(iter(dataset)), total=len(dataset))
    preds = defaultdict(list)
    for i, (seq, im) in pbar:
        if seq is None or im is None:
            continue
        #loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
        dec = model.generate(im.to(device), temperature=args.get('temperature', .2))
        pred = detokenize(dec, dataset.tokenizer)
        preds[i].append(pred)
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
    if len(edit_dists) > 0:
        edit_distance = np.mean(edit_dists)
    if len(token_acc) > 0:
        token_accuracy = np.mean(token_acc)

    print('\n%s\n%s' % (truth, pred))
    print('BLEU: %.2f' % bleu_score)
    return bleu_score, edit_distance, token_accuracy, bleus, edit_dists, token_acc, preds

def parse_prediction(pred):
    pred = np.array(pred).squeeze()
    return ''.join(pred)