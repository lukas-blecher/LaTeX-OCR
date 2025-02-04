from pix2tex.dataset.dataset import Im2LatexDataset
import os
import sys
import argparse
import logging
import yaml

import torch
from munch import Munch
from tqdm.auto import tqdm
import wandb
import torch.nn as nn
from pix2tex.eval import evaluate, evaluate_step
from pix2tex.models import get_model
# from pix2tex.utils import *
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, OnExceptionCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import CSVLogger


def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)
    device = args.device
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    max_bleu, max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models(e, step=0):
        torch.save(model.state_dict(), os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step)))
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))

    opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))
            for i, (seq, im) in enumerate(dset):
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0
                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j+microbatch].to(device), seq['attention_mask'][j:j+microbatch].bool().to(device)
                        loss = model.data_parallel(im[j:j+microbatch].to(device), device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask)*microbatch/args.batchsize
                        loss.backward()  # data parallism loss is a vector
                        total_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    opt.step()
                    scheduler.step()
                    dset.set_description('Loss: %.4f' % total_loss)
                    if args.wandb:
                        wandb.log({'train/loss': total_loss})
                if (i+1+len(dataloader)*e) % args.sample_freq == 0:
                    bleu_score, edit_distance, token_accuracy = evaluate(model, valdataloader, args, num_batches=int(args.valbatches*e/args.epochs), name='val')
                    if bleu_score > max_bleu and token_accuracy > max_token_acc:
                        max_bleu, max_token_acc = bleu_score, token_accuracy
                        save_models(e, step=i)
            if (e+1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
            if args.wandb:
                wandb.log({'train/epoch': e+1})
    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt
    save_models(e, step=len(dataloader))


class DataModule(pl.LightningDataModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args

        train_dataloader = Im2LatexDataset().load(args.data)
        train_dataloader.update(**args, test=False)
        val_dataloader = Im2LatexDataset().load(args.valdata)
        val_args = args.copy()
        val_args.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
        val_dataloader.update(**val_args)
        dataset_tokenizer = val_dataloader.tokenizer

        self.dataset_tokenizer = dataset_tokenizer
        self.train_data = train_dataloader
        self.valid_data = val_dataloader

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.valid_data


class OCR_Model(pl.LightningModule):
    def __init__(self, args, dataset_tokenizer, **kwargs):
        super().__init__()
        self.args = args
        self.dataset_tokenizer = dataset_tokenizer

        model = get_model(args)
        if args.load_chkpt is not None:
            model.load_state_dict(torch.load(args.load_chkpt))
        self.model = model
        if torch.cuda.is_available() and not args.no_cuda:
            gpu_memory_check(model, args)

        microbatch = args.get('micro_batchsize', -1)
        if microbatch == -1:
            microbatch = args.batchsize
        self.microbatch = microbatch

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        args = self.args
        opt = get_optimizer(args.optimizer)(self.model.parameters(), args.lr, betas=args.betas)
        scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            }
        }

    def training_step(self, train_batch, batch_idx):
        args = self.args
        (seq, im) = train_batch
        if seq is not None and im is not None:
            total_loss = 0
            for j in range(0, len(im), self.microbatch):
                tgt_seq, tgt_mask = seq['input_ids'][j:j+self.microbatch], seq['attention_mask'][j:j+self.microbatch].bool()
                loss = self.model.data_parallel(im[j:j+self.microbatch], device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask)*self.microbatch/args.batchsize
                total_loss += loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            if args.wandb:
                wandb.log({'train/loss': total_loss})

        self.log('train_loss', total_loss, on_epoch=True, on_step=False, prog_bar=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        bleu_score, edit_distance, token_accuracy = evaluate_step(self.model, self.dataset_tokenizer, val_batch, self.args, name='val')
        metric_dict = {'bleu_score': bleu_score, 'edit_distance': edit_distance, 'token_accuracy': token_accuracy}
        self.log_dict(metric_dict, on_epoch=True, on_step=False, prog_bar=True)
        return metric_dict

    def on_train_epoch_end(self):
        if self.args.wandb:
            wandb.log({'train/epoch': self.current_epoch+1})


class OCR():
    def __init__(self, args):
        self.args = args
        self.logger = CSVLogger(save_dir='pl_logs', name='')
        self.out_path = os.path.join(args.model_path, args.name)
        os.makedirs(self.out_path, exist_ok=True)
        self.data_model_setup()
        self.callbacks_setup()

    def data_model_setup(self):
        self.Data = DataModule(self.args)
        dataset_tokenizer = self.Data.dataset_tokenizer
        self.Model = OCR_Model(self.args, dataset_tokenizer)

    def callbacks_setup(self):
        save_name = f'pl_{args.name}' + '_{epoch}_{step}'

        # NOTE: currently lightning doesn't support multiple monitor metrics
        save_ckpt = ModelCheckpoint(monitor='bleu_score', mode='max', filename=save_name, dirpath=self.out_path,
                                       every_n_epochs=self.args.save_freq, save_top_k=10, save_last=True)

        # BUG: exp_save_name was alaways like pl_pix2tex_0_0.ckpt. possibly a bug in lightning
        exp_save_name = f'pl_pix2tex_{self.Model.current_epoch}_{self.Model.global_step}'
        excpt = OnExceptionCheckpoint(dirpath=self.out_path, filename=exp_save_name)
        bar = RichProgressBar(leave=True, theme=RichProgressBarTheme(
                            description='green_yellow', progress_bar='green1', progress_bar_finished='green1'))
        self.callbacks = [save_ckpt, excpt, bar]

    def fit(self):
        args = self.args
        accelerator = 'gpu' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
        trainer = pl.Trainer(accelerator=accelerator, callbacks=self.callbacks, logger=self.logger,
                            max_epochs=args.epochs, val_check_interval=args.sample_freq)
        trainer.fit(self.Model, self.Data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/debug.yaml')
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
        args = Munch(wandb.config)
    # train(args)

    ocr = OCR(args)
    ocr.fit()
