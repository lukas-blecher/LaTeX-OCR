import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import imagesize
import glob
import os
from os.path import join
from collections import defaultdict
import pickle
from PIL import Image
from transformers import PreTrainedTokenizerFast


class Im2LatexDataset:
    def __init__(self, equations=None, images=None, tokenizer=None, shuffle=True, batchsize=16, max_dimensions=(1024, 512)):
        """Generates a torch dataset from pairs of `equations` and `images`.

        Args:
            equations (str, optional): Path to equations. Defaults to None.
            images (str, optional): Directory where images are saved. Defaults to None.
            tokenizer (str, optional): Path to saved tokenizer. Defaults to None.
            shuffle (bool, opitonal): Defaults to True. 
            batchsize (int, optional): Defaults to 16.
            max_dimensions (tuple(int, int), optional): Maximal dimensions the model can handle
        """

        if images is not None and equations is not None:
            assert tokenizer is not None
            self.images = glob.glob(join(images, '*.png'))
            self.sample_size = len(self.images)
            eqs = open(equations, 'r').read().split('\n')
            self.indices = [int(os.path.basename(img).split('.')[0]) for img in self.images]
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)
            self.pad_token = "[PAD]"
            self.bos_token = "[BOS]"
            self.eos_token = "[EOS]"
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.shuffle = shuffle
            self.batchsize = batchsize
            self.data = defaultdict(lambda: [])
            # check the image dimension for every image and group them together
            for i, im in enumerate(self.images):
                width, height = imagesize.get(im)
                if width <= max_dimensions[0] and height <= max_dimensions[1]:
                    self.data[(width, height)].append((eqs[self.indices[i]], im))
            self.data = dict(self.data)
            self.groups = list(self.data.keys())
            self.size = 0
            for k in self.groups:
                div, mod = divmod(len(self.data[k]), self.batchsize)
                self.size += div  # + (1 if mod > 0 else 0)

            self.transform = transforms.Compose([transforms.PILToTensor()])  # , transforms.Normalize([200],[255/2]),transforms.RandomPerspective(fill=0)])
            iter(self)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.i = 0
        self.pairs = []

        for k in self.groups:
            info = np.array(self.data[k], dtype=object)
            p = torch.randperm(len(info)) if self.shuffle else torch.arange(len(info))
            for i in range(0, len(info), self.batchsize):
                batch = info[p[i:i+self.batchsize]]
                if len(batch.shape) == 1:
                    batch = batch[None, :]
                if len(batch) < self.batchsize:
                    continue
                self.pairs.append(batch)
        if self.shuffle:
            self.pairs = np.random.permutation(np.array(self.pairs, dtype=object))
        else:
            self.pairs = np.array(self.pairs, dtype=object)
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        self.i += 1
        return self.prepare_data(self.pairs[self.i])

    def prepare_data(self, batch):
        """loads images into memory

        Args:
            batch (numpy.array[[str, str]]): array of equations and image path pairs

        Returns:
            tuple(torch.tensor, torch.tensor): data in memory
        """

        eqs, ims = batch.T
        images = []
        for path in list(ims):
            images.append(self.transform(Image.open(path)))
        tok = self.tokenizer(list(eqs), return_token_type_ids=False)
        # pad with bos and eos token
        for k, p in zip(tok, [[self.bos_token_id, self.eos_token_id], [1, 1]]):
            tok[k] = pad_sequence([torch.LongTensor([p[0]]+x+[p[1]]) for x in tok[k]], batch_first=True, padding_value=self.pad_token_id)

        return tok, torch.cat(images).float().unsqueeze(1)/255

    def load(self, filename, args=[]):
        """returns a pickled version of a dataset

        Args:
            filename (str): Path to dataset
        """
        with open(filename, 'rb') as file:
            x = pickle.load(file)
        return x

    def save(self, filename):
        """save a pickled version of a dataset

        Args:
            filename (str): Path to dataset
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def update(self, args):
        for k in ['batchsize', 'shuffle']:
            if k in args:
                setattr(self, k, args[k])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model', add_help=False)
    parser.add_argument('-i', '--images', type=str, required=True, help='Image folder')
    parser.add_argument('-e', '--equations', type=str, required=True, help='equations text file')
    parser.add_argument('-t', '--tokenizer', default=None, help='Pretrained tokenizer file')
    parser.add_argument('-o', '--out', default='dataset.pkl', help='output dataset')
    args = parser.parse_args()
    Im2LatexDataset(args.equations, args.images, args.tokenizer).save(args.out)
