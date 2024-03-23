
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
from pix2tex.eval import detokenize
import re

def extraer_numero(texto):
    """
    Función para extraer el primer número encontrado en una cadena de texto.

    Parámetros:
    - texto (str): La cadena de texto de la cual extraer el número.

    Retorna:
    - str: El primer número encontrado en la cadena de texto.
    - None: Si no se encuentra ningún número.
    """
    # Utilizar expresión regular para encontrar todos los números en la cadena
    numeros = re.findall(r'\d+', texto)

    # Asumiendo que solo hay un número en la cadena, obtener el primer resultado
    numero = numeros[0] if numeros else None

    return numero





def get_bleu_per_seq(model, dataset: Im2LatexDataset, args: Munch, num_batches: int = None, name: str = 'test'):
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
    output = {}
    iter_ds = iter(dataset)
    pbar = tqdm(enumerate(iter_ds), total=len(dataset))

# 
    for i, (seq, im) in pbar:
        #image_index = dataset.indices[i]
        try:
            if seq is None or im is None:
                continue
            
            #genero prediccion
            dec = model.generate(im.to(device), temperature=args.get('temperature', .2))
            pred = detokenize(dec, dataset.tokenizer)
            truth = detokenize(seq['input_ids'], dataset.tokenizer)
            #calculo bleu score
            bleu = metrics.bleu_score(pred, [alternatives(x) for x in truth])
            #armo la expresion de vuelta
            pred2 = token2str(dec, dataset.tokenizer)
            truth2 = token2str(seq['input_ids'], dataset.tokenizer)
            #post process y edit_dist
            edit_dist = []
            for predi, truthi in zip(pred2, truth2):
                ts = post_process(truthi)
                if len(ts) > 0:
                    edit_dist.append(distance(post_process(predi), ts)/len(ts))           
            
            #Busco el nombre de la imagen 
            batch = iter_ds.pairs[iter_ds.i - 1]
            _,ims=batch.T
            label = extraer_numero(ims[0])
            output[label] = {
                "Truth":truth2,
                "predicted":pred2,
                "Bleu_score": bleu,
                "Edit dist": np.mean(edit_dist)}
            
        except IndexError:
            output[i] = "IndexError"
            

    
    
    return output