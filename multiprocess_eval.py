import pickle
from PIL import Image
import os
import numpy as np
import pix2tex as p2t
from munch import Munch
import yaml
from pix2tex import cli as p2t
import matplotlib.pyplot as plt
from pix2tex.dataset.dataset import Im2LatexDataset
from pix2tex.eval import evaluate
from torchtext.data import metrics
import multiprocessing

hw_config = Munch({
    "config": "settings/handwritten_training.yaml",
    "checkpoint": "../../hw_checkpoints/handwritten_training/handwritten_training_e19_step63.pth",
    "no_cuda": True,
    "no_resize": False
})

original_config = Munch({
    "config": "settings/config.yaml",
    "checkpoint": "model/checkpoints/weights.pth",
    "no_cuda": True,
    "no_resize": False
}) 
hw_model = p2t.LatexOCR(hw_config)
original_model = p2t.LatexOCR()
# load test set for handwritten files
# hw_test_set = Im2LatexDataset().load("pix2tex/dataset/handwritten/test_hw.pkl")
# original_test_set = Im2LatexDataset().load("pix2tex/dataset/formulae/test.pkl")
# load yaml files to parse configurations
with open("pix2tex/model/settings/handwritten_training.yaml", 'r') as f:
    hw_config_yaml = Munch(yaml.safe_load(f))

with open("pix2tex/model/settings/config.yaml", 'r') as f:
    original_config_yaml = Munch(yaml.safe_load(f))

hw_config_yaml.device = "cpu"



def compute_pred(fname):
    """
    Compute the prediction of the original and 
    HW model on a given image
    """
    img_path = os.path.join("pix2tex/dataset/formulae", "test", fname)
    img = Image.open(img_path)

    original_pred = original_model(img, original_config_yaml)
    hw_pred = hw_model(img, hw_config_yaml)

    label_idx = int(fname.split(".")[0])

    return (label_idx, hw_pred, original_pred)

if __name__ == "__main__":
    # Compute multiprocessing
    NUM_CORES = 4
    pool = multiprocessing.Pool(NUM_CORES)
    fnames = os.listdir("pix2tex/dataset/formulae/test/")
    results = pool.map(compute_pred, fnames[:1500])

    pool.close()
    pool.join()

    STORED_HW = {label_idx: hw_pred for label_idx, hw_pred, _ in results}
    STORED_ORIGINAL = {label_idx: original_pred for label_idx, _, original_pred in results}

    with open("hw_results-original-data.pkl", "wb") as f:
        pickle.dump(STORED_HW, f)
    
    with open("original_results-original-data.pkl", "wb") as f:
        pickle.dump(STORED_ORIGINAL, f)
