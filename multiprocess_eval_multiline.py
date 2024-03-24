import pickle
# import pix2tex
from munch import Munch
import yaml
from pix2tex import cli as p2t
from pix2tex.dataset.dataset import Im2LatexDataset
import multiprocessing
from metrics_facu import get_bleu_per_seq

multiline_config = Munch({
    "config": "../../multiline_checkpoints/local_multiline_pix2text/config.yaml",
    "checkpoint": "../../multiline_checkpoints/local_multiline_pix2text/local_multiline_pix2text_e01_step99.pth",
    "no_cuda": True,
    "no_resize": False,
    "device" : "cpu"
})

original_config = Munch({
    "config": "settings/config.yaml",
    "checkpoint": "model/checkpoints/weights.pth",
    "no_cuda": True,
    "no_resize": False,
    "device": "cpu"
}) 
multiline_model = p2t.LatexOCR(multiline_config)
original_model = p2t.LatexOCR()
# load test set for handwritten files
# load yaml files to parse configurations
with open("multiline_checkpoints/local_multiline_pix2text/config.yaml", 'r') as f:
    ml_config_yaml = Munch(yaml.safe_load(f))

with open("pix2tex/model/settings/config.yaml", 'r') as f:
    original_config_yaml = Munch(yaml.safe_load(f))

# just in case, set the device to cpu manually
ml_config_yaml.device = "cpu"
original_config_yaml.device = "cpu"

# test dataset
test_dataset = Im2LatexDataset().load("pix2tex/dataset/multiline/test_dataset.pkl")

def evaluate_and_pickle(model, dataset, config, fname):
    """
    Evaluate the model on the given dataset and store the results
    """
    results = get_bleu_per_seq(model, dataset, config)

    with open("notebooks/multiline_performance/{}.pkl".format(fname), "wb") as f:
        pickle.dump(results, f)
    print("Results stored in {}.pkl".format(fname))


if __name__ == "__main__":
    # Compute multiprocessing
    NUM_CORES = 4
    pool = multiprocessing.Pool(NUM_CORES)
    args = [(multiline_model.model, test_dataset, ml_config_yaml, "MP-multiline_results"),
             (original_model.model, test_dataset, original_config_yaml, "MP-original_results")]
    pool.map(evaluate_and_pickle, args)

    pool.close()
    pool.join()
