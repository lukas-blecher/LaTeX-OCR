#!/bin/bash

if [ $1 == "setup" ]; then
    echo "Setting up python virtual environment"
    echo "Entering virtual environment"
    source ./venv/bin/activate

    pip3 install 'pix2tex[train]'
    pip3 install pytorch-lightning rich

    # install and login wandb
    pip3 install wandb
    wandb login

elif [ $1 == "generate" ]; then
    echo "Generate images dataset"
    # eg. python3 -m pix2tex.dataset.dataset --equations path_to_textfile --images path_to_images --out dataset.pkl
    python3 -m pix2tex.dataset.dataset --equations pix2tex/dataset/data/math.txt --images pix2tex/dataset/data/train --out pix2tex/dataset/data/train.pkl
    python3 -m pix2tex.dataset.dataset --equations pix2tex/dataset/data/math.txt --images pix2tex/dataset/data/val --out pix2tex/dataset/data/val.pkl

elif [ $1 == "train" ]; then
    echo "Training model"
    python3 -m pix2tex.train --config pix2tex/model/settings/config.yaml

else
    echo "Invalid argument"
    echo "Usage: ./run.sh [setup|generate|train|test]"
fi
