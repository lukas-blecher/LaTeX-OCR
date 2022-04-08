#!/usr/bin/env python

import setuptools
import os
from pix2tex.model.checkpoints.get_latest_checkpoint import download_checkpoints

setuptools.setup(
    name='pix2tex',
    version='0.0.3',
    description="pix2tex: Using a ViT to convert images of equations into LaTeX code.",
    author='Lukas Blecher',
    url='https://lukas-blecher.github.io/LaTeX-OCR/',
    packages=setuptools.find_packages(),
    package_data={
        'pix2tex': [
            'resources/*',
            'model/checkpoints/*.pth',
            'model/settings/*.yaml',
            'model/dataset/*.json',
        ]
    },
    install_requires=[
        "tqdm>=4.47.0",
        "munch>=2.5.0",
        "torch>=1.7.1",
        "torchvision>=0.8.1",
        "opencv_python_headless>=4.1.1.26",
        "requests>=2.22.0",
        "einops>=0.3.0",
        "chardet>=3.0.4",
        "x_transformers==0.15.0",
        "imagesize>=1.2.0",
        "transformers==4.2.2",
        "tokenizers==0.9.4",
        "numpy>=1.19.5",
        "Pillow>=8.1.0",
        "PyYAML>=5.4.1",
        "torchtext>=0.6.0",
        "albumentations>=0.5.2",
        "pandas>=1.0.0",
        "timm",
        "python-Levenshtein>=0.12.2",
    ],
    extras_require={
        "gui":  [
            "PyQt5",
            "PyQtWebEngine",
            "pynput",
            "screeninfo",
        ]
    },
    entry_points={
        'console_scripts': [
            'pix2tex_gui = pix2tex.gui:main',
            'pix2tex_cli = pix2tex.cli:main',
        ],
    }
)

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'pix2tex', 'models', 'checkpoints', 'weights.pth')):
    download_checkpoints()
