# pix2tex - LaTeX OCR

[![GitHub](https://img.shields.io/github/license/lukas-blecher/LaTeX-OCR)](https://github.com/lukas-blecher/LaTeX-OCR) [![PyPI](https://img.shields.io/pypi/v/pix2tex?logo=pypi)](https://pypi.org/project/pix2tex) [![PyPI - Downloads](https://img.shields.io/pypi/dm/pix2tex?logo=pypi)](https://pypi.org/project/pix2tex) [![GitHub all releases](https://img.shields.io/github/downloads/lukas-blecher/LaTeX-OCR/total?color=blue&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR/releases) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukas-blecher/LaTeX-OCR/blob/main/notebooks/LaTeX_OCR_test.ipynb) 

The goal of this project is to create a learning based system that takes an image of a math formula and returns corresponding LaTeX code. 

![header](https://user-images.githubusercontent.com/55287601/109183599-69431f00-778e-11eb-9809-d42b9451e018.png)

## Using the model
To run the model you need Python 3.7+

Install the package `pix2tex`: 

```
pip install pix2tex[gui]
```

Model checkpoints will be downloaded automatically.

There are three ways to get a prediction from an image. 
1. You can use the command line tool by calling `pix2tex`. Here you can parse already existing images from the disk and images in your clipboard.

2. Thanks to [@katie-lim](https://github.com/katie-lim), you can use a nice user interface as a quick way to get the model prediction. Just call the GUI with `latexocr`. From here you can take a screenshot and the predicted latex code is rendered using [MathJax](https://www.mathjax.org/) and copied to your clipboard.

    Under linux, it is possible to use the GUI with `gnome-screenshot` which comes with multiple monitor support. You just need to run `latexocr --gnome` (**Note:** you should install `gnome-screenshot` beforehand).

    ![demo](https://user-images.githubusercontent.com/55287601/117812740-77b7b780-b262-11eb-81f6-fc19766ae2ae.gif)

    If the model is unsure about the what's in the image it might output a different prediction every time you click "Retry". With the `temperature` parameter you can control this behavior (low temperature will produce the same result).

3. You can use an API. This has additional dependencies. Install via `pip install -U pix2tex[api]` and run
    ```bash
    python -m pix2tex.api.run
    ```
    to start a [Streamlit](https://streamlit.io/) demo that connects to the API at port 8502.

The model works best with images of smaller resolution. That's why I added a preprocessing step where another neural network predicts the optimal resolution of the input image. This model will automatically resize the custom image to best resemble the training data and thus increase performance of images found in the wild. Still it's not perfect and might not be able to handle huge images optimally, so don't zoom in all the way before taking a picture. 

Always double check the result carefully. You can try to redo the prediction with an other resolution if the answer was wrong.

## Training the model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukas-blecher/LaTeX-OCR/blob/main/notebooks/LaTeX_OCR_training.ipynb)

1. First we need to combine the images with their ground truth labels. I wrote a dataset class (which needs further improving) that saves the relative paths to the images with the LaTeX code they were rendered with. To generate the dataset pickle file run 

```
python -m pix2tex.dataset.dataset --equations path_to_textfile --images path_to_images --out dataset.pkl
```
To use your own tokenizer pass it via `--tokenizer` (See below).

You can find my generated training data on the [Google Drive](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO) as well (formulae.zip - images, math.txt - labels). Repeat the step for the validation and test data. All use the same label text file.

2. Edit the `data` (and `valdata`) entry in the config file to the newly generated `.pkl` file. Change other hyperparameters if you want to. See `pix2tex/model/settings/config.yaml` for a template.
3. Now for the actual training run 
```
python -m pix2tex.train --config path_to_config_file
```

If you want to use your own data you might be interested in creating your own tokenizer with
```
python -m pix2tex.dataset.dataset --equations path_to_textfile --vocab-size 8000 --out tokenizer.json
```
Don't forget to update the path to the tokenizer in the config file and set `num_tokens` to your vocabulary size.

## Model
The model consist of a ViT [[1](#References)] encoder with a ResNet backbone and a Transformer [[2](#References)] decoder.

### Performance
| BLEU score | normed edit distance | token accuracy |
| ---------- | -------------------- | -------------- |
| 0.88       | 0.10                 | 0.60           |

## Data
We need paired data for the network to learn. Luckily there is a lot of LaTeX code on the internet, e.g. [wikipedia](https://www.wikipedia.org), [arXiv](https://www.arxiv.org). We also use the formulae from the [im2latex-100k](https://zenodo.org/record/56198#.V2px0jXT6eA) [[3](#References)] dataset.
All of it can be found [here](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO)

### Dataset Requirements
In order to render the math in many different fonts we use  XeLaTeX, generate a PDF and finally convert it to a PNG. For the last step we need to use some third party tools: 
* [XeLaTeX](https://www.ctan.org/pkg/xetex)
* [ImageMagick](https://imagemagick.org/) with [Ghostscript](https://www.ghostscript.com/index.html). (for converting pdf to png)
* [Node.js](https://nodejs.org/) to run [KaTeX](https://github.com/KaTeX/KaTeX) (for normalizing Latex code)
* Python 3.7+ & dependencies (specified in `setup.py`)

### Fonts
Latin Modern Math, GFSNeohellenicMath.otf, Asana Math, XITS Math, Cambria Math


## TODO
- [x] add more evaluation metrics
- [x] create a GUI
- [ ] add beam search
- [ ] support handwritten formulae (kinda done, see training colab notebook)
- [ ] reduce model size (distillation)
- [ ] find optimal hyperparameters
- [ ] tweak model structure
- [ ] fix data scraping and scrape more data
- [ ] trace the model ([#2](https://github.com/lukas-blecher/LaTeX-OCR/issues/2))


## Contribution
Contributions of any kind are welcome.

## Acknowledgment
Code taken and modified from [lucidrains](https://github.com/lucidrains), [rwightman](https://github.com/rwightman/pytorch-image-models), [im2markup](https://github.com/harvardnlp/im2markup), [arxiv_leaks](https://github.com/soskek/arxiv_leaks), [pkra: Mathjax](https://github.com/pkra/MathJax-single-file), [harupy: snipping tool](https://github.com/harupy/snipping-tool)

## References
[1] [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

[2] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[3] [Image-to-Markup Generation with Coarse-to-Fine Attention](https://arxiv.org/abs/1609.04938v2)
