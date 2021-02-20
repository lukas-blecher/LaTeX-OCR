# pix2tex - LaTeX OCR
The goal of this project is to create a learning based system that takes an image of a math formula and returns corresponding LaTeX code. As a physics student I often find myself writing down Latex code from a reference image. I wanted to streamline my workflow and began looking into solutions, but besides the Freemium [Mathpix](https://mathpix.com/) I could not find anything ready-to-use that runs locally. That's why I decided to create it myself.


## Using the model
1. Download/Clone this repository
2. For now you need to install the Python dependencies specified in `requirements.txt` (look [further down](https://github.com/lukas-blecher/LaTeX-OCR#Requirements))
3. Download the `weights.pth` file and the `config.yaml` file from my [Google Drive](https://drive.google.com/drive/folders/1cgmyiaT5uwQJY2pB0ngebuTcK5ivKXIb) and place them in the `checkpoints` and `settings` directory respectively
4. Download the `tokenizer.json` from my Google Drive and place it in the `dataset/data` directory 

The `pix2tex.py` file offers a quick way to get the model prediction of an image. First you need to copy the formula image into the clipboard memory for example by using a snipping tool (on Windows built in `Win`+`Shift`+`S`). Next just call the script with `python pix2tex.py`. It will print out the predicted Latex code for that image and also copy it into your clipboard.

## Data
We need paired data for the network to learn. Luckily there is a lot of LaTeX code on the internet, e.g. [wikipedia](www.wikipedia.org), [arXiv](www.arxiv.org). We also use the formulae from the [im2latex-100k](https://zenodo.org/record/56198#.V2px0jXT6eA) dataset.

### Fonts
Latin Modern Math, GFSNeohellenicMath.otf, Asana Math, XITS Math, Cambria Math

## Requirements
### Evaluation
* PyTorch (tested on v1.7.0)
* Python 3.7+ & dependencies (`requirements.txt`)
  ```
  pip install -r requirements.txt
  ```
### Dataset
In order to render the math in many different fonts we use  XeLaTeX, generate a PDF and finally convert it to a PNG. For the last step we need to use some third party tools: 
* [XeLaTeX](https://www.ctan.org/pkg/xetex)
* [ImageMagick](https://imagemagick.org/) with [Ghostscript](https://www.ghostscript.com/index.html).
* [Node.js](https://nodejs.org/) to run [KaTeX](https://github.com/KaTeX/KaTeX)
* [`de-macro`](https://www.ctan.org/pkg/de-macro) >= 1.4
* Python 3.7+ & dependencies (`requirements.txt`)

## TODO
- [ ] support handwritten formulae
- [ ] reduce model size
- [ ] create a standalone application


## Contribution
Contributions of any kind are welcome.

## Acknowledgement
Code taken and modified from [lucidrains](https://github.com/lucidrains), [rwightman](https://github.com/rwightman/pytorch-image-models), [im2markup](https://github.com/harvardnlp/im2markup), [arxiv_leaks](https://github.com/soskek/arxiv_leaks)
