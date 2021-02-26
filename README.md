# pix2tex - LaTeX OCR
The goal of this project is to create a learning based system that takes an image of a math formula and returns corresponding LaTeX code. As a physics student I often find myself writing down Latex code from a reference image. I wanted to streamline my workflow and began looking into solutions, but besides the Freemium [Mathpix](https://mathpix.com/) I could not find anything ready-to-use that runs locally. That's why I decided to create it myself.

![header](https://user-images.githubusercontent.com/55287601/109183599-69431f00-778e-11eb-9809-d42b9451e018.png)

## Using the model
1. Download/Clone this repository
2. For now you need to install the Python dependencies specified in `requirements.txt` (look [further down](#Requirements))
3. Download the `weights.pth` file from my [Google Drive](https://drive.google.com/drive/folders/1cgmyiaT5uwQJY2pB0ngebuTcK5ivKXIb) and place it in the `checkpoints` directory

The `pix2tex.py` file offers a quick way to get the model prediction of an image. First you need to copy the formula image into the clipboard memory for example by using a snipping tool (on Windows built in `Win`+`Shift`+`S`). Next just call the script with `python pix2tex.py`. It will print out the predicted Latex code for that image and also copy it into your clipboard.

**Note:** As of right now it works best with images of smaller resolution. Don't zoom in all the way before taking a picture. Double check the result carefully. You can try to redo the prediction with an other resolution if the answer was wrong.

## Model
The model consist of a ViT [[1](#References)] encoder with a ResNet backbone and a Transformer [[2](#References)] decoder.

### Performance
BLEU score: 0.87

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
  install `timm` directly `pip install -U git+https://github.com/rwightman/pytorch-image-models.git`
### Dataset
In order to render the math in many different fonts we use  XeLaTeX, generate a PDF and finally convert it to a PNG. For the last step we need to use some third party tools: 
* [XeLaTeX](https://www.ctan.org/pkg/xetex)
* [ImageMagick](https://imagemagick.org/) with [Ghostscript](https://www.ghostscript.com/index.html). (for converting pdf to png)
* [Node.js](https://nodejs.org/) to run [KaTeX](https://github.com/KaTeX/KaTeX) (for normalizing Latex code)
* [`de-macro`](https://www.ctan.org/pkg/de-macro) >= 1.4 (only for parsing arxiv papers)
* Python 3.7+ & dependencies (`requirements.txt`)

## TODO
- [ ] support handwritten formulae
- [ ] reduce model size
- [ ] find optimal hyperparameters
- [ ] create a standalone application


## Contribution
Contributions of any kind are welcome.

## Acknowledgement
Code taken and modified from [lucidrains](https://github.com/lucidrains), [rwightman](https://github.com/rwightman/pytorch-image-models), [im2markup](https://github.com/harvardnlp/im2markup), [arxiv_leaks](https://github.com/soskek/arxiv_leaks)

## References
[1] [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

[2] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
