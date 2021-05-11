# pix2tex - LaTeX OCR
The goal of this project is to create a learning based system that takes an image of a math formula and returns corresponding LaTeX code. 

![header](https://user-images.githubusercontent.com/55287601/109183599-69431f00-778e-11eb-9809-d42b9451e018.png)

## Requirements
### Model
* PyTorch (tested on v1.7.1)
* Python 3.7+ & dependencies (`requirements.txt`)
  ```
  pip install -r requirements.txt
  ```
### Dataset
In order to render the math in many different fonts we use  XeLaTeX, generate a PDF and finally convert it to a PNG. For the last step we need to use some third party tools: 
* [XeLaTeX](https://www.ctan.org/pkg/xetex)
* [ImageMagick](https://imagemagick.org/) with [Ghostscript](https://www.ghostscript.com/index.html). (for converting pdf to png)
* [Node.js](https://nodejs.org/) to run [KaTeX](https://github.com/KaTeX/KaTeX) (for normalizing Latex code)
* [`de-macro`](https://www.ctan.org/pkg/de-macro) >= 1.4 (only for parsing arxiv papers)
* Python 3.7+ & dependencies (`requirements.txt`)

## Using the model
1. Download/Clone this repository
2. For now you need to install the Python dependencies specified in `requirements.txt` (look [above](#Requirements))
3. Download the `weights.pth` (and optionally `image_resizer.pth`) file from my [Google Drive](https://drive.google.com/drive/folders/1cgmyiaT5uwQJY2pB0ngebuTcK5ivKXIb) and place it in the `checkpoints` directory

Thanks to [@katie-lim](https://github.com/katie-lim), you can use a nice user interface as a quick way to get the model prediction. Just call the GUI with `python gui.py`. From here you can take a screenshot and the predicted latex code is rendered using [MathJax](https://www.mathjax.org/) and copied to your clipboard.

![demo](https://user-images.githubusercontent.com/55287601/117812740-77b7b780-b262-11eb-81f6-fc19766ae2ae.gif)

If the model is unsure about the what's in the image it might output a different prediction every time you click "Retry". With the `temperature` parameter you can control this behavior (low temperature will produce the same result).

Alternatively you can use `pix2tex.py` with similar functionality as `gui.py`, only as command line tool. In this case you don't need to install PyQt5. Using this script you can also parse already existing images from the disk.

**Note:** As of right now it works best with images of smaller resolution. Don't zoom in all the way before taking a picture. Double check the result carefully. You can try to redo the prediction with an other resolution if the answer was wrong.

**Update:** I have trained an image classifier on randomly scaled images of the training data to predict the original size.
This model will automatically resize the custom image to best resemble the training data and thus increase performance of images found in the wild. To use this preprocessing step, all you have to do is download the second weights file mentioned above. You should be able to take bigger (or smaller) images of the formula and still get a satisfying result

## Training the model
1. First we need to combine the images with their ground truth labels. I wrote a dataset class (which needs further improving) that saves the relative paths to the images with the LaTeX code they were rendered with. To generate the dataset pickle file run 

```
python dataset/dataset.py --equations path_to_textfile --images path_to_images --tokenizer path_to_tokenizer --out dataset.pkl
```

You can find my generated training data on the [Google Drive](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO) as well (formulae.zip - images, math.txt - labels). Repeat the step for the validation and test data. All use the same label text file.

2. Edit the `data` entry in the config file to the newly generated `.pkl` file. Change other hyperparameters if you want to. See `settings/default.yaml` for a template.
3. Now for the actual training run 
```
python train.py --config path_to_config_file
```


## Model
The model consist of a ViT [[1](#References)] encoder with a ResNet backbone and a Transformer [[2](#References)] decoder.

### Performance
| BLEU score | normed edit distance |
| ---------- | -------------------- |
| 0.88       | 0.10                 |

## Data
We need paired data for the network to learn. Luckily there is a lot of LaTeX code on the internet, e.g. [wikipedia](www.wikipedia.org), [arXiv](www.arxiv.org). We also use the formulae from the [im2latex-100k](https://zenodo.org/record/56198#.V2px0jXT6eA) dataset.
All of it can be found [here](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO)

### Fonts
Latin Modern Math, GFSNeohellenicMath.otf, Asana Math, XITS Math, Cambria Math


## TODO
- [x] add more evaluation metrics
- [x] create a GUI
- [ ] add beam search
- [ ] support handwritten formulae
- [ ] reduce model size (distillation)
- [ ] find optimal hyperparameters
- [ ] tweak model structure
- [ ] fix data scraping and scrape more data
- [ ] trace the model


## Contribution
Contributions of any kind are welcome.

## Acknowledgment
Code taken and modified from [lucidrains](https://github.com/lucidrains), [rwightman](https://github.com/rwightman/pytorch-image-models), [im2markup](https://github.com/harvardnlp/im2markup), [arxiv_leaks](https://github.com/soskek/arxiv_leaks), [pkra: Mathjax](https://github.com/pkra/MathJax-single-file), [harupy: snipping tool](https://github.com/harupy/snipping-tool)

## References
[1] [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

[2] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
