Installation
============

Python package
--------------

To run the model you need Python 3.7+

If you don't have PyTorch installed. Follow their instructions [here](https://pytorch.org/get-started/locally/).

Install the package `pix2tex`: 

```
pip install pix2tex[gui]
```

Model checkpoints will be downloaded automatically when first running the script.

To install
- with GUI dependencies use tag `[gui]`.
- with training dependencies use tag `[train]`.
- with api dependencies use tag `[api]`.
- all dependencies use tag `[all]`.

Docker
------

The API can be used from a docker container, available on [DockerHub](https://hub.docker.com/r/lukasblecher/pix2tex)
```
docker pull lukasblecher/pix2tex:api
docker run -p 8502:8502 lukasblecher/pix2tex:api
```
This starts the API which is available at port 8502.

To use the [Streamlit](https://streamlit.io/) demo run instead
```
docker run -it -p 8501:8501 --entrypoint python lukasblecher/pix2tex:api pix2tex/api/run.py
```
and navigate to [http://localhost:8501/](http://localhost:8501/)
