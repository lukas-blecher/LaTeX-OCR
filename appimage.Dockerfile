FROM python:3.8-slim-buster
VOLUME /result

RUN apt-get update && \
	apt-get install -y git file gpg wget build-essential python3-dev && \
	pip install git+https://github.com/niess/python-appimage

RUN mkdir -p /math2pix/math2pix/model/checkpoints && cd /math2pix/math2pix/model/checkpoints/ && \
	wget -nc https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/weights.pth && \
	wget -nc https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/image_resizer.pth

RUN groupadd -r appimage && useradd -m --no-log-init -r -g appimage appimage

WORKDIR /math2pix

ADD setup.py /math2pix/
ADD pix2tex /math2pix/pix2tex/
ADD appimage /math2pix/appimage/

RUN chown -R appimage:appimage /math2pix

USER appimage

RUN cp pix2tex/resources/icon.svg appimage/icon.svg && \
	cp appimage/pre_requirements.txt appimage/requirements.txt && \
	echo "$(pwd)[gui]" >> appimage/requirements.txt

RUN mkdir -p /result

RUN PIP_NO_CACHE_DIR=off python -m python_appimage build app -p 3.8 /math2pix/appimage

USER root
CMD cp /math2pix/*.AppImage /result
