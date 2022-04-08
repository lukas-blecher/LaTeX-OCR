FROM python:3.8-slim-buster
VOLUME /result
RUN mkdir -p /result
RUN apt-get update && \
	apt-get install -y git file gpg wget build-essential python3-dev && \
	pip install git+https://github.com/niess/python-appimage

RUN groupadd -r appimage && useradd -m --no-log-init -r -g appimage appimage

WORKDIR /latexocr

ADD setup.py /latexocr/
ADD pix2tex /latexocr/pix2tex/
ADD appimage /latexocr/appimage/

RUN chown -R appimage:appimage /latexocr

USER appimage

RUN cp pix2tex/resources/icon.svg appimage/icon.svg && \
	cp appimage/pre_requirements.txt appimage/requirements.txt && \
	echo "$(pwd)[gui]" >> appimage/requirements.txt


RUN PIP_NO_CACHE_DIR=off python -m python_appimage build app -p 3.8 /latexocr/appimage

USER root
CMD cp /latexocr/*.AppImage /result
