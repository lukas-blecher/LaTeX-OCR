#!/bin/sh

mkdir -p build
docker build . -f appimage.Dockerfile -t tmp/pix2tex-appimage
docker run --rm -v $PWD/build:/result tmp/pix2tex-appimage
