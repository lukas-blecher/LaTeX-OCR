#! /bin/bash
USE_TF=OFF {{ python-executable }} -s \
	"${APPDIR}/opt/python{{ python-version }}/bin/pix2tex_gui" \
	--no-cuda "$@"
