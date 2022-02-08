#! /bin/bash
USE_TF=OFF {{ python-executable }} \
	"${APPDIR}/opt/python{{ python-version }}/bin/pix2tex_gui" \
	--no-cuda "$@"
