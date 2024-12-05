# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
import os
import re
import sys
import io
import glob
import tempfile
import shlex
import subprocess
import traceback
from PIL import Image


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\setmathfont{%s}
\begin{document}
\thispagestyle{empty}
%s
\end{document}
'''

    def __init__(self, math, dpi=250, font='Latin Modern Math'):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        self.math = math
        self.dpi = dpi
        self.font = font
        self.prefix_line = self.BASE.split("\n").index(
            "%s")  # used for calculate error formula index

    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False
        try:
            workdir = tempfile.gettempdir()
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
            # print(self.BASE % (self.font, self.math))
            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % (self.font, '\n'.join(self.math))
                # print(document)
                f.write(document)

            png, error_index = self.convert_file(
                texfile, workdir, return_bytes=return_bytes)
            return png, error_index

        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False):
        infile = infile.replace('\\', '/')
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = 'xelatex -interaction nonstopmode -file-line-error -output-directory %s %s' % (
                workdir.replace('\\', '/'), infile)

            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            sout, serr = p.communicate()
            # extract error line from sout
            error_index, _ = extract(text=sout, expression=r"%s:(\d+)" % os.path.basename(infile))
            # extract success rendered equation
            if error_index != []:
                # offset index start from 0, same as self.math
                error_index = [int(_)-self.prefix_line-1 for _ in error_index]
            # Convert the PDF file to PNG's
            pdffile = infile.replace('.tex', '.pdf')
            result, _ = extract(
                text=sout, expression="Output written on %s \((\d+)? page" % pdffile)
            if int(result[0]) != len(self.math):
                raise Exception('xelatex rendering error, generated %d formula\'s page, but the total number of formulas is %d.' % (
                    int(result[0]), len(self.math)))
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))

            cmd = 'convert -density %i -colorspace gray %s -quality 90 %s' % (
                self.dpi,
                pdffile,
                pngfile,
            )  # -bg Transparent -z 9
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sout, serr = p.communicate()
            if p.returncode != 0:
                raise Exception('PDFpng error', serr, cmd, os.path.exists(
                    pdffile), os.path.exists(infile))
            if return_bytes:
                if len(self.math) > 1:
                    png = [open(pngfile.replace('.png', '')+'-%i.png' %
                                i, 'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace(
                        '.png', '')+'.png', 'rb').read()]
            else:
                # return path
                if len(self.math) > 1:
                    png = [(pngfile.replace('.png', '')+'-%i.png' % i)
                           for i in range(len(self.math))]
                else:
                    png = [(pngfile.replace('.png', '')+'.png')]
            return png, error_index
        except Exception as e:
            print(e)
        finally:
            # Cleanup temporaries
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            if return_bytes:
                ims = glob.glob(basefile+'*.png')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, return_error_index=False, **kwargs):
    pngs, error_index = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return (images, error_index) if return_error_index else images


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = r'\begin{equation}\mathcal{ L}\nonumber\end{equation}'

    print('Equation is: %s' % src)
    print(Latex([src]).write())
