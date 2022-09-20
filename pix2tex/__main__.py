#!/usr/bin/env python
def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml', help='path to config file')
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth', help='path to weights file')
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    parser.add_argument('--no-resize', action='store_true', help='Resize the image beforehand')

    parser.add_argument('-s', '--show', action='store_true', help='Show the rendered predicted latex code (cli only)')
    parser.add_argument('-k', '--katex', action='store_true', help='Render the latex code in the browser (cli only)')

    parser.add_argument('--gui', action='store_true', help='Use GUI (gui only)')

    parser.add_argument('file', nargs='*', type=str, default=None, help='Predict LaTeX code from image file instead of clipboard (cli only)')
    arguments = parser.parse_args()

    import os
    import sys

    name = os.path.split(sys.argv[0])[-1]
    if arguments.gui or name in ['pix2tex_gui', 'latexocr']:
        from .gui import main
    else:
        from .cli import main
    main(arguments)


if __name__ == '__main__':
    main()
