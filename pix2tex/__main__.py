#!/usr/bin/env python
# https://github.com/iterative/shtab/blob/5358dda86e8ea98bf801a43a24ad73cd9f820c63/examples/customcomplete.py#L11-L22
YAML_FILE = {
    "bash": "_shtab_greeter_compgen_yaml_file",
    "zsh": "_files -g '*.yaml'",
    "tcsh": "f:*.yaml",
}
PTH_FILE = {
    "bash": "_shtab_greeter_compgen_pth_file",
    "zsh": "_files -g '*.pth'",
    "tcsh": "f:*.pth",
}
PREAMBLE = {
    "bash": """\
# $1=COMP_WORDS[1]
_shtab_greeter_compgen_yaml_file() {
  compgen -d -- $1  # recurse into subdirs
  compgen -f -X '!*?.yaml' -- $1
}

_shtab_greeter_compgen_pth_file() {
  compgen -d -- $1  # recurse into subdirs
  compgen -f -X '!*?.pth' -- $1
}
""",
}


def main():
    from argparse import ArgumentParser
    try:
        import shtab
    except ImportError:
        from . import _shtab as shtab

    parser = ArgumentParser('pix2tex')
    shtab.add_argument_to(parser, preamble=PREAMBLE)

    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml', help='path to config file').complete = YAML_FILE
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth', help='path to weights file').complete = PTH_FILE
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    parser.add_argument('--no-resize', action='store_true', help='Resize the image beforehand')

    parser.add_argument('-s', '--show', action='store_true', help='Show the rendered predicted latex code (cli only)')
    parser.add_argument('-k', '--katex', action='store_true', help='Render the latex code in the browser (cli only)')

    parser.add_argument('--gui', action='store_true', help='Use GUI (gui only)')

    parser.add_argument('file', nargs='*', type=str, default=None, help='Predict LaTeX code from image file instead of clipboard (cli only)').complete = shtab.FILE
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
