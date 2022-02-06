#!/usr/bin/env python3

'''Simple installer for the graphical user interface of pix2tex'''

import argparse
import os
import sys


def _check_file(
    main_file
):
    if os.path.exists(main_file):
        return
    raise FileNotFoundError(
        f'Unable to find file {main_file}'
    )


def _make_desktop_file(
    desktop_path,
    desktop_entry
):
    with open(desktop_path, 'w') as desktop_file:
        desktop_file.write(desktop_entry)


def setup_desktop(
    gui_file = 'gui.py',
    icon_file = 'resources/icon.svg',
):
    '''Main function for setting up .desktop files (on Linux)'''
    parser = argparse.ArgumentParser(
        description='Simple installer for the pix2tex GUI'
    )

    parser.add_argument(
        'pix2tex_dir',
        default='.',
        nargs='?',
        help='The directory where pix2tex was downloaded'
    )

    parser.add_argument(
        '--uninstall', '-u',
        action='store_true',
        help='Uninstalls the desktop entry'
    )

    parser.add_argument(
        '--venv_dir', '-e',
        help='In case a virtual environment is needed for running pix2tex, specifies its directory'
    )

    parser.add_argument(
        '--overwrite', '-o',
        action='store_true',
        help='Unconditionally overwrite .desktop file (if it exists)'
    )

    args = parser.parse_args()

    # where the desktop file will be created
    desktop_dir = os.environ.get(
        'XDG_DATA_HOME',
        os.path.join(os.environ.get('HOME'), '.local/share/applications')
    )
    desktop_path = os.path.abspath(os.path.join(desktop_dir, 'pix2tex.desktop'))

    # check if we want to uninstall it instead
    if args.uninstall:
        if os.path.exists(desktop_path):
            remove = input(
                f'Are you sure you want to remove the pix2tex desktop entry {desktop_path}? [y/n]'
            )
            if remove.lower() == 'y':
                try:
                    os.remove(desktop_path)
                    print('Successfully uninstalled the desktop entry')
                    return 0
                except:
                    raise OSError(
                        f'Something went wrong, unable to remove the desktop entry {desktop_path}'
                    )
            elif remove.lower() == 'n':
                print(
                    'Not removing the desktop entry;' \
                    'if you wish to install/uninstall pix2tex, please run this script again'
                )
                return 0
        else:
            print('No file to remove')
            return 0

    _check_file(os.path.join(args.pix2tex_dir, gui_file))
    _check_file(os.path.join(args.pix2tex_dir, icon_file))

    pix2tex_dir = os.path.abspath(args.pix2tex_dir)
    gui_path = os.path.join(pix2tex_dir, gui_file)
    icon_path = os.path.join(pix2tex_dir, icon_file)

    interpreter_path = \
        os.path.join(args.venv_dir, 'bin/python3') \
        if (args.venv_dir and os.path.exists(os.path.join(args.venv_dir, 'bin/python3'))) \
        else sys.executable
    interpreter_path = os.path.abspath(interpreter_path)

    desktop_entry = f"""[Desktop Entry]
Version=1.0
Name=pix2tex
Comment=LaTeX math recognition using machine learning
Exec={interpreter_path} {gui_path}
Icon={icon_path}
Terminal=false
Type=Application
Categories=Utility;
"""

    if os.path.exists(desktop_path):
        if not args.overwrite:
            overwrite = input(
                f'Desktop entry {desktop_path} exists, do you wish to overwrite it? [y/n]'
            )
            if overwrite.lower() == 'y':
                _make_desktop_file(desktop_path, desktop_entry)
            elif overwrite.lower() == 'n':
                print('Not overwriting existing desktop entry, exiting...', file=sys.stderr)
                return 1
            else:
                print('Unable to understand input, exiting...', file=sys.stderr)
                return 255
        else:
            _make_desktop_file(desktop_path, desktop_entry)
    else:
        _make_desktop_file(desktop_path, desktop_entry)

    return 0


if __name__ == '__main__':
    setup_desktop()
