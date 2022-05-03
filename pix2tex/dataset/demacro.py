# modified from https://tex.stackexchange.com/a/521639

import argparse
import re
import logging
from collections import Counter
import time
from pix2tex.dataset.extract_latex import remove_labels


class DemacroError(Exception):
    pass


def main():
    args = parse_command_line()
    data = read(args.input)
    data = pydemacro(data)
    if args.output is not None:
        write(args.output, data)
    else:
        print(data)


def parse_command_line():
    parser = argparse.ArgumentParser(description='Replace \\def with \\newcommand where possible.')
    parser.add_argument('input', help='TeX input file with \\def')
    parser.add_argument('--output', '-o', default=None, help='TeX output file with \\newcommand')
    return parser.parse_args()


def read(path):
    with open(path, mode='r') as handle:
        return handle.read()


def bracket_replace(string: str) -> str:
    '''
    replaces all layered brackets with special symbols
    '''
    layer = 0
    out = list(string)
    for i, c in enumerate(out):
        if c == '{':
            if layer > 0:
                out[i] = 'Ḋ'
            layer += 1
        elif c == '}':
            layer -= 1
            if layer > 0:
                out[i] = 'Ḍ'
    return ''.join(out)


def undo_bracket_replace(string):
    return string.replace('Ḋ', '{').replace('Ḍ', '}')


def sweep(t, cmds):
    num_matches = 0
    for c in cmds:
        nargs = int(c[1][1]) if c[1] != r'' else 0
        optional = c[2] != r''
        if nargs == 0:
            num_matches += len(re.findall(r'\\%s([\W_^\dĊ])' % c[0], t))
            if num_matches > 0:
                t = re.sub(r'\\%s([\W_^\dĊ])' % c[0], r'%s\1' % c[-1].replace('\\', r'\\'), t)
        else:
            matches = re.findall(r'(\\%s(?:\[(.+?)\])?' % c[0]+r'{(.+?)}'*(nargs-(1 if optional else 0))+r')', t)
            num_matches += len(matches)
            for i, m in enumerate(matches):
                r = c[-1]
                if m[1] == r'':
                    matches[i] = (m[0], c[2][1:-1], *m[2:])
                for j in range(1, nargs+1):
                    r = r.replace(r'#%i' % j, matches[i][j+int(not optional)])
                t = t.replace(matches[i][0], r)
    return t, num_matches


def unfold(t):
    #t = queue.get()
    t = t.replace('\n', 'Ċ')
    t = bracket_replace(t)
    commands_pattern = r'\\(?:re)?newcommand\*?{\\(.+?)}[\sĊ]*(\[\d\])?[\sĊ]*(\[.+?\])?[\sĊ]*{(.*?)}'
    cmds = re.findall(commands_pattern, t)
    t = re.sub(r'(?<!\\)'+commands_pattern, 'Ċ', t)
    cmds = sorted(cmds, key=lambda x: len(x[0]))
    cmd_names = Counter([c[0] for c in cmds])
    for i in reversed(range(len(cmds))):
        if cmd_names[cmds[i][0]] > 1:
            # something went wrong here. No multiple definitions allowed
            del cmds[i]
        elif '\\newcommand' in cmds[i][-1]:
            logging.debug("Command recognition pattern didn't work properly. %s" % (undo_bracket_replace(cmds[i][-1])))
            del cmds[i]
    start = time.time()
    try:
        for i in range(10):
            # check for up to 10 nested commands
            if i > 0:
                t = bracket_replace(t)
            t, N = sweep(t, cmds)
            if time.time()-start > 5: # not optimal. more sophisticated methods didnt work or are slow
                raise TimeoutError
            t = undo_bracket_replace(t)
            if N == 0 or i == 9:
                #print("Needed %i iterations to demacro" % (i+1))
                break
            elif N > 4000:
                raise ValueError("Too many matches. Processing would take too long.")
    except ValueError:
        pass
    except TimeoutError:
        pass
    except re.error as e:
        raise DemacroError(e)
    t = remove_labels(t.replace('Ċ', '\n'))
    # queue.put(t)
    return t


def pydemacro(t: str) -> str:
    r"""Replaces all occurences of newly defined Latex commands in a document.
    Can replace `\newcommand`, `\def` and `\let` definitions in the code.

    Args:
        t (str): Latex document

    Returns:
        str: Document without custom commands
    """
    return unfold(convert(re.sub('\n+', '\n', re.sub(r'(?<!\\)%.*\n', '\n', t))))


def replace(match):
    prefix = match.group(1)
    if (
            prefix is not None and
            (
                'expandafter' in prefix or
                'global' in prefix or
                'outer' in prefix or
                'protected' in prefix
            )
    ):
        return match.group(0)

    result = r'\newcommand'
    if prefix is None or 'long' not in prefix:
        result += '*'

    result += '{' + match.group(2) + '}'
    if match.lastindex == 3:
        result += '[' + match.group(3) + ']'

    result += '{'
    return result


def convert(data):
    data = re.sub(
        r'((?:\\(?:expandafter|global|long|outer|protected)(?:\s+|\r?\n\s*)?)*)?\\def\s*(\\[a-zA-Z]+)\s*(?:#+([0-9]))*\{',
        replace,
        data,
    )
    return re.sub(r'\\let[\sĊ]*(\\[a-zA-Z]+)\s*=?[\sĊ]*(\\?\w+)*', r'\\newcommand*{\1}{\2}\n', data)


def write(path, data):
    with open(path, mode='w') as handle:
        handle.write(data)

    print('=> File written: {0}'.format(path))


if __name__ == '__main__':
    main()
