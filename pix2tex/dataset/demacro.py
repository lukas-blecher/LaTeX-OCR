# modified from https://tex.stackexchange.com/a/521639

import argparse
import re
from pix2tex.dataset.extract_latex import remove_labels


def main():
    args = parse_command_line()
    data = read(args.input)
    data = convert(data)
    data = unfold(data)
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


def convert(data):
    return re.sub(
        r'((?:\\(?:expandafter|global|long|outer|protected)'
        r'(?: +|\r?\n *)?)*)?'
        r'\\def *(\\[a-zA-Z]+) *(?:#+([0-9]))*\{',
        replace,
        data,
    )


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
            t = re.sub(r'\\%s([\W_^\d])' % c[0], r'%s\1' % c[-1].replace('\\', r'\\'), t)
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
    t = remove_labels(t).replace('\n', 'Ċ')

    cmds = re.findall(r'\\(?:re)?newcommand\*?{\\(.+?)}\s*(\[\d\])?(\[.+?\])?{(.+?)}Ċ', t)
    cmds = sorted(cmds, key=lambda x: len(x[0]))
    for _ in range(10):
        # check for up to 10 nested commands
        t = bracket_replace(t)
        t, N = sweep(t, cmds)
        t = undo_bracket_replace(t)
        if N == 0:
            break
    return t.replace('Ċ', '\n')


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


def write(path, data):
    with open(path, mode='w') as handle:
        handle.write(data)

    print('=> File written: {0}'.format(path))


if __name__ == '__main__':
    main()
