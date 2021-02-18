# modified from https://tex.stackexchange.com/a/521639

import argparse
import re


def main():
    args = parse_command_line()
    data = read(args.input)
    data = convert(data)
    if args.demacro:
        data = unfold(data)
    write(args.output, data)


def parse_command_line():
    parser = argparse.ArgumentParser(description='Replace \\def with \\newcommand where possible.')
    parser.add_argument('input', help='TeX input file with \\def')
    parser.add_argument('--output', '-o', required=True, help='TeX output file with \\newcommand')
    parser.add_argument('--demacro', action='store_true', help='replace all commands with their definition')

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


def unfold(t):
    cmds = re.findall(r'\\(?:re)?newcommand\*?{\\(.+?)}\s*(\[\d\])?(\[.+?\])?{(.+?)}\n', t)
    cmds = sorted(cmds, key=lambda x: len(x[0]))
    # print(cmds)
    for c in cmds:
        nargs = int(c[1][1]) if c[1] != r'' else 0
        # print(c)
        if nargs == 0:
            #t = t.replace(r'\\%s' % c[0], c[-1])
            t = re.sub(r'\\%s([\W_^\d])' % c[0], r'%s\1' % c[-1].replace('\\', r'\\'), t)
        else:
            matches = re.findall(r'(\\%s(?:\[(.+?)\])?' % c[0]+r'{(.+?)}'*(nargs-(1 if c[2] != r'' else 0))+r')', t)
            # print(matches)
            for i, m in enumerate(matches):
                r = c[-1]
                if m[1] == r'':
                    matches[i] = (m[0], c[2][1:-1], *m[2:])
                for j in range(1, nargs+1):
                    r = r.replace(r'#%i' % j, matches[i][j])
                t = t.replace(matches[i][0], r)
    return t


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
