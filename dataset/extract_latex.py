import re
import numpy as np

MIN_CHARS = 20
MAX_CHARS = 3000
dollar = re.compile(r'((?<!\$)\${1,2}(?!\$))(.{%i,%i}?)(?<!\\)(?<!\$)\1(?!\$)' % (1, MAX_CHARS))
inline = re.compile(r'(\\\((.*?)(?<!\\)\\\))|(\\\[(.{%i,%i}?)(?<!\\)\\\])' % (1, MAX_CHARS))
equation = re.compile(r'\\begin\{(equation|math|displaymath)\*?\}(.{%i,%i}?)\\end\{\1\*?\}' % (1, MAX_CHARS), re.S)
align = re.compile(r'(\\begin\{(align|alignedat|alignat|flalign|eqnarray|aligned|split|gather)\*?\}(.{%i,%i}?)\\end\{\2\*?\})' % (1, MAX_CHARS), re.S)
displaymath = re.compile(r'(\\displaystyle)(.{%i,%i}?)(\}(?:<|"))' % (1, MAX_CHARS))
outer_whitespace = re.compile(
    r'^\\,|\\,$|^~|~$|^\\ |\\ $|^\\thinspace|\\thinspace$|^\\!|\\!$|^\\:|\\:$|^\\;|\\;$|^\\enspace|\\enspace$|^\\quad|\\quad$|^\\qquad|\\qquad$|^\\hspace{[a-zA-Z0-9]+}|\\hspace{[a-zA-Z0-9]+}$|^\\hfill|\\hfill$')


def check_brackets(s):
    a = []
    surrounding = False
    for i, c in enumerate(s):
        if c == '{':
            if i > 0 and s[i-1] == '\\':  # not perfect
                continue
            else:
                a.append(1)
            if i == 0:
                surrounding = True
        elif c == '}':
            if i > 0 and s[i-1] == '\\':
                continue
            else:
                a.append(-1)
    b = np.cumsum(a)
    if len(b) > 1 and b[-1] != 0:
        raise ValueError(s)
    surrounding = s[-1] == '}' and surrounding
    if not surrounding:
        return s
    elif (b == 0).sum() == 1:
        return s[1:-1]
    else:
        return s


def clean_matches(matches, min_chars=MIN_CHARS):
    template = r'\\%s\s?\{(.*?)\}'
    sub = [re.compile(template % s) for s in ['ref', 'cite', 'label', 'caption']]
    faulty = []
    for i in range(len(matches)):
        if 'tikz' in matches[i]:  # do not support tikz at the moment
            faulty.append(i)
            continue
        for s in sub:
            matches[i] = re.sub(s, '', matches[i])
        matches[i] = matches[i].replace('\n', '').replace(r'\notag', '').replace(r'\nonumber', '')
        matches[i] = re.sub(outer_whitespace, '', matches[i])
        if len(matches[i]) < min_chars:
            faulty.append(i)
            continue
        # try:
        #     matches[i] = check_brackets(matches[i])
        # except ValueError:
        #     faulty.append(i)
        if matches[i][-1] == '\\' or 'newcommand' in matches[i][-1]:
            faulty.append(i)

    matches = [m.strip() for i, m in enumerate(matches) if i not in faulty]
    return list(set(matches))


def find_math(s, wiki=False):
    matches = []
    x = re.findall(inline, s)
    matches.extend([(g[1] if g[1] != '' else g[-1]) for g in x])
    if not wiki:
        patterns = [dollar, equation, align]
        groups = [1, 1, 0]
    else:
        patterns = [displaymath]
        groups = [1]
    for i, pattern in zip(groups, patterns):
        x = re.findall(pattern, s)
        matches.extend([g[i] for g in x])

    return clean_matches(matches)
