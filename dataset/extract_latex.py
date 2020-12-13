import re
import numpy as np

max_chars = 3000
dollar = re.compile(r'((?<!\$)\${1,2}(?!\$))(.{%i,%i}?)(?<!\\)(?<!\$)\1(?!\$)' % (1, max_chars))
inline = re.compile(r'(\\\((.*?)(?<!\\)\\\))|(\\\[(.{%i,%i}?)(?<!\\)\\\])' % (1, max_chars))
equation = re.compile(r'\\begin\{(equation|eqnarray|align|alignat|math|displaymath|gather)\*?\}(.{%i,%i}?)\\end\{\1\*?\}' % (1, max_chars), re.S)
align = re.compile(r'(\\begin\{(align|alignat|flalign|eqnarray)\*?\}(.{%i,%i}?)\\end\{\2\*?\})' % (1, max_chars), re.S)
displaymath = re.compile(r'(\\displaystyle)(.{%i,%i}?)(\}(?:<|"))' % (1, max_chars))
whitespace = re.compile(
    r'\s{2,}?|^\s|\s$|^\\,|\\,$|^~|~$|^\\ |\\ $|^\\thinspace|\\thinspace$|^\\!|\\!$|^\\:|\\:$|^\\;|\\;$|^\\enspace|\\enspace$|^\\quad|\\quad$|^\\qquad|\\qquad$|^\\hspace{[a-zA-Z0-9]+}|\\hspace{[a-zA-Z0-9]+}$|^\\hfill|\\hfill$')


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


def clean_matches(matches, min_chars=10):
    template = r'\\%s\{(.*?)\}'
    sub = [re.compile(template % s) for s in ['ref', 'label', 'caption']]
    faulty = []
    for i in range(len(matches)):
        for s in sub:
            matches[i] = re.sub(s, '', matches[i])
        matches[i] = matches[i].replace('\n', '')
        # brackets around _ and ^ vec sqrt
        # matches[i] = re.sub(r'(\^|\_|\\vec\ |\\sqrt\ )([a-zA-Z0-9\\])', r'\1{\2}', matches[i])
        matches[i] = re.sub(whitespace, '', matches[i])
        if len(matches[i]) < min_chars:
            faulty.append(i)
            continue
        try:
            matches[i] = check_brackets(matches[i])
        except ValueError:
            faulty.append(i)

    matches = [m for i, m in enumerate(matches) if i not in faulty]
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
