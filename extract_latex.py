import re

max_chars = 3000
dollar = re.compile(r'((?<!\$)\${1,2}(?!\$))(.{%i,%i}?)(?<!\\)(?<!\$)\1(?!\$)' % (1, max_chars))
inline = re.compile(r'(\\\((.*?)(?<!\\)\\\))|(\\\[(.{%i,%i}?)(?<!\\)\\\])' % (1, max_chars))
equation = re.compile(r'\\begin\{(equation|eqnarray|align|alignat|math|displaymath|gather)\*?\}(.{%i,%i}?)(?<!\\)\\end\{\1\*?\}' % (1, max_chars), re.S)
align = re.compile(r'(\\begin\{(align|alignat|flalign|eqnarray)\*?\}(.{%i,%i}?)\\end\{\2\*?\})' % (1, max_chars), re.S)
displaymath = re.compile(r'(\\displaystyle)(.{%i,%i}?)(?<!\\)(\}(?:<|"))' % (1, max_chars))
whitespace = re.compile(
    r'\s{2,}?|^\s|\s$|^\\,|\\,$|^\\ |\\ $|^\\thinspace|\\thinspace$|^\\!|\\!$|^\\:|\\:$|^\\;|\\;$|^\\enspace|\\enspace$|^\\quad|\\quad$|^\\qquad|\\qquad$|^\\hspace{[a-zA-Z0-9]+}|\\hspace{[a-zA-Z0-9]+}$|^\\hfill|\\hfill$')


def clean_matches(matches, min_chars=10):
    template = r'\\%s\{(.*?)\}'
    sub = [re.compile(template % s) for s in ['ref', 'label', 'caption']]
    for i in range(len(matches)):
        for s in sub:
            matches[i] = re.sub(s, '', matches[i])
        matches[i] = matches[i].replace('\n', '')
        # brackets around _ and ^ vec sqrt
        # matches[i] = re.sub(r'(\^|\_|\\vec\ |\\sqrt\ )([a-zA-Z0-9]|\\(?:[a-zA-Z])+)', r'\1{\2}', matches[i]) # not complete yet. failure regression: x_\vec p
        matches[i] = re.sub(whitespace, '', matches[i])

    matches = [m for m in matches if len(m) >= min_chars]
    return list(set(matches))


def find_math(s, html=False):
    matches = []
    x = re.findall(inline, s)
    matches.extend([(g[1] if g[1] != '' else g[-1]) for g in x])
    patterns = [dollar, equation, align]
    groups = [1, 1, 0]
    if html:
        patterns.append(displaymath)
        groups.append(1)
    for i, pattern in zip(groups, patterns):
        x = re.findall(pattern, s)
        matches.extend([g[groups[i]] for g in x])

    return clean_matches(matches)
