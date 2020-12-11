import re

dollar = re.compile(r'((?<!\$)\${1,2}(?!\$))(.*?)(?<!\\)(?<!\$)\1(?!\$)')
inline = re.compile(r'(\\\((.*?)(?<!\\)\\\))|(\\\[(.*?)(?<!\\)\\\])')
equation = re.compile(r'\\begin\{equation\*?\}(.*?)(?<!\\)\\end\{equation\*?\}', re.S)


def clean_matches(matches):
    template = r'\\%s\{(.*?)\}'
    sub = [re.compile(template % s) for s in ['ref', 'label', 'caption']]
    for i in range(len(matches)):
        for s in sub:
            matches[i] = re.sub(s, '', matches[i])
        matches[i] = matches[i].replace('\n', '').replace(r'&amp;', r'&').replace(r'&gt;', r'>').replace(r'&lt;', r'<')
        matches[i] = re.sub(r'(\^|\_|\\vec\ |\\sqrt\ )([a-zA-Z0-9])', r'\1{\2}', matches[i])
    return matches


def find_math(s):
    matches = []
    x = re.findall(dollar, s)
    matches.extend([g[1] for g in x])
    x = re.findall(inline, s)
    matches.extend([(g[1] if g[1] != '' else g[-1]) for g in x])
    x = re.findall(equation, s)
    matches.extend(x)
    return clean_matches(matches)
