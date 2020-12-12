import html
import requests
import re
import tempfile
from dataset.arxiv import *
from dataset.extract_latex import *


wikilinks = re.compile(r'href="/wiki/(.*?)"')
wiki_base = 'https://en.wikipedia.org/wiki/'


def parse_url(url):
    r = requests.get(url)
    if r.ok:
        return html.unescape(r.text)


def parse_wiki(url):
    text = parse_url(url)  # ['https://en.wikipedia.org/wiki/'+l for l in ]
    linked = list(set([l for l in re.findall(wikilinks, text) if not ':' in l]))
    return find_math(text, html=True), linked


# recursive wiki search
def recursive_wiki(start, depth=1, skip=[]):
    '''Recursivley search wikipedia for math. Every link on the starting page `start` will be visited in the next round and so on, until there is no 
    math in the child page anymore. This will be repeated `depth` times. Be careful approximatley `depth=3` reaches the entirety of Wikipedia'''
    start = start.split('/')[-1]
    visited, links = set(skip), [start]
    math = []
    try:
        for i in range(int(depth)):
            for link in links:
                if not link in visited:
                    print('%i searching %s' % (len(visited), wiki_base+link))
                    m, l = parse_wiki(wiki_base+link)
                    # check if we got any math from this wiki page and
                    # if not terminate the tree
                    if len(m) > 0:
                        links.extend(l)
                        math.extend(m)
                    visited.add(link)
        return list(visited), list(set(math))
    except:
        return list(visited), list(set(math))
