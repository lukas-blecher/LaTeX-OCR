import os
import sys
import random
from tqdm import tqdm
import html
import requests
import re
import argparse
import logging
from typing import Callable, List, Tuple
from pix2tex.dataset.extract_latex import find_math

htmltags = re.compile(r'<(noscript|script)>.*?<\/\1>', re.S)
wikilinks = re.compile(r'href="/wiki/(.*?)"')
wiki_base = 'https://en.wikipedia.org/wiki/'
stackexchangelinks = re.compile(r'(?:(https:\/\/\w+)\.stack\w+\.com|)\/questions\/(\d+\/[\w\d\/-]+)')
math_stack_exchange_base = 'https://math.stackexchange.com/questions/'
physics_stack_exchange_base = 'https://physics.stackexchange.com/questions/'

# recursive search


def recursive_search(parser: Callable,  seeds: List[str], depth: int = 2, skip: List[str] = [], unit: str = 'links', base_url: str = None, **kwargs) -> Tuple[List[str], List[str]]:
    """Find math recursively. Look in `seeds` for math and further sites to look.

    Args:
        parser (Callable): A function that returns a `Tuple[List[str], List[str]]` of math and ids (for `base_url`) respectively.
        seeds (List[str]): Fist set of ids.
        depth (int, optional): How many iterations to look for. Defaults to 2.
        skip (List[str], optional): List of alreadly visited ids. Defaults to [].
        unit (str, optional): Tqdm verbose unit description. Defaults to 'links'.
        base_url (str, optional): Base url to add ids to. Defaults to None.

    Returns:
        Tuple[List[str],List[str]]: Returns list of found math and visited ids respectively.
    """
    visited, links = set(skip), set(seeds)
    math = []
    try:
        for i in range(int(depth)):
            link_list = list(links)
            random.shuffle(link_list)
            t_bar = tqdm(link_list, initial=len(visited), unit=unit)
            for link in t_bar:
                if not link in visited:
                    t_bar.set_description('searching %s' % (link[:15]))
                    if base_url:
                        m, l = parser(base_url+link, **kwargs)
                    else:
                        m, l = parser(link, **kwargs)
                    # check if we got any math from this wiki page and
                    # if not terminate the tree
                    if len(m) > 0:
                        for li in l:
                            links.add(li)
                        # t_bar.total = len(links)
                        math.extend(m)
                    visited.add(link)
        return list(visited), list(set(math))
    except Exception as e:
        logging.debug(e)
        return list(visited), list(set(math))
    except KeyboardInterrupt:
        return list(visited), list(set(math))


def parse_url(url, encoding=None):
    r = requests.get(url)
    if r.ok:
        if encoding:
            r.encoding = encoding
        return html.unescape(re.sub(htmltags, '', r.text))
    return ''


def parse_wiki(url):
    text = parse_url(url)
    linked = list(set([l for l in re.findall(wikilinks, text) if not ':' in l]))
    return find_math(text, wiki=True), linked


def parse_stack_exchange(url):
    text = parse_url(url)
    linked = list(set([l[1] for l in re.findall(stackexchangelinks, text) if url.startswith(l[0])]))
    return find_math(text, wiki=False), linked

# recursive wiki search


def recursive_wiki(seeds, depth=4, skip=[], base_url=wiki_base):
    '''Recursivley search wikipedia for math. Every link on the starting page `start` will be visited in the next round and so on, until there is no 
    math in the child page anymore. This will be repeated `depth` times.'''
    start = [s.split('/')[-1] for s in seeds]
    return recursive_search(parse_wiki, start, depth, skip, base_url=base_url, unit=' links')


def recursive_stack_exchange(seeds, depth=4, skip=[], base_url=math_stack_exchange_base):
    '''Recursively search through stack exchange questions'''
    start = [s.partition(base_url.split('//')[-1])[-1] for s in seeds]
    return recursive_search(parse_stack_exchange, start, depth, skip, base_url=base_url, unit=' questions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract math from websites')
    parser.add_argument('-m', '--mode', default='auto', choices=['auto', 'wiki', 'math_stack', 'physics_stack'],
                        help='What website to scrape. Choices: `auto` determine by input, `wiki` wikipedia, \
                        `math_stack` math.stackexchange, `physics_stack` physics.stackexchange.')
    parser.add_argument(nargs='*', dest='url', default=['https://en.wikipedia.org/wiki/Mathematics', 'https://en.wikipedia.org/wiki/Physics'],
                        help='starting url(s). Default: Mathematics, Physics wiki pages')
    parser.add_argument('-o', '--out', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), help='output directory')
    args = parser.parse_args()
    if '.' in args.out:
        args.out = os.path.dirname(args.out)
    # determine website
    if args.mode == 'auto':
        if len(args.url) == 0:
            raise ValueError('Provide an starting url')
        url = args.url[0]
        if re.search(wikilinks, url) is not None:
            args.mode = 'wiki'
        elif re.search(stackexchangelinks, url) is not None:
            if 'math' in url:
                args.mode = 'math_stack'
            elif 'physics' in url:
                args.mode = 'physics_stack'
        else:
            raise NotImplementedError('The website was not recognized')
    skips = os.path.join(args.out, f'visited_{args.mode}.txt')
    if os.path.exists(skips):
        skip = open(skips, 'r', encoding='utf-8').read().split('\n')
    else:
        skip = []
    try:
        if args.mode == 'physics_stack':
            visited, math = recursive_stack_exchange(args.url, base_url=physics_stack_exchange_base)
        elif args.mode == 'math_stack':
            visited, math = recursive_stack_exchange(args.url, base_url=math_stack_exchange_base)
        elif args.mode == 'wiki':
            visited, math = recursive_wiki(args.url)
    except KeyboardInterrupt:
        pass
    print('Found %i instances of math latex code. Save to %s' % (len(math), args.out))
    for l, name in zip([visited, math], [f'visited_{args.mode}.txt', f'math_{args.mode}.txt']):
        f = os.path.join(args.out, name)
        if not os.path.exists(f):
            open(f, 'w').write('')
        f = open(f, 'a', encoding='utf-8')
        for element in l:
            f.write(element)
            f.write('\n')
        f.close()
