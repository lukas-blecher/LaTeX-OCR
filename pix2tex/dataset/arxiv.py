# modified from https://github.com/soskek/arxiv_leaks

import argparse
import subprocess
import os
import glob
import re
import sys
import argparse
import logging
import tarfile
import tempfile
import logging
import requests
import urllib.request
from tqdm import tqdm
from urllib.error import HTTPError
from pix2tex.dataset.extract_latex import find_math
from pix2tex.dataset.scraping import recursive_search
from pix2tex.dataset.demacro import *

# logging.getLogger().setLevel(logging.INFO)
arxiv_id = re.compile(r'(?<!\d)(\d{4}\.\d{5})(?!\d)')
arxiv_base = 'https://export.arxiv.org/e-print/'


def get_all_arxiv_ids(text):
    '''returns all arxiv ids present in a string `text`'''
    ids = []
    for id in arxiv_id.findall(text):
        ids.append(id)
    return list(set(ids))


def download(url, dir_path='./'):
    idx = os.path.split(url)[-1]
    file_name = idx + '.tar.gz'
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        return file_path
    logging.info('\tdownload {}'.format(url) + '\n')
    try:
        r = urllib.request.urlretrieve(url, file_path)
        return r[0]
    except HTTPError:
        logging.info('Could not download %s' % url)
        return 0


def read_tex_files(file_path:str, demacro:bool=False)->str:
    """Read all tex files in the latex source at `file_path`. If it is not a `tar.gz` file try to read it as text file.

    Args:
        file_path (str): Path to latex source
        demacro (bool, optional): Deprecated. Call external `de-macro` program. Defaults to False.

    Returns:
        str: All Latex files concatenated into one string.
    """    
    tex = ''
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                tf = tarfile.open(file_path, 'r')
                tf.extractall(tempdir)
                tf.close()
                texfiles = [os.path.abspath(x) for x in glob.glob(os.path.join(tempdir, '**', '*.tex'), recursive=True)]
            except tarfile.ReadError as e:
                texfiles = [file_path]  # [os.path.join(tempdir, file_path+'.tex')]
            if demacro:
                ret = subprocess.run(['de-macro', *texfiles], cwd=tempdir, capture_output=True)
                if ret.returncode == 0:
                    texfiles = glob.glob(os.path.join(tempdir, '**', '*-clean.tex'), recursive=True)
            for texfile in texfiles:
                try:
                    ct = open(texfile, 'r', encoding='utf-8').read()
                    tex += ct
                except UnicodeDecodeError as e:
                    logging.debug(e)
                    pass
    except Exception as e:
        logging.debug('Could not read %s: %s' % (file_path, str(e)))
        raise e
    tex = pydemacro(tex)
    return tex


def download_paper(arxiv_id, dir_path='./'):
    url = arxiv_base + arxiv_id
    return download(url, dir_path)


def read_paper(targz_path, delete=False, demacro=False):
    paper = ''
    if targz_path != 0:
        paper = read_tex_files(targz_path, demacro=demacro)
        if delete:
            os.remove(targz_path)
    return paper


def parse_arxiv(id, save=None, demacro=True):
    if save is None:
        dir = tempfile.gettempdir()
    else:
        dir = save
    text = read_paper(download_paper(id, dir), delete=save is None, demacro=demacro)

    return find_math(text, wiki=False), []


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description='Extract math from arxiv')
    parser.add_argument('-m', '--mode', default='top100', choices=['top', 'ids', 'dirs'],
                        help='Where to extract code from. top: current 100 arxiv papers (-m top int for any other number of papers), id: specific arxiv ids. \
                              Usage: `python arxiv.py -m ids id001 [id002 ...]`, dirs: a folder full of .tar.gz files. Usage: `python arxiv.py -m dirs directory [dir2 ...]`')
    parser.add_argument(nargs='*', dest='args', default=[])
    parser.add_argument('-o', '--out', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), help='output directory')
    parser.add_argument('-d', '--demacro', dest='demacro', action='store_true',
                        help='Deprecated - Use de-macro (Slows down extraction, may but improves quality). Install https://www.ctan.org/pkg/de-macro')
    parser.add_argument('-s', '--save', default=None, type=str, help='When downloading files from arxiv. Where to save the .tar.gz files. Default: Only temporary')
    args = parser.parse_args()
    if '.' in args.out:
        args.out = os.path.dirname(args.out)
    skips = os.path.join(args.out, 'visited_arxiv.txt')
    if os.path.exists(skips):
        skip = open(skips, 'r', encoding='utf-8').read().split('\n')
    else:
        skip = []
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    try:
        if args.mode == 'ids':
            visited, math = recursive_search(parse_arxiv, args.args, skip=skip, unit='paper', save=args.save, demacro=args.demacro)
        elif args.mode == 'top':
            num = 100 if len(args.args) == 0 else int(args.args[0])
            url = 'https://arxiv.org/list/physics/pastweek?skip=0&show=%i' % num  # 'https://arxiv.org/list/hep-th/2203?skip=0&show=100'
            ids = get_all_arxiv_ids(requests.get(url).text)
            math, visited = [], ids
            for id in tqdm(ids):
                try:
                    m, _ = parse_arxiv(id, save=args.save, demacro=args.demacro)
                    math.extend(m)
                except ValueError:
                    pass
        elif args.mode == 'dirs':
            files = []
            for folder in args.args:
                files.extend([os.path.join(folder, p) for p in os.listdir(folder)])
            math, visited = [], []
            for f in tqdm(files):
                try:
                    text = read_paper(f, delete=False, demacro=args.demacro)
                    math.extend(find_math(text, wiki=False))
                    visited.append(os.path.basename(f))
                except DemacroError as e:
                    logging.debug(f + str(e))
                    pass
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.debug(e)
                    raise e
        else:
            raise NotImplementedError
    except KeyboardInterrupt:
        pass
    print('Found %i instances of math latex code' % len(math))
    # print('\n'.join(math))
    # sys.exit(0)
    for l, name in zip([visited, math], ['visited_arxiv.txt', 'math_arxiv.txt']):
        f = os.path.join(args.out, name)
        if not os.path.exists(f):
            open(f, 'w').write('')
        f = open(f, 'a', encoding='utf-8')
        for element in l:
            f.write(element)
            f.write('\n')
        f.close()
