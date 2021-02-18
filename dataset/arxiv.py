# modified from https://github.com/soskek/arxiv_leaks

import argparse
import json
import os
import glob
import re
import sys
import argparse
import logging
import shutil
import subprocess
import tarfile
import tempfile
import chardet
import logging
import requests
import urllib.request
from urllib.error import HTTPError
try:
    from extract_latex import *
    from scraping import *
    from demacro import *
except:
    from dataset.extract_latex import *
    from dataset.scraping import *
    from dataset.demacro import *

# logging.getLogger().setLevel(logging.INFO)
arxiv_id = re.compile(r'(?<!\d)(\d{4}\.\d{5})(?!\d)')
arxiv_base = 'https://arxiv.org/e-print/'


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


def read_tex_files(file_path, demacro=True):
    tex = ''
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                tf = tarfile.open(file_path, 'r')
                tf.extractall(tempdir)
                tf.close()
                texfiles = [os.path.abspath(x) for x in glob.glob(os.path.join(tempdir, '**', '*.tex'), recursive=True)]
                # de-macro
                if demacro:
                    ret = subprocess.run(['de-macro', *texfiles], cwd=tempdir, capture_output=True)
                    if ret.returncode == 0:
                        texfiles = glob.glob(os.path.join(tempdir, '**', '*-clean.tex'), recursive=True)
            except tarfile.ReadError as e:
                texfiles = [file_path]  # [os.path.join(tempdir, file_path+'.tex')]
                #shutil.move(file_path, texfiles[0])

            for texfile in texfiles:
                try:
                    tex += open(texfile, 'r', encoding=chardet.detect(open(texfile, 'br').readline())['encoding']).read()                
                except UnicodeDecodeError:
                    pass
            tex = unfold(convert(tex))
    except Exception as e:
        logging.debug('Could not read %s: %s' % (file_path, str(e)))
        pass
    # remove comments
    return re.sub(r'(?<!\\)%.*\n', '', tex)


def download_paper(arxiv_id, dir_path='./'):
    url = arxiv_base + arxiv_id
    return download(url, dir_path)


def read_paper(targz_path, delete=True, demacro=True):
    paper = ''
    if targz_path != 0:
        paper = read_tex_files(targz_path, demacro)
        if delete:
            os.remove(targz_path)
    return paper


def parse_arxiv(id, demacro=True):
    tempdir = tempfile.gettempdir()
    text = read_paper(download_paper(id, tempdir), demacro=demacro)
    #print(text, file=open('paper.tex', 'w'))
    #linked = list(set([l for l in re.findall(arxiv_id, text)]))

    return find_math(text, wiki=False), []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract math from arxiv')
    parser.add_argument('-m', '--mode', default='top100', choices=['top100', 'id', 'dir'],
                        help='Where to extract code from. top100: current 100 arxiv papers, id: specific arxiv ids. \
                              Usage: `python arxiv.py -m id id001 id002`, dir: a folder full of .tar.gz files. Usage: `python arxiv.py -m dir directory`')
    parser.add_argument(nargs='+', dest='args', default=[])
    parser.add_argument('-o', '--out', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), help='output directory')
    parser.add_argument('-d', '--no-demacro', dest='demacro', action='store_false', help='Use de-macro (Slows down extraction but improves quality)')
    args = parser.parse_args()
    if '.' in args.out:
        args.out = os.path.dirname(args.out)
    skips = os.path.join(args.out, 'visited_arxiv.txt')
    if os.path.exists(skips):
        skip = open(skips, 'r', encoding='utf-8').read().split('\n')
    else:
        skip = []
    if args.mode == 'ids':
        visited, math = recursive_search(parse_arxiv, args.args, skip=skip, unit='paper')
    elif args.mode == 'top100':
        url = 'https://arxiv.org/list/hep-th/2012?skip=0&show=100'  # https://arxiv.org/list/hep-th/2012?skip=0&show=100
        ids = get_all_arxiv_ids(requests.get(url).text)
        math, visited = [], ids
        for id in tqdm(ids):
            m, _ = parse_arxiv(id)
            math.extend(m)
    elif args.mode == 'dir':
        dirs = os.listdir(args.args[0])
        math, visited = [], []
        for f in tqdm(dirs):
            try:
                text = read_paper(os.path.join(args.args[0], f), False, args.demacro)
                math.extend(find_math(text, wiki=False))
                visited.append(os.path.basename(f))
            except Exception as e:
                logging.debug(e)
                pass
    else:
        raise NotImplementedError

    for l, name in zip([visited, math], ['visited_arxiv.txt', 'math_arxiv.txt']):
        f = os.path.join(args.out, name)
        if not os.path.exists(f):
            open(f, 'w').write('')
        f = open(f, 'a', encoding='utf-8')
        for element in l:
            f.write(element)
            f.write('\n')
        f.close()
