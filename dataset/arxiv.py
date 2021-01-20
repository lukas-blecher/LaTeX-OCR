# modified from https://github.com/soskek/arxiv_leaks

import argparse
import json
import os
import glob
import re
import sys
import subprocess
import tarfile
import tempfile
import chardet
import logging
import requests
import urllib.request
from urllib.error import HTTPError
from extract_latex import *
from scraping import *

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


def read_tex_files(file_path):
    tex = ''
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            tf = tarfile.open(file_path, 'r')
            tf.extractall(tempdir)
            tf.close()
            texfiles = [os.path.abspath(x) for x in glob.glob(os.path.join(tempdir, '**', '*.tex'), recursive=True)]
            # de-macro
            ret = subprocess.run(['de-macro', *texfiles], cwd=tempdir, capture_output=True)
            if ret.returncode == 0:
                texfiles = glob.glob(os.path.join(tempdir, '**', '*-clean.tex'), recursive=True)
            for texfile in texfiles:
                try:
                    tex += open(texfile, 'r', encoding=chardet.detect(open(texfile, 'br').readline())['encoding']).read()
                except UnicodeDecodeError:
                    pass

    except tarfile.ReadError:
        try:
            tex += open(file_path, 'r', encoding=chardet.detect(open(file_path, 'br').readline())['encoding']).read()
        except Exception as e:
            logging.info('Could not read %s: %s' % (file_path, str(e)))
            pass
    # remove comments
    return re.sub(r'(?<!\\)%.*\n', '', tex)


def read_paper(arxiv_id, dir_path='./'):
    url = arxiv_base + arxiv_id
    targz_path = download(url, dir_path)
    paper = ''
    if targz_path != 0:
        paper = read_tex_files(targz_path)
        os.remove(targz_path)
    return paper


def parse_arxiv(id):
    tempdir = tempfile.gettempdir()
    text = read_paper(id, tempdir)
    #print(text, file=open('paper.tex', 'w'))
    #linked = list(set([l for l in re.findall(arxiv_id, text)]))

    return find_math(text, wiki=False), []


if __name__ == '__main__':
    skips = os.path.join(sys.path[0], 'dataset', 'data', 'visited_arxiv.txt')
    if os.path.exists(skips):
        skip = open(skips, 'r', encoding='utf-8').read().split('\n')
    else:
        skip = []
    if len(sys.argv) > 1:
        arxiv_ids = sys.argv[1:]
        visited, math = recursive_search(parse_arxiv, arxiv_ids, skip=skip, unit='paper')

    else:
        url = 'https://arxiv.org/list/hep-th/2012?skip=0&show=100'  # https://arxiv.org/list/hep-th/2012?skip=0&show=100
        ids = get_all_arxiv_ids(requests.get(url).text)
        math, visited = [], ids
        for id in tqdm(ids):
            m, _ = parse_arxiv(id)
            math.extend(m)

    for l, name in zip([visited, math], ['visited_arxiv.txt', 'math_arxiv.txt']):
        f = open(os.path.join(sys.path[0], 'dataset', 'data', name), 'a', encoding='utf-8')
        for element in l:
            f.write(element)
            f.write('\n')
        f.close()
