# taken from https://github.com/soskek/arxiv_leaks

import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
import logging
import requests
sys.path.insert(0, os.path.abspath('..' if os.path.dirname(sys.argv[0])=='' else '.'))
from dataset.extract_latex import *
from dataset.scraping import *

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

    r = requests.get(url)
    logging.info('\tdownload {}'.format(url) + '\n')
    if r.ok:
        with open(file_path, 'wb') as f:
            f.write(r.content)
        return file_path
    else:
        return 0


def read_tex_files(file_path):
    tex = ''
    try:
        with tarfile.open(file_path, 'r') as tf:
            for ti in tf:
                if ti.name.endswith('.tex'):
                    with tf.extractfile(ti) as f:
                        tex += f.read().decode('utf-8')
    except tarfile.ReadError:
        try:
            tex += open(file_path, 'r').read().decode('utf-8')
        except:
            logging.info('could not read %s' % file_path)
            pass

    return tex


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
    linked = list(set([l for l in re.findall(arxiv_id, text)]))
    # remove comments
    text = re.sub('r(?<!\\)%.+', '', text)
    return find_math(text, wiki=False), linked


if __name__ == '__main__':
    skips = os.path.join(sys.path[0], 'dataset', 'data', 'visited_arxiv.txt')
    skip = open(skips, 'r', encoding='utf-8').read().split('\n')
    if len(sys.argv) > 2:
        url = sys.argv[1]
        visited, math = recursive_search([parse_arxiv], url, skip=skip, unit='papers')

    else:
        url = 'https://arxiv.org/list/math/2012?skip=0&show=100'
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
