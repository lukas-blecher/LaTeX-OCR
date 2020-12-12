# taken from https://github.com/soskek/arxiv_leaks

import argparse
import json
import os
import re
import sys
import tarfile
import logging
import requests


arxiv_id_pt = re.compile(r'(?<!\d)(\d{4}\.\d{5})(?!\d)')
url_base = 'https://arxiv.org/e-logging.info/'


def get_all_arxiv_ids(text):
    '''returns all arxiv ids present in a string `text`'''
    ids = []
    for arxiv_id in arxiv_id_pt.findall(text):
        ids.append(arxiv_id)
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


def read_papers(arxiv_ids, dir_path='./'):
    results = {}
    for arxiv_id in arxiv_ids:
        logging.info('[{}]'.format(arxiv_id) + '\n')
        result = read_paper(arxiv_id, dir_path)
        if result:
            if 'title' in result:
                logging.info('\t({})'.format(result['title']) + '\n')
                logging.info('\t {}'.format(' / '.join(result['authors'])) + '\n')
            results[arxiv_id] = result
    return results


def read_paper(arxiv_id, dir_path='./'):
    url = url_base + arxiv_id
    targz_path = download(url, dir_path)
    if not targz_path:
        return []
    else:
        return read_tex_files(targz_path)


def read_tex_files(file_path):
    try:
        with tarfile.open(file_path, 'r') as tf:
            for ti in tf:
                if ti.name.endswith('.tex'):
                    with tf.extractfile(ti) as f:
                        tex = f.read().decode('utf-8')
    except tarfile.ReadError:
        try:
            tex = open(file_path, 'r').read().decode('utf-8')
        except:
            logging.info('could not read %s' % file_path)
            pass

    return tex
