import requests
import os

from torch.hub import download_url_to_file, get_dir

url = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/latest'


def get_latest_tag():
    r = requests.get(url)
    tag = r.url.split('/')[-1]
    if tag == 'releases':
        return 'v0.0.1'
    return tag


def download_checkpoints():
    checkpoints_dir = os.path.join(get_dir(), 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    tag = 'v0.0.1'  # get_latest_tag()
    weights = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/weights.pth' % tag
    resizer = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/image_resizer.pth' % tag
    cached_files = []
    for url in [weights, resizer]:
        cached_file = os.path.join(checkpoints_dir, os.path.split(url)[-1])
        if not os.path.exists(cached_file):
            download_url_to_file(url, cached_file)
        cached_files += [cached_file]
    return cached_files

if __name__ == '__main__':
    download_checkpoints()
