import requests
import os

url = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/latest'


def get_latest_tag():
    r = requests.get(url)
    tag = r.url.split('/')[-1]
    if tag == 'releases':
        return 'v0.0.1'
    return tag


def download_checkpoints():
    tag = get_latest_tag()
    path = os.path.dirname(__file__)
    print('download weights', tag, 'to path', path)
    weights = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/weights.pth' % tag
    resizer = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/image_resizer.pth' % tag
    for url, name in zip([weights, resizer], ['weights.pth', 'resizer.pth']):
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(path, name), "wb").write(r.content)


if __name__ == '__main__':
    download_checkpoints()
