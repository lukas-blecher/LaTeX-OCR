import requests
import os
import tqdm
import io

url = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/latest'


def get_latest_tag():
    r = requests.get(url)
    tag = r.url.split('/')[-1]
    if tag == 'releases':
        return 'v0.0.1'
    return tag


def download_as_bytes_with_progress(url: str, name: str = None) -> bytes:
    # source: https://stackoverflow.com/questions/71459213/requests-tqdm-to-a-variable
    resp = requests.get(url, stream=True, allow_redirects=True)
    total = int(resp.headers.get('content-length', 0))
    bio = io.BytesIO()
    if name is None:
        name = url
    with tqdm.tqdm(
        desc=name,
        total=total,
        unit='b',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            bar.update(len(chunk))
            bio.write(chunk)
    return bio.getvalue()


def download_checkpoints():
    tag = 'v0.0.1'  # get_latest_tag()
    path = os.path.dirname(__file__)
    print('download weights', tag, 'to path', path)
    weights = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/weights.pth' % tag
    resizer = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/image_resizer.pth' % tag
    for url, name in zip([weights, resizer], ['weights.pth', 'image_resizer.pth']):
        file = download_as_bytes_with_progress(url, name)
        open(os.path.join(path, name), "wb").write(file)


if __name__ == '__main__':
    download_checkpoints()
