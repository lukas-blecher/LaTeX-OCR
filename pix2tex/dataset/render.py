
from pix2tex.dataset.latex2png import Latex, tex2pil
import argparse
import sys
import os
import glob
import shutil
from tqdm.auto import tqdm
import cv2
import numpy as np
from PIL import Image
import subprocess


def get_installed_fonts(tex_path: str):
    cmd = "find %s -name *Math*.otf" % tex_path
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True,
                               shell=True
                               )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(stderr)
    fonts = [_.split(os.sep)[-1] for _ in stdout.split('\n')][:-1]
    fonts.extend(["Latin Modern Math"]*len(fonts))
    return fonts


def render_dataset(dataset: np.ndarray, unrendered: np.ndarray, args) -> np.ndarray:
    """Renders a list of tex equations

    Args:
        dataset (numpy.ndarray): List of equations
        unrendered (numpy.ndarray): List of integers of size `dataset` that give the name of the saved image
        args (Union[Namespace, Munch]): additional arguments: mode (equation or inline), out (output directory), divable (common factor )
                                        batchsize (how many samples to render at once), dpi, font (Math font), preprocess (crop, alpha off)
                                        shuffle (bool)

    Returns:
        numpy.ndarray: equation indices that could not be rendered
    """
    assert len(unrendered) == len(dataset), 'unrendered and dataset must be of equal size'
    math_mode = '$$'if args.mode == 'equation' else '$'
    os.makedirs(args.out, exist_ok=True)
    # remove successfully rendered equations
    rendered = np.array([int(os.path.basename(img).split('.')[0])
                         for img in glob.glob(os.path.join(args.out, '*.png'))])
    valid = [i for i, j in enumerate(unrendered) if j not in rendered]
    # update unrendered and dataset
    dataset = dataset[valid]
    unrendered = unrendered[valid]
    order = np.random.permutation(len(dataset)) if args.shuffle else np.arange(len(dataset))
    faulty = []
    for batch_offset in tqdm(range(0, len(dataset), args.batchsize), desc="global batch index"):
        batch = dataset[order[batch_offset:batch_offset+args.batchsize]]
        #batch = [x for j, x in enumerate(batch) if order[i+j] not in indices]
        if len(batch) == 0:
            continue
        valid_math = np.asarray([[i, "%s %s %s" % (math_mode, x, math_mode)] for i, x in enumerate(
            batch) if x != ''], dtype=object)  # space used to prevent escape $
        #print('\n', i, len(math), '\n'.join(math))
        font = font = np.random.choice(args.font) if len(
            args.font) > 1 else args.font[0]
        dpi = np.random.choice(np.arange(min(args.dpi), max(args.dpi))) if len(
            args.dpi) > 1 else args.dpi[0]
        if len(valid_math) > 0:
            valid_idx, math = valid_math.T
            valid_idx = valid_idx.astype(np.int32)
            try:
                if args.preprocess:
                    pngs, error_index = tex2pil(
                        math, dpi=dpi, font=font, return_error_index=True)
                else:
                    pngs, error_index = Latex(math, dpi=dpi, font=font).write(
                        return_bytes=False)
                # error_index not count "" line, use valid_idx transfer to real index matching in batch index
                local_error_index = valid_idx[error_index]
                # tranfer in batch index to global batch index
                global_error_index = [
                    batch_offset+_ for _ in local_error_index]
                faulty.extend(list(unrendered[order[global_error_index]]))
            except Exception as e:
                print("\n%s" % e, end='')
                faulty.extend(
                    list(unrendered[order[batch_offset:batch_offset+args.batchsize]]))
                continue

            for inbatch_idx, order_idx in enumerate(range(batch_offset, batch_offset+args.batchsize)):
                # exclude render failed equations and blank line
                if inbatch_idx in local_error_index or inbatch_idx not in valid_idx:
                    continue
                outpath = os.path.join(args.out, '%07d.png' % unrendered[order[order_idx]])
                png_idx = np.where(valid_idx == inbatch_idx)[0][0]
                if args.preprocess:
                    try:
                        data = np.asarray(pngs[png_idx])
                        # print(data.shape)
                        # To invert the text to white
                        gray = 255*(data[..., 0] < 128).astype(np.uint8)
                        white_pixels = np.sum(gray == 255)
                        # some png will be whole white, because some equation's syntax is wrong
                        # eg.$$ \mathit { \Iota \Kappa \Lambda \Mu \Nu \Xi \Omicron \Pi } $$
                        # extract from wikipedia english dump file https://dumps.wikimedia.org/enwiki/latest/
                        white_percentage = (white_pixels / (gray.shape[0] * gray.shape[1]))
                        if white_percentage == 0:
                            continue
                        # Find all non-zero points (text)
                        coords = cv2.findNonZero(gray)
                        # Find minimum spanning bounding box
                        a, b, w, h = cv2.boundingRect(coords)
                        rect = data[b:b+h, a:a+w]
                        im = Image.fromarray((255-rect[..., -1]).astype(np.uint8)).convert('L')
                        dims = []
                        for x in [w, h]:
                            div, mod = divmod(x, args.divable)
                            dims.append(args.divable*(div + (1 if mod > 0 else 0)))
                        padded = Image.new('L', dims, 255)
                        padded.paste(im, (0, 0, im.size[0], im.size[1]))
                        padded.save(outpath)
                    except Exception as e:
                        print(e)
                        pass
                else:
                    shutil.move(pngs[png_idx], outpath)
    # prevent repeat between two error_index and imagemagic error
    faulty = list(set(faulty))
    faulty.sort()
    return np.array(faulty)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render dataset')
    parser.add_argument('-i', '--data', type=str,
                        required=True, help='file of list of latex code')
    parser.add_argument('-o', '--out', type=str,
                        required=True, help='output directory')
    parser.add_argument('-b', '--batchsize', type=int, default=100,
                        help='How many equations to render at once')
    parser.add_argument('-f', '--font', nargs='+', type=str,
                        default="", help='font to use.')
    parser.add_argument('-fp', '--fonts_path', type=str,
                        default="/usr/local/texlive/", help='installed font path')
    parser.add_argument('-m', '--mode', choices=[
                        'inline', 'equation'], default='equation', help='render as inline or equation')
    parser.add_argument('--dpi', type=int, default=[110, 170], nargs='+', help='dpi range to render in')
    parser.add_argument('-p', '--no-preprocess', dest='preprocess', default=True,
                        action='store_false', help='crop, remove alpha channel, padding')
    parser.add_argument('-d', '--divable', type=int, default=32,
                        help='To what factor to pad the images')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='Whether to shuffle the equations in the first iteration')
    args = parser.parse_args(sys.argv[1:])
    args.font = args.font if args.font != "" else get_installed_fonts(
        args.fonts_path)
    print(args.font)
    dataset = np.array(open(args.data, 'r').read().split('\n'), dtype=object)
    unrendered = np.arange(len(dataset))
    failed = np.array([])
    while unrendered.tolist() != failed.tolist():
        failed = unrendered
        unrendered = render_dataset(dataset[unrendered], unrendered, args)
        if len(unrendered) < 50*args.batchsize:
            args.batchsize = max([1, args.batchsize//2])
        args.shuffle = True
