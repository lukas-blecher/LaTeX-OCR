try:
    from dataset.latex2png import *
except ModuleNotFoundError:
    from latex2png import *
import argparse
import sys
import os
import glob
import shutil
from tqdm.auto import tqdm
import cv2
import numpy as np


def render_dataset(dataset, args):
    '''Renders a list of tex equations

    Args:
        dataset (str): Path to file with tex lines
        args (Namespace or Munch): additional arguments: mode (equation or inline), out (output directory), divable (common factor )
                                   batchsize (how many samples to render at once), dpi, font (Math font), preprocess (crop, alpha off)

    Returns:
        list: equation indices that could not be rendered. 
    '''
    math_mode = '$$'if args.mode == 'equation' else '$'
    os.makedirs(args.out, exist_ok=True)
    indices = [int(os.path.basename(img).split('.')[0]) for img in glob.glob(os.path.join(args.out, '*.png'))]

    def skip(ind):
        return [i for i in range(ind, ind+args.batchsize) if i in indices]
    faulty = []
    for i in tqdm(range(0, len(dataset), args.batchsize)):
        batch = dataset[i:i+args.batchsize]
        batch = [x for j, x in enumerate(batch) if j not in skip(i)]
        if len(batch) == 0:
            continue
        math = [math_mode+x+math_mode for x in batch if x != '']
        #print('\n', i, len(math), '\n'.join(math))
        if len(args.font) > 1:
            font = np.random.choice(args.font)
        else:
            font = args.font[0]
        if len(math) > 0:
            try:
                if args.preprocess:
                    pngs = tex2pil(math, dpi=args.dpi, font=font)
                else:
                    pngs = Latex(math, dpi=args.dpi, font=font).write(return_bytes=False)
            except Exception as e:
                print(e)
                faulty.append([i, i+len(dataset)])
                continue

            for j, k in enumerate(range(i, i+len(pngs))):
                outpath = os.path.join(args.out, '%06d.png' % k)
                if args.preprocess:
                    try:
                        data = np.asarray(pngs[j])
                        # print(data.shape)
                        gray = 255*(data[..., 0] < 128).astype(np.uint8)  # To invert the text to white
                        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
                        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
                        rect = data[b:b+h, a:a+w]
                        im = Image.fromarray((255-rect[..., -1]).astype(np.uint8)).convert('L')
                        dims = []
                        for x in [w, h]:
                            div, mod = divmod(x, args.divable)
                            dims.append(args.divable*(div + (1 if mod > 0 else 0)))
                        padded = Image.new('L', dims, 255)
                        padded.paste(im, im.getbbox())
                        padded.save(outpath)
                    except Exception as e:
                        pass
                else:
                    shutil.move(pngs[j], outpath)

    return faulty


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render dataset')
    parser.add_argument('-i', '--data', type=str, required=True, help='file of list of latex code')
    parser.add_argument('-o', '--out', type=str, required=True, help='output directory')
    parser.add_argument('-b', '--batchsize', type=int, default=100, help='How many equations to render at once')
    parser.add_argument('-f', '--font', nargs='+', type=str, default=['Latin Modern Math', 'GFSNeohellenicMath.otf', 'Asana Math', 'XITS Math',
                                                                      'Cambria Math', 'Latin Modern Math', 'Latin Modern Math', 'Latin Modern Math'], help='font to use. default = Latin Modern Math')
    parser.add_argument('-m', '--mode', choices=['inline', 'equation'], default='equation', help='render as inline or equation')
    parser.add_argument('--dpi', type=int, default=180, help='dpi to render in')
    parser.add_argument('-p', '--preprocess', default=True, action='store_false', help='crop, remove alpha channel, padding')
    parser.add_argument('-d', '--divable', type=int, default=32, help='To what factor to pad the images')
    args = parser.parse_args(sys.argv[1:])

    dataset = open(args.data, 'r').read().split('\n')
    render_dataset(dataset, args)
