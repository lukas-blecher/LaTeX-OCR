import argparse
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input file')
    parser.add_argument('-o', '--output', default=None, help='output file')
    args = parser.parse_args()

    d = open(args.input, 'r').read().split('\n')
    reqs = ['\\', '_', '^', '(', ')', '{', '}']
    deleted = 0
    for i in tqdm(reversed(range(len(d))), total=len(d)):
        if not any([r in d[i] for r in reqs]):
            del d[i]
            deleted += 1
    print('removed %i lines' % deleted)
    f = args.output
    if f is None:
        f = args.input
    open(f, 'w').write('\n'.join(d))
