# taken and modified from https://github.com/harvardnlp/im2markup
# tokenize latex formulas
import sys
import os
import re
import argparse
import logging
import subprocess
import shutil


def process_args(args):
    parser = argparse.ArgumentParser(description='Preprocess (tokenize or normalize) latex formulas')

    parser.add_argument('--mode', '-m', dest='mode',
                        choices=['tokenize', 'normalize'], default='normalize',
                        help=('Tokenize (split to tokens seperated by space) or normalize (further translate to an equivalent standard form).'
                              ))
    parser.add_argument('--input-file', '-i', dest='input_file',
                        type=str, required=True,
                        help=('Input file containing latex formulas. One formula per line.'
                              ))
    parser.add_argument('--output-file', '-o', dest='output_file',
                        type=str, required=True,
                        help=('Output file.'
                              ))
    parser.add_argument('-n', '--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default=None,
                        help=('Log file path, default=log.txt'))
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)

    input_file = parameters.input_file
    output_file = parameters.output_file

    assert os.path.exists(input_file), input_file
    shutil.copy(input_file, output_file)
    operators = '\s?'.join('|'.join(['arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot', 'coth', 'csc', 'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf',
                                     'injlim', 'ker', 'lg', 'lim', 'liminf', 'limsup', 'ln', 'log', 'max', 'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh']))
    ops = re.compile(r'\\operatorname {(%s)}' % operators)
    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as fout:
        prepre = open(output_file, 'r').read().replace('\r', ' ')  # delete \r
        # replace split, align with aligned
        prepre = re.sub(r'\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}', r'\\begin{aligned}\2\\end{aligned}', prepre, flags=re.S)
        prepre = re.sub(r'\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}', r'\\begin{matrix}\2\\end{matrix}', prepre, flags=re.S)
        fout.write(prepre)

    # print(os.path.abspath(__file__))
    cmd = r"cat %s | node %s %s > %s " % (temp_file, os.path.join(os.path.dirname(__file__), 'preprocess_latex.js'), parameters.mode, output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        logging.error('FAILED: %s' % cmd)
    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)
    with open(temp_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                if len(tokens_out) > 5:
                    post = ' '.join(tokens_out)
                    # use \sin instead of \operatorname{sin}
                    names = ['\\'+x.replace(' ', '') for x in re.findall(ops, post)]
                    post = re.sub(ops, lambda match: str(names.pop(0)), post).replace(r'\\ \end{array}', r'\end{array}')
                    fout.write(post+'\n')
    os.remove(temp_file)


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
