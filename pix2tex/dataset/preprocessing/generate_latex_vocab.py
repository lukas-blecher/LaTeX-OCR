import sys, logging, argparse, os

def process_args(args):
    parser = argparse.ArgumentParser(description='Generate vocabulary file.')

    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file containing <img_name> <line_idx> per line. This should be the file used for training.'
                        ))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, required=True,
                        help=('Input file containing a tokenized formula per line.'
                        ))
    parser.add_argument('--output-file', dest='output_file',
                        type=str, required=True,
                        help=('Output file for putting vocabulary.'
                        ))
    parser.add_argument('--unk-threshold', dest='unk_threshold',
                        type=int, default=1,
                        help=('If the number of occurences of a token is less than (including) the threshold, then it will be excluded from the generated vocabulary.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
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

    logging.info('Script being executed: %s'%__file__)

    label_path = parameters.label_path
    assert os.path.exists(label_path), label_path
    data_path = parameters.data_path
    assert os.path.exists(data_path), data_path

    formulas = open(label_path).readlines()
    vocab = {}
    max_len = 0
    with open(data_path) as fin:
        for line in fin:
            _, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            tokens_out = []
            for token in tokens:
                tokens_out.append(token)
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    vocab_sort = sorted(list(vocab.keys()))
    vocab_out = []
    num_unknown = 0
    for word in vocab_sort:
        if vocab[word] > parameters.unk_threshold:
            vocab_out.append(word)
        else:
            num_unknown += 1
    #vocab = ["'"+word.replace('\\','\\\\').replace('\'', '\\\'')+"'" for word in vocab_out]
    vocab = [word for word in vocab_out]

    with open(parameters.output_file, 'w') as fout:
        fout.write('\n'.join(vocab))
    logging.info('#UNK\'s: %d'%num_unknown)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
