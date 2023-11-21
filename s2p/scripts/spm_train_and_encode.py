import sentencepiece as spm
import argparse
import os
import os.path as osp


def main():
    parser = get_parser()
    args = parser.parse_args()
    target_dir = args.target_dir
    os.makedirs(target_dir, exist_ok=True)
    model_prefix = osp.join(target_dir, "spm")
    if not osp.exists(model_prefix+'.model'):
        spm.SentencePieceTrainer.train(input=args.input, model_prefix=model_prefix, vocab_size=args.vocab_size, add_dummy_prefix=False)
    if args.encode_fpath:
        output_fpath = args.encode_fpath.split('.txt')[0] + "_spm.txt"
        sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        with open(args.encode_fpath, 'r') as fr, open(output_fpath, 'w') as fw:
            for l in fr:
                l_encoded = ' '.join(sp.encode(l.strip(), out_type=str))
                fw.write(l_encoded + '\n')


def get_parser():
    parser = argparse.ArgumentParser(
        description="generate bpe dict for advanced wav2vec-U"
    )
    # fmt: off
    parser.add_argument('input', help='input file for spm to train and encode with')
    parser.add_argument('--vocab_size', default=100, help='vocab size for the spm model')
    parser.add_argument("--target_dir", default="./")
    parser.add_argument("--encode_fpath", default="")
    return parser

if __name__ == "__main__":
    main()