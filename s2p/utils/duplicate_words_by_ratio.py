import argparse
import os
import os.path as osp
import sys

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    words_counts_dict = {}
    for l in sys.stdin:
        l = l.strip()
        word, count = l.split(' ')
        words_counts_dict[word] = int(int(count) / args.compression_ratio)
    sorted_items = sorted(words_counts_dict.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_items:
        for _ in range(count):
            print(word)
    


def get_parser():
    parser = argparse.ArgumentParser(
        description="duplicate words by counts for bpe"
    )
    # fmt: off
    parser.add_argument('--compression_ratio', default=100, help='vocab size for the spm model')
    return parser

if __name__ == "__main__":
    main()