import os
import os.path as osp
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", '-i', default="", type=str, required=True)
    parser.add_argument("--output", '-o', default="", type=str, required=True)
    args = parser.parse_args()
    dict_path = args.input
    lexicon_outpath = args.output
    dictfr = open(dict_path, 'r')
    lines = dictfr.readlines()
    dictfr.close()
    with open(lexicon_outpath, 'w') as lexiconfw:
        for line in lines:
            phone = line.split(' ')[0]
            lexiconfw.write(f"{phone}\t{phone}\n")
            # print(phone)

if __name__ == "__main__":
    main()