#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import glob

def parse_dir(d):
    aw, bw = d.split('/')[-1].split('_')[-2:]
    return float(aw), float(bw)

def get_score(path):
    wer = float("inf")
    score = float("inf")
    try:
        res_file = glob.glob(osp.join(path, "*.res"))[0]
        with open(res_file, 'r') as res_fr:
            res = res_fr.readline().strip()
            wer = float(res.split("WER:")[-1].split(',')[0].strip())
            score = float(res.split("score:")[-1].split(',')[0].strip())
    except: 
        print(f"error when getting wer and score in {path}")
    return wer, score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_root",
        help="the root dir of kaldi decoding",
    )
    parser.add_argument(
        "--type",
        default="words",
        help="select the decode type from {words, phones}",
    )
    args = parser.parse_args()
    save_root = args.save_root
    print(f"Root of parameter-search decoding: {save_root}")
    # ${DECODE_TYPE}/ckpt_${CKPT_SELECTION}_${SUBSET}_${BEAM}_${aw}_${bw}
    score_dict = {}
    wer_dict = {}
    best_score_params = None
    best_wer_params = None
    best_score = float("inf")
    best_wer   = float("inf")
    for root, dirs, files in os.walk(osp.join(save_root, "details")):
        for d in dirs:
            aw, bw = parse_dir(d)
            wer, score = get_score(osp.join(root, d))
            wer_dict[(aw, bw)] = wer
            score_dict[(aw, bw)] = score
            if score <= best_score:
                best_score = score
                best_score_params = (aw, bw)
            if wer <= best_wer:
                best_wer = wer
                best_wer_params = (aw, bw)
        with open(osp.join(save_root, "best_score.txt"), 'w') as bfw:
            weights = list(score_dict.keys())
            weights.sort()
            last_aw = weights[0][0]
            print("Score matrix:", file=bfw)
            for aw, bw in weights:
                if aw != last_aw:
                    print("\n", file=bfw)
                    last_aw = aw
                # else:
                score = f"{score_dict[(aw, bw)]:3.3f}"
                if score == "inf":
                    score = "*inf*"
                print(f"{score}", file=bfw, end='\t')
            print("\n\nWER matrix:", file=bfw)
            last_aw = weights[0][0]
            for aw, bw in weights:
                if aw != last_aw:
                    print("\n", file=bfw)
                    last_aw = aw
                print(f"{wer_dict[(aw, bw)]:3.3f}", file=bfw, end='\t')
            print("\n\nResults:", file=bfw)
            print(f"Best score: {best_score_params}, score={best_score}", file=bfw)
            print(f"Best WER: {best_wer_params}, wer={best_wer}", file=bfw)


        


if __name__ == "__main__":
    main()
