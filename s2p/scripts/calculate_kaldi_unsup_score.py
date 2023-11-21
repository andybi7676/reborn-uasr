import kenlm
import os
import os.path as osp
import editdistance
import math
import tqdm
import argparse

def parse_dir(d):
    aw, bw = d.split('/')[-1].split('_')[-2:]
    return float(aw), float(bw)

def get_ppl(f_path, kenlm):
    lengths_t = 0
    sent_t = 0
    lm_score = 0
    lm_ppl = float("inf")
    with open(f_path, 'r') as fr:
        for line in fr:
            line.replace("<SIL>", "")
            line.replace("  ", " ")
            line = line.strip()
            words = line.split()
            lengths_t += len(words)
            lm_score += kenlm.score(line)
            sent_t += 1
    lm_ppl = math.pow(
        10, -lm_score / (lengths_t + sent_t)
    )
    return lm_ppl

def read_file(fpath):
    lines = []
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.strip()
            lines.append(line)
    return lines

def get_vt_diff(hyp_fpath, vit_trans):
    hyps = read_file(hyp_fpath)
    vt_err_t = sum(
        editdistance.eval(vt, h) for vt, h in zip(vit_trans, hyps)
    )

    vt_length_t = sum(len(vt) for vt in vit_trans)
    return vt_err_t / vt_length_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_root", "-r", 
        help="the root dir of kaldi decoding",
        default=False
    )
    parser.add_argument(
        "--training_data_dir", '-t',
        help="for calculating min ppl"
    )
    parser.add_argument(
        "--viterbi_fpath", "-v",
        help="The viterbi transcript",
    )
    parser.add_argument(
        "--kenlm", "-k",
        help="path of kenlm",
        default="Use the 4-gram in training_data_dir"
    )
    parser.add_argument(
        "--min_vit_uer", "-u", default=0.03,
        type=float,
        help="the min vt_uer for u-tuning"
    )
    parser.add_argument(
        "--subset", "-s",
        default="valid_small"
    )
    args = parser.parse_args()

    save_root = args.save_root
    vit_fpath = args.viterbi_fpath
    kenlm_path = args.kenlm
    train_data_dir = args.training_data_dir
    subset = args.subset
    min_vit_uer = args.min_vit_uer
    if kenlm_path == "Use the 4-gram in training_data_dir":
        kenlm_path = osp.join(train_data_dir, "lm.phones.filtered.04.bin")
    lm_train_data = osp.join(train_data_dir, "lm.phones.filtered.txt")

    print("Loading kenlm......")
    kenlm_model = kenlm.Model(kenlm_path)
    print("Kenlm loaded.")


    if lm_train_data:
        print("Getting training lm_ppl......")
        min_lm_ppl = get_ppl(lm_train_data, kenlm_model)
        print(f"training_lm_ppl={min_lm_ppl}")
    if not save_root: return

    vit_trans = read_file(vit_fpath)
    best_score_params = None
    best_score = float("inf")
    uer = float("inf")
    ppl = float("inf")
    ppl_dict = {}
    uer_dict = {}
    score_dict = {}
    for root, dirs, files in os.walk(osp.join(save_root, "details")):
        if len(dirs) == 0: continue
        for d in tqdm.tqdm(dirs):
            aw, bw = parse_dir(d)
            hyp_fpath = osp.join(root, d, f"{subset}.txt")
            lm_ppl = get_ppl(hyp_fpath, kenlm_model)
            ppl_dict[(aw, bw)] = lm_ppl
            uer_dict[(aw, bw)] = lm_ppl
            vt_diff = get_vt_diff(hyp_fpath, vit_trans)
            weighted_score = math.log(lm_ppl) * max(vt_diff, min_vit_uer)
            score_dict[(aw, bw)] = weighted_score
            if weighted_score <= best_score:
                best_score = weighted_score
                best_score_params = (aw, bw)
    
    with open(osp.join(save_root, "best_unsup_score.txt"), 'w') as bfw:
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
        print("\n\nResults:", file=bfw)
        print(f"Best unsup score: {best_score_params}, score={best_score}", file=bfw)

if __name__ == "__main__":
    main()