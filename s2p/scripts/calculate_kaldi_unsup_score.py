import kenlm
import os
import os.path as osp
import editdistance
import math
import tqdm
import argparse
import numpy as np
from collections import defaultdict

def parse_dir(d):
    aw, bw = d.split('/')[-1].split('_')[-2:]
    return float(aw), float(bw)

def get_lm_related_scores(hyps, kenlm):
    lengths = []
    lm_scores = []
    for line in hyps:
        line.replace("<SIL>", "")
        line.replace("  ", " ")
        line = line.strip()
        words = line.split()
        lengths.append(len(words))
        lm_score = kenlm.score(line)
        lm_scores.append(lm_score)
    return lengths, lm_scores

def get_entropies(hyps, kenlm):
    lengths, lm_scores = get_lm_related_scores(hyps, kenlm)
    lm_entropies = [-log_prob / (l+2) for l, log_prob in zip(lengths, lm_scores)]
    total_lm_entropy = -sum(lm_scores) / (sum(lengths) + 2*len(lengths))
    return np.array(lm_entropies), total_lm_entropy

def get_total_ppl(lines, kenlm):
    lengths, lm_scores = get_lm_related_scores(lines, kenlm)
    total_lm_score, total_lengths = sum(lm_scores), sum(lengths)
    total_ppl = math.pow(
        10,
        -total_lm_score / (total_lengths + 2*len(lengths))
    )
    return total_ppl

def read_file(fpath):
    lines = []
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.strip()
            lines.append(line)
    return lines

def get_vt_diffs(hyps, vit_trans):
    hyps = [h.replace(" <SIL> ", " ") for h in hyps]
    vt_errs = np.array([
        editdistance.eval(vt, h) for vt, h in zip(vit_trans, hyps)
    ])

    vt_lengths = np.array([ len(vt.split()) for vt in vit_trans ])
    total_vt_diff = sum(vt_errs) / sum(vt_lengths)
    return vt_errs / vt_lengths, total_vt_diff

def build_lexicon(lexicon_fpath):
    wrd_to_phn = defaultdict(lambda: None)
    with open(lexicon_fpath, 'r') as fr:
        for line in fr:
            items = line.rstrip().split('\t')
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1].split()
    return wrd_to_phn

def words_to_phones(hyps, wrd_to_phn, insert_sil):
    new_hyps = []
    for hyp in hyps:
        words = hyp.split()
        phones = []
        for w in words:
            phns = wrd_to_phn[w]
            if type(phns) == type([]):
                phones.extend(phns)
                if insert_sil:
                    phones.append("<SIL>")
        if insert_sil:
            phones = phones[:-1]
        new_hyp = " ".join(phones)
        new_hyps.append(new_hyp)
    return new_hyps


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
        "--lexicon", 
        help="Use lexicon for word-level selection",
        default="",
    )
    parser.add_argument(
        "--min_vit_uer", "-u", default=0.03,
        type=float,
        help="the min vt_uer for u-tuning"
    )
    parser.add_argument(
        "--train_data_ppl", 
        type=float, 
        default=None
    )
    parser.add_argument(
        "--subset", "-s",
        default="valid_small"
    )
    parser.add_argument(
        "--insert_sil",
        action='store_true'
    )
    args = parser.parse_args()
    print(args)

    save_root = args.save_root
    vit_fpath = args.viterbi_fpath
    kenlm_path = args.kenlm
    train_data_dir = args.training_data_dir
    subset = args.subset
    min_vit_uer = args.min_vit_uer
    if kenlm_path == "Use the 4-gram in training_data_dir":
        kenlm_path = osp.join(train_data_dir, "lm.phones.filtered.04.bin")
        if not osp.exists(kenlm_path):
            kenlm_path = osp.join(train_data_dir, "train_text_phn.04.bin")
    
    assert osp.exists(kenlm_path), kenlm_path

    print("Loading kenlm......")
    kenlm_model = kenlm.Model(kenlm_path)
    print("Kenlm loaded.")


    if args.train_data_ppl:
        min_lm_ppl = args.train_data_ppl
        print(f"Training lm_ppl={min_lm_ppl}")
    else:
        print("Getting training lm_ppl......")
        lm_train_data = osp.join(train_data_dir, "lm.phones.filtered.txt")
        if not osp.exists(lm_train_data):
            lm_train_data = osp.join(train_data_dir, "../train_text.phn")
        assert osp.exists(lm_train_data), lm_train_data
        lines = read_file(lm_train_data)
        min_lm_ppl = get_total_ppl(lines, kenlm_model)
        print(f"\tTraining_lm_ppl={min_lm_ppl}")
    min_lm_entropy = math.log(min_lm_ppl, 10)
    print(f"\tTraining_lm_entropy={min_lm_entropy}")
    if not save_root: return

    wrd_to_phn = None
    if args.lexicon != "":
        lexicon_fpath = args.lexicon
        print(f"Use lexicon, fpath={lexicon_fpath}")
        assert osp.exists(lexicon_fpath), lexicon_fpath
        wrd_to_phn = build_lexicon(lexicon_fpath)

    vit_trans = read_file(vit_fpath)
    best_score_params = None
    best_score = float("inf")
    uer = float("inf")
    ppl = float("inf")
    ppl_dict = {}
    uer_dict = {}
    score_dict = {}
    alpha = 2.712
    for root, dirs, files in os.walk(osp.join(save_root, "details")):
        if len(dirs) == 0: continue
        for d in tqdm.tqdm(dirs):
            aw, bw = parse_dir(d)
            hyp_fpath = osp.join(root, d, f"{subset}.txt")
            hyps = read_file(hyp_fpath)
            if wrd_to_phn:
                hyps = words_to_phones(hyps, wrd_to_phn, args.insert_sil)
            lm_entropies, total_lm_entropy = get_entropies(hyps, kenlm_model)
            # print(lm_entropies)
            lm_ppl = math.pow(10, lm_entropies.mean())
            ppl_dict[(aw, bw)] = lm_ppl
            uer_dict[(aw, bw)] = lm_ppl
            vt_diffs, total_vt_diff = get_vt_diffs(hyps, vit_trans)
            # weighted_score = lm_entropies * max(vt_diff, min_vit_uer)
            assert len(vt_diffs) == len(lm_entropies)
            # weighted_score = sum([max(lm_entorpy, min_lm_entropy) * max(vt_diff, min_vit_uer) for lm_entorpy, vt_diff in zip(lm_entropies, vt_diffs)])
            # weighted_score = total_lm_entropy * total_vt_diff
            weighted_score = alpha * total_lm_entropy + total_vt_diff
            score_dict[(aw, bw)] = (total_lm_entropy, total_vt_diff, weighted_score)
            if weighted_score <= best_score:
                best_score = weighted_score
                best_score_params = (aw, bw)
    
    with open(osp.join(save_root, f"best_unsup_score_alpha={alpha}.txt"), 'w') as bfw:
        weights = list(score_dict.keys())
        weights.sort()
        last_aw = weights[0][0]
        print("Score matrix:", file=bfw)
        score_list=[]
        for aw, bw in weights:
            if aw != last_aw:
                print("\n", file=bfw)
                last_aw = aw
            # else:
            score = score_dict[(aw, bw)]
            weighted_score = f"{score[-1]:3.5f}"
            if weighted_score == "inf":
                weighted_score = "*inf*"
            print(f"{weighted_score}", file=bfw, end='\t')
            score_list.append((aw, bw, score))
        score_list.sort(key=lambda x: x[-1][-1], reverse=False)
        print(f"\n\nResults:", file=bfw)
        print(f"Best unsup score: {best_score_params}, score={best_score}", file=bfw)
        print("\n\nDetails:", file=bfw)
        for sc in score_list:
            (aw, bw, score) = sc
            print(f"({aw}, {bw}): entorpy: {score[0]}, user: {score[1]}, weighted_score: {score[2]}", file=bfw)

if __name__ == "__main__":
    main()