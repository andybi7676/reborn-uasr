import os
import os.path as osp
import argparse
import math
import kenlm

def timit_to_libri(lines):
    new_lines = []
    for l in lines:
        new_l = l.replace("sil", "<SIL>")
        new_l = new_l.replace("dx", "t")
        new_l = new_l.upper()
        new_lines.append(new_l)
    return new_lines

def remove_middle_sil(lines, sil_tok="<SIL>"):
    new_lines = []
    for l in lines:
        new_l = l.replace(f" {sil_tok}", "")
        new_l += f" {sil_tok}"
        new_lines.append(new_l)
    return new_lines

def libri_to_timit(lines):
    new_lines = []
    for l in lines:
        new_l = l.replace("<SIL>", "sil")
        new_l = new_l.replace("AO", "aa")
        new_l = new_l.lower()
        new_lines.append(new_l)
    return new_lines

def read_text_fpath(fpath):
    lines = []
    with open(fpath, 'r') as fr:
        for l in fr:
            l = l.strip()
            lines.append(l)
    return lines

def main(args):
    print(args)
    kenlm_model_fpath = args.kenlm_fpath
    kenlm_model = kenlm.Model(kenlm_model_fpath)

    lines = read_text_fpath(args.train_text_fpath)
    if args.remove_sil:
        sil_tok = "<SIL>"
        for l in lines:
            if "sil" in l:
                sil_tok = "sil"
                break
        lines = remove_middle_sil(lines, sil_tok=sil_tok)
        print(lines[:10])

    text_preprocess_function = None
    if args.text_preprocess_func != None:
        text_preprocess_function = eval(args.text_preprocess_func)
    if text_preprocess_function:
        lines = text_preprocess_function(lines)

    lm_logprob_sum = sum([kenlm_model.score(l) for l in lines])
    total_lengths = sum([len(l.split()) for l in lines])
    total_lengths += len(lines) # account for eos
    final_ppl = math.pow(
        10, -lm_logprob_sum / total_lengths
    )
    
    print(final_ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kenlm_fpath",
        required=True,
        default="",
        help="kenlm fpath for ppl calculation",
    )
    parser.add_argument(
        "--train_text_fpath",
        required=True
    )
    parser.add_argument(
        "--text_preprocess_func",
        default=None,
    )
    parser.add_argument(
        "--remove_sil",
        action='store_true',
    )
    args = parser.parse_args()

    main(args)