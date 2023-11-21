import os
import os.path as osp
import argparse
import numpy as np
import edit_distance

def main(args):
    # print(args)
    error_ary = []
    with open(args.hyp, 'r') as hyp_fr, open(
        args.ref, 'r'
    ) as ref_fr:
        for hyp_l, ref_l in zip(hyp_fr, ref_fr):
            hyp = hyp_l.strip().split()
            ref = ref_l.strip().split()
            hyp_sub_errs_npy = np.zeros(len(hyp))
            sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
            opcodes = sm.get_opcodes()

            for opc in opcodes:
                edit_type = opc[0]
                if edit_type == "replace" or edit_type == "delete":
                    s, e = opc[1], opc[2]
                    hyp_sub_errs_npy[s:e] = 1
            error_ary.append(hyp_sub_errs_npy)
    output_fpath = args.out
    if output_fpath == "":
        output_fpath = '.'.join(args.hyp.split('.')[:-1]) + "_subdelerr"
    np.save(output_fpath, np.array(error_ary, dtype="object"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp",
        default="",
        help="a sample arg",
    )
    parser.add_argument(
        "--ref", 
        default="",
        help="a sample arg",
    )
    parser.add_argument(
        "--out", 
        default="",
        help="a sample arg",
    )
    args = parser.parse_args()

    main(args)