import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

# MERGE_CONSECUTIVE_AMOUNT = 3
# def bd_to_new_bd(bd: str, merge_three_consecutive=True, merge_two_consecutive=True):
#     # pattern 1 1 1 --> 0 1 0
#     new_bd = bd
#     if merge_three_consecutive:
#         new_bd = bd.replace("1 1 1", "0 1 0")
#     if merge_two_consecutive:
#         new_bd = new_bd.replace("1 1", "0 1")
#     return new_bd.split(" ")
def bd_to_new_bd(bd: str, rf: str, i, left_shift=False):
    new_bd = []
    bd = bd.split(" ")
    one_count = np.array([int(b) for b in bd]).sum()
    rf = rf.split(" ")
    assert one_count+2 == len(rf), f"{i}: {one_count}\n{rf}" # kernel size = 4 & end of boundary does not account for the last "1"
    cur_idx = 0
    cur_token = rf[cur_idx]
    for b in bd:
        if b == "1":
            cur_idx += 1
            nxt_token = rf[cur_idx]
            if cur_token != nxt_token:
                cur_token = nxt_token
                new_bd.append("1")
                continue
        new_bd.append("0")
    if left_shift:
        return new_bd[1:] + ["0"]
    return new_bd
        


def main(args):
    print(args)
    bds_fpath = args.bds_fpath
    raw_output_fpath = args.raw_output_fpath
    new_bds_fpath = args.new_bds_fpath
    length_fpath = args.length_fpath
    left_shift = args.left_shift
    with open(bds_fpath, "r") as f, open(raw_output_fpath) as rf, open(new_bds_fpath, "w") as fw, open(length_fpath, "r") as lf:
        bds = f.readlines()
        rfs = rf.readlines()
        lengths = lf.readlines()
        for i, (bd, rf, l) in enumerate(tqdm(zip(bds, rfs, lengths), total=len(lengths), desc=f"Converting bds to ids...", dynamic_ncols=True)):
            bd, rf = bd.strip(), rf.strip()
            new_bd = bd_to_new_bd(bd, rf, i, left_shift=left_shift)
            if len(new_bd) != int(l.strip()):
                print(f"Length mismatch: {len(new_bd)} vs {l} @line{i}")
            print(" ".join([str(nbd) for nbd in new_bd]), file=fw, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bds_fpath",
        default="",
        required=True,
    )
    parser.add_argument(
        "--raw_output_fpath",
        default="",
        required=True,
    )
    parser.add_argument(
        "--new_bds_fpath",
        default="",
        required=True,
    )
    parser.add_argument(
        "--length_fpath",
        required=True,
    )
    parser.add_argument(
        "--left_shift",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)