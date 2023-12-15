import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

MERGE_CONSECUTIVE_AMOUNT = 3
def bd_to_new_bd(bd: str, merge_three_consecutive=True, merge_two_consecutive=False):
    # pattern 1 1 1 --> 0 1 0
    new_bd = bd
    if merge_three_consecutive:
        new_bd = bd.replace("1 1 1", "0 1 0")
    if merge_two_consecutive:
        new_bd = new_bd.replace("1 1", "0 1")
    return new_bd.split(" ")

def main(args):
    print(args)
    bds_fpath = args.bds_fpath
    new_bds_fpath = args.new_bds_fpath
    length_fpath = args.length_fpath
    with open(bds_fpath, "r") as f, open(new_bds_fpath, "w") as fw, open(length_fpath, "r") as lf:
        bds = f.readlines()
        lengths = lf.readlines()
        for i, (bd, l) in enumerate(tqdm(zip(bds, lengths), total=len(lengths), desc=f"Converting bds to ids...", dynamic_ncols=True)):
            bd = bd.strip()
            new_bd = bd_to_new_bd(bd)
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
        "--new_bds_fpath",
        default="",
        required=True,
    )
    parser.add_argument(
        "--length_fpath",
        required=True,
    )
    args = parser.parse_args()

    main(args)