import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

def bd_to_id(bd_npy):
    id_npy = np.zeros_like(bd_npy)
    id_npy[0] = 1
    for i in range(1, len(bd_npy)):
        if bd_npy[i] == 1:
            id_npy[i] = id_npy[i-1] + 1
        else:
            id_npy[i] = id_npy[i-1]
    return id_npy

def main(args):
    print(args)
    bds_fpath = args.bds_fpath
    ids_fpath = args.ids_fpath
    length_fpath = args.length_fpath
    with open(bds_fpath, "r") as f, open(ids_fpath, "w") as fw, open(length_fpath, "r") as lf:
        bds = f.readlines()
        lengths = lf.readlines()
        for i, (bd, l) in enumerate(tqdm(zip(bds, lengths), total=len(lengths), desc=f"Converting bds to ids...", dynamic_ncols=True)):
            bd_npy = np.array(list(map(int, bd.strip().split(" "))))
            id_npy = bd_to_id(bd_npy)
            if len(id_npy) != int(l.strip()) or len(id_npy) != len(bd_npy):
                print(f"Length mismatch: {len(id_npy)} vs {l} @line{i}")
            print(" ".join([str(id) for id in id_npy]), file=fw, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bds_fpath",
        default="",
        required=True,
    )
    parser.add_argument(
        "--ids_fpath",
        default="",
        required=True,
    )
    parser.add_argument(
        "--length_fpath",
        required=True,
    )
    args = parser.parse_args()

    main(args)