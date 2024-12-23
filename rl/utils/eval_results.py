import sys
import os
import argparse
from omegaconf import OmegaConf

def cal_lengths(refs, hyps):
    ref_lengths = []
    hyp_lengths = []
    for ref in refs:
        ref_lengths.append(len(ref))
    for hyp in hyps:
        hyp_lengths.append(len(hyp))
    return ref_lengths, hyp_lengths

def main(args):
    from s2p.utils.per import read_phn_file, cal_per
    if args.split != "test":
        ref = args.ref.replace("test", args.split)
    else:
        ref = args.ref
    hyps = read_phn_file(args.hyp)
    refs = read_phn_file(ref)
    S, D, I, N, count = cal_per(refs, hyps)
    ref_lengths, hyp_lengths = cal_lengths(refs, hyps)
    print(f"PER: {(S+D+I)/N *100} %")
    print(f"DEL RATE: {(D)/N *100} %")
    print(f"INS RATE: {(I)/N *100} %")
    print(f"SUB RATE: {(S)/N *100} %")
    print(f"total lines: {count}")
    print(f"total ref tokens: {sum(ref_lengths)}")
    print(f"total hyp tokens: {sum(hyp_lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate w2vu segmental results",
    )
    parser.add_argument(
        "--env",
        default="../../env.yaml",
        help="custom local env file for github collaboration",
    )
    parser.add_argument(
        "--hyp",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--ref",
        default="./golden/test.phones.txt"
    )
    parser.add_argument(
        "--split",
        default="test"
    )
    args = parser.parse_args()
    env = OmegaConf.load(args.env)
    sys.path.append(f"{env.WORK_DIR}")
    main(args)