import sys
import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

def main(args):
    # evaluate boundaries f1
    # from s2p.scripts.phoneseg_utils import PrecisionRecallMetric
    pred_file = open(args.hyp, 'r')
    # gt_file = open(args.ref, 'r')

    pred_lines = pred_file.readlines()
    # gt_lines = gt_file.readlines()
    # assert len(pred_lines) == len(gt_lines)

    # metric_tracker_harsh = PrecisionRecallMetric(tolerance=1, mode="harsh")
    # metric_tracker_lenient = PrecisionRecallMetric(tolerance=1, mode="lenient")
    sil_counts = 0
    nonsil_counts = 0
    line_counts = 0

    for pred in tqdm(pred_lines, total=len(pred_lines)):
        pred = pred.strip().split()
        # gt = gt.strip().split()
        # assert len(pred) == len(gt)

        # location of non-boundary frames
        for i in pred:
            if i == '<SIL>':
                sil_counts += 1
            else:
                nonsil_counts += 1
        line_counts += 1
    print(f"sil_counts: {sil_counts}")
    print(f"line_counts: {line_counts}")
    print(f"sil_counts/line_counts: {sil_counts/line_counts}")
    print(f"nonsil_counts: {nonsil_counts}")
    print(f"percentage of sil: {sil_counts/(sil_counts+nonsil_counts)}")

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
    args = parser.parse_args()
    env = OmegaConf.load(args.env)
    sys.path.append(f"{env.WORK_DIR}")
    main(args)