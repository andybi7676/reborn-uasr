import sys
import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

def main(args):
    # evaluate boundaries f1
    from s2p.scripts.phoneseg_utils import PrecisionRecallMetric
    pred_file = open(args.hyp, 'r')
    gt_file = open(args.ref, 'r')

    pred_lines = pred_file.readlines()
    gt_lines = gt_file.readlines()
    assert len(pred_lines) == len(gt_lines)

    metric_tracker_harsh = PrecisionRecallMetric(tolerance=1, mode="harsh")
    metric_tracker_lenient = PrecisionRecallMetric(tolerance=1, mode="lenient")

    for pred, gt in tqdm(zip(pred_lines, gt_lines), total=len(pred_lines)):
        pred = pred.strip().split()
        gt = gt.strip().split()
        assert len(pred) == len(gt)

        # location of non-boundary frames
        pred = [[i for i, frame in enumerate(pred) if frame != '0']]
        gt = [[i for i, frame in enumerate(gt) if frame != '0']]

        # Ground truth first, model prediction second
        metric_tracker_harsh.update(gt, pred)
        metric_tracker_lenient.update(gt, pred)

    tracker_metrics_harsh = metric_tracker_harsh.get_stats()
    tracker_metrics_lenient = metric_tracker_lenient.get_stats()

    print(f"{'SCORES:':<15} {'Lenient':>10} {'Harsh':>10}")
    for k in tracker_metrics_harsh.keys():
        print("{:<15} {:>10.4f} {:>10.4f}".format(k+":", tracker_metrics_lenient[k], tracker_metrics_harsh[k]))


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
        default="../../data/audio/ls_100h_clean/large_clean_mfa/CLUS128/test.boundaries"
    )
    args = parser.parse_args()
    env = OmegaConf.load(args.env)
    sys.path.append(f"{env.WORK_DIR}")
    main(args)