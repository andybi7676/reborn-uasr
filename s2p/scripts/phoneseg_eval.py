# Usage: 
# python s2p/scripts/phoneseg_eval.py \
# data/large_clean/CLUS128/test.adj_boundaries \
# data/large_clean_mfa/CLUS128/test.boundaries

import sys
from tqdm import tqdm
from phoneseg_utils import PrecisionRecallMetric

# predicted and ground truth boundaries files
pred_file = open(sys.argv[1], 'r')
gt_file = open(sys.argv[2], 'r')

pred_lines = pred_file.readlines()
gt_lines = gt_file.readlines()
assert len(pred_lines) == len(gt_lines)

metric_tracker_harsh = PrecisionRecallMetric(tolerance=1, mode="harsh")
metric_tracker_lenient = PrecisionRecallMetric(tolerance=1, mode="lenient")

for pred, gt in tqdm(zip(pred_lines, gt_lines), total=len(pred_lines)):
    pred = pred.strip().split()
    gt = gt.strip().split()
    assert len(pred) == len(gt)

    pred = [[i for i, frame in enumerate(pred) if frame != '0']]
    gt = [[i for i, frame in enumerate(gt) if frame != '0']]

    metric_tracker_harsh.update(gt, pred)
    metric_tracker_lenient.update(gt, pred)

tracker_metrics_harsh = metric_tracker_harsh.get_stats()
tracker_metrics_lenient = metric_tracker_lenient.get_stats()

print(f"{'SCORES:':<15} {'Lenient':>10} {'Harsh':>10}")
for k in tracker_metrics_harsh.keys():
    print("{:<15} {:>10.4f} {:>10.4f}".format(k+":", tracker_metrics_lenient[k], tracker_metrics_harsh[k]))

