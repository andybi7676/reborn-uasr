print("Testing the required import functionality...")

try:
    from fairseq import checkpoint_utils, tasks, utils
except:
    print("Import error, please make sure that you have run `source path.sh` and the variable FAIRSEQ_ROOT is correct.")
    exit(1)

try:
    from rl.reward.scorer import Scorer
    from s2p.utils.per import cal_per
except:
    print("Import error, please make sure that you have run `source path.sh` and the variable REBORN_WORK_DIR is correct.")
    exit(1)

print("SUCCESS")