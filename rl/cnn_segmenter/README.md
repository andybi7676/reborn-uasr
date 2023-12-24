# Train RL agent

## Data Preparation (For computing token error rate between rl agent and Wav2VecU) 
1. Move `../utils/w2vu_segmental_results/logit_segment/train.txt` to `$WORKDIR/data/audio/ls_100h_clean/large_clean/precompute_pca512/train.logit_segment`
2. Repeat for `valid.txt` and `test.txt`

## Start Training

* Custumize hyperparameter(line 523-531), reward coefficient and other parameters(line 26-36)
```
python3 train_rl_cnnagent.py
```

## Evaluation
```
bash test_rl_cnnagent.sh
```
