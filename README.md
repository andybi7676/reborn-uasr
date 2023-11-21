# uasr-rl
This repo is for our RL2023 in-course project -- "Improving Phoneme Segmentation in Unsupervised ASR with Reinforcement Learning"

## Instruction of loading and using w2vu models
#### 1.  Install fairseq. You can try your current version first. If it does not work, you can try the version provided in the repo.
#### 2.  Create `env.yaml` under `$WORK_DIR/uasr-rl`. The file would be ignored by git. We can put any custom user variables in the file during development.
#### 3.  Set the variable: `WORK_DIR` in `$WORK_DIR/uasr-rl/env.yaml`, for example: 
```
WORK_DIR: /home/b07502072/uasr-rl
```
#### 4.  run the provided example `$WORK_DIR/uasr-rl/rl/load_w2vu_example.py` and it should work.
```
cd $WORK_DIR/uasr-rl/rl
python load_w2vu_example.py
```