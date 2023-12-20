#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
FAIRSEQ_ROOT=/home/andybi7676/Desktop/fairseq
set -e
set -u
set -o pipefail

tgt_dir=$1
seg_dir=$2
dim=512

train_split=train
valid_split=valid
test_split=test

all_splits="valid_small"
echo "processing splits: $all_splits"

for split in $all_splits; do

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/precompute_pca$dim --cluster-dir $tgt_dir/$seg_dir \
  --split $split --save-dir $tgt_dir/precompute_pca${dim}_${seg_dir}_mean --pooling mean

done

echo "Pre-processed."