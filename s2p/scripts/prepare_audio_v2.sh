#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

source_dir=$1
tgt_dir=$2
model=$3

if [ -z "$4" ]
  then
    dim=64
  else
    dim=$4
fi

echo "using $dim clusters for auxilary target"

if [ -z "$5" ]
  then
    layer=14
  else
    layer=$5
fi

# echo "extracting from layer $layer"

train_split=train
valid_split=valid
test_split=asr_test

all_splits=()

if [[ -f "$source_dir/train.tsv" ]]; then
    all_splits+=($train_split)
fi

if [[ -f "$source_dir/valid.tsv" ]]; then
    all_splits+=($valid_split)
fi

if [[ -f "$source_dir/$test_split.tsv" ]]; then
    all_splits+=($test_split)
fi

# echo "processing splits: $all_splits"

# mkdir -p $tgt_dir

# cp $source_dir/*.tsv $tgt_dir
# cp $source_dir/*.wrd $tgt_dir
# cp $source_dir/*.ltr $tgt_dir
# cp $source_dir/*.phn $tgt_dir
# cp $source_dir/dict* $tgt_dir

setopt shwordsplit
# set -o shwordsplit

# for split in $all_splits; do
#   python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
#   --save-dir $tgt_dir --checkpoint $model --layer $layer
# done
# echo "Finished extract features."

mkdir -p $tgt_dir/mfcc

# Consider spliting corpus into chuncks for large corpus, see HuBERT preprocessing for more details
for split in $all_splits; do
    python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py \
    $tgt_dir $split 1 0 $tgt_dir/mfcc
    
    if [[ $split = $train_split ]]; then
        python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py \
        $tgt_dir/mfcc $split 1 $tgt_dir/mfcc/cls$dim $dim --percent 1
    fi

    python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py \
    $tgt_dir/mfcc $split $tgt_dir/mfcc/cls$dim 1 0 $tgt_dir/mfcc/cls${dim}_idx
    cp $tgt_dir/mfcc/cls${dim}_idx/${split}_0_1.km $tgt_dir/$split.km
done
echo "Finished mfcc kmeans clustering"