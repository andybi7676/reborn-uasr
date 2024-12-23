#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
set -u
set -o pipefail

echo $FAIRSEQ_ROOT

source_dir=$1
tgt_dir=$2
model=$3
dim=512
layer=15
n_clus=128
# if [ -z "$4" ]
#   then
#     dim=512
#   else
#     dim=$4
# fi

echo "using $dim dim for PCA"

# if [ -z "$5" ]
#   then
#     layer=14
#   else
#     layer=$5
# fi

echo "extracting from layer $layer"

train_split=train
valid_split=valid
test_split=test

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
# all_splits="train valid test dev-other test-other test-bds"
all_splits="test-bds"
echo "processing splits: $all_splits"

mkdir -p $tgt_dir

# cp $source_dir/*.tsv $tgt_dir
# cp $source_dir/*.wrd $tgt_dir
# cp $source_dir/*.ltr $tgt_dir
# cp $source_dir/*.phn $tgt_dir
# cp $source_dir/dict* $tgt_dir

# setopt shwordsplit
# set -o shwordsplit

for split in $all_splits; do
#   python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
#   --save-dir $tgt_dir --checkpoint $model --layer $layer
  python scripts/wav2vec_extract_features.py $source_dir --split $split \
  --save-dir $tgt_dir --checkpoint $model --layer $layer
done
echo "Finished extract features."

# echo "Clustering..."
# python scripts/wav2vec_cluster_faiss.py $tgt_dir/${train_split}.tsv \
#     --checkpoint $model --save-dir $tgt_dir -f "CLUS$n_clus" --sample-pct 1.0
# echo "Finished clustering."

for split in $all_splits; do
  python scripts/wav2vec_apply_cluster_faiss.py $tgt_dir \
  --checkpoint $model --path $tgt_dir/CLUS$n_clus --split $split
done
echo "Applied cluster features."

# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py $tgt_dir/${train_split}.npy --output $tgt_dir/pca --dim $dim
# echo "Ran PCA."

for split in $all_splits; do
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py $tgt_dir --split $split --save-dir $tgt_dir/precompute_pca$dim --pca-path $tgt_dir/pca/${dim}_pca --batch-size 1048000

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/precompute_pca$dim --cluster-dir $tgt_dir/CLUS$n_clus \
  --split $split --save-dir $tgt_dir/precompute_pca${dim}_cls${n_clus}_mean --pooling mean

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py $tgt_dir/precompute_pca${dim}_cls${n_clus}_mean \
  --save-dir $tgt_dir/precompute_pca${dim}_cls${n_clus}_mean_pooled --split $split
done

echo "Post processed."