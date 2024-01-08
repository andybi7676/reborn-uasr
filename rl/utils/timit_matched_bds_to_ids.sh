segmenter_dir=../cnn_segmenter/output/local/rl_agent/new_timit_matched_from_bc_relative_to_viterbi_more_epoch_ppl_norm
data_dir=../../data/audio/timit/matched/large_clean
output_dir=$data_dir/postITER1

mkdir $output_dir
for split in train valid test all-test; do
    cp $segmenter_dir/raw/$split.postprocessed.bds $output_dir/$split.bds
    python bds_to_ids.py \
        --bds_fpath $segmenter_dir/raw/$split.postprocessed.bds \
        --ids_fpath $output_dir/$split.src \
        --length_fpath ../../data/audio/timit/matched/large_clean/precompute_pca512/$split.lengths
done
