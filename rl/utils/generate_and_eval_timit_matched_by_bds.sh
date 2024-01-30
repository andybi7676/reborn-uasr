set -eu

boundary_dir=../../data/audio/timit/matched/large_clean/GOLDEN
generator_ckpt=../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/best_unsup/checkpoint_best.pt
output_dir=$boundary_dir/eval_on_oracle_output
config_name=timit_matched
feats_dir=../../data/audio/timit/matched/large_clean/precompute_pca512
golden_dir=$feats_dir
all_splits="train valid test"

for split in $all_splits; do
    echo "Processing $split..."
    boundary_fpath=$boundary_dir/$split.bds
    # logit segmented
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --boundary_fpath $boundary_fpath \
        --output_dir $output_dir/logit_segmented \
        --split $split
    # raw
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --boundary_fpath $boundary_fpath \
        --no_logit_segment \
        --output_dir $output_dir/raw \
        --split $split
    # postprocess boundaries
    # python postprocess_boundaries.py \
    #     --bds_fpath $output_dir/raw/$split.bds \
    #     --raw_output_fpath $output_dir/raw/$split.txt \
    #     --new_bds_fpath $output_dir/raw/$split.postprocessed.bds \
    #     --length_fpath $feats_dir/$split.lengths

    echo "# $split" >> $output_dir/result.txt
    python eval_results.py --hyp $output_dir/logit_segmented/$split.txt --ref $golden_dir/$split.phn >> $output_dir/result.txt
    python eval_boundaries.py --hyp $output_dir/raw/$split.bds --ref $feats_dir/../GOLDEN/$split.bds >> $output_dir/result.txt
    echo "" >> $output_dir/result.txt
    tail -n 13 $output_dir/result.txt
done
