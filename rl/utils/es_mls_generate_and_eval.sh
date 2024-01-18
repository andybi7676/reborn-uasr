segmenter_dir=../cnn_segmenter/output/local/rl_agent/es_mls_from_bc_rel_to_viterbi_normed_ppl_len0.4_2nd_best_unsup_more_epoch
segmenter_ckpt=$segmenter_dir/rl_agent_segmenter_best.pt
generator_ckpt=../../s2p/multirun/es_mls/xlsr_100hr/es_unpaired_all/second_best_unsup/checkpoint_best.pt
output_dir=$segmenter_dir
config_name=es_mls
feats_dir=../../data/es_mls/xlsr_100hr/precompute_pca512
golden_dir=../../data/es_mls/labels/100hr
all_splits="test"

for split in $all_splits; do
    echo "Processing $split..."
    # logit segmented
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --segmenter_ckpt $segmenter_ckpt \
        --output_dir $output_dir/logit_segmented \
        --split $split
    # raw
    # python generate_w2vu_segmental_results.py \
    #     --config $config_name \
    #     --feats_dir $feats_dir \
    #     --generator_ckpt $generator_ckpt \
    #     --segmenter_ckpt $segmenter_ckpt \
    #     --no_logit_segment \
    #     --output_dir $output_dir/raw \
    #     --split $split
    # # postprocess boundaries
    # python postprocess_boundaries.py \
    #     --bds_fpath $output_dir/raw/$split.bds \
    #     --raw_output_fpath $output_dir/raw/$split.txt \
    #     --new_bds_fpath $output_dir/raw/$split.postprocessed.bds \
    #     --length_fpath $feats_dir/$split.lengths

    echo "# $split" >> $output_dir/result.txt
    python eval_results.py --hyp $output_dir/logit_segmented/$split.txt --ref $golden_dir/$split.phn >> $output_dir/result.txt
    # python eval_boundaries.py --hyp $output_dir/raw/$split.postprocessed.bds --ref ../../data/audio/timit/matched/large_clean/GOLDEN/$split.bds >> $output_dir/result.txt
    echo "" >> $output_dir/result.txt
    tail -n 13 $output_dir/result.txt
done