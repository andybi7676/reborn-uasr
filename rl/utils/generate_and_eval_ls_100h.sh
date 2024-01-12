segmenter_dir=../cnn_segmenter/output/local/rl_agent/iter2_seg_from_iter1
segmenter_ckpt=$segmenter_dir/rl_agent_segmenter_best.pt
generator_ckpt=../../s2p/multirun/ls_100h/large_clean_postITER1/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed1/checkpoint_best.pt
output_dir=$segmenter_dir
config_name=librispeech
feats_dir=../../data/audio/ls_100h_clean/large_clean/precompute_pca512
golden_dir=./golden/librispeech
all_splits="valid valid_small"

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
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --segmenter_ckpt $segmenter_ckpt \
        --no_logit_segment \
        --output_dir $output_dir/raw \
        --split $split
    # postprocess boundaries
    python postprocess_boundaries.py \
        --bds_fpath $output_dir/raw/$split.bds \
        --raw_output_fpath $output_dir/raw/$split.txt \
        --new_bds_fpath $output_dir/raw/$split.postprocessed.bds \
        --length_fpath $feats_dir/$split.lengths

    echo "# $split" >> $output_dir/result.txt
    python eval_results.py --hyp $output_dir/logit_segmented/$split.txt --ref $golden_dir/$split.phones.txt >> $output_dir/result.txt
    # python eval_boundaries.py --hyp $output_dir/raw/$split.postprocessed.bds --ref ../../data/audio/timit/matched/large_clean/GOLDEN/$split.bds >> $output_dir/result.txt
    echo "" >> $output_dir/result.txt
    tail -n 13 $output_dir/result.txt
done
