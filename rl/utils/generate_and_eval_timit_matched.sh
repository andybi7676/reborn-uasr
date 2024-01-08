segmenter_dir=../cnn_segmenter/output/local/rl_agent/tiny_lm_timit_matched_from_bc_relative_to_viterbi_more_epoch_ppl_norm
generator_ckpt=../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/cp4_gp1.5_sw0.5/seed5/checkpoint_best.pt
output_dir=$segmenter_dir
split=all-test
golden_dir=./golden/timit/matched

# logit segmented
python generate_w2vu_segmental_results.py \
    --config timit_matched \
    --feats_dir ../../data/audio/timit/matched/large_clean/precompute_pca512 \
    --generator_ckpt $generator_ckpt \
    --segmenter_ckpt $segmenter_dir/rl_agent_segmenter_best.pt \
    --output_dir $output_dir/logit_segmented \
    --split $split
# raw
python generate_w2vu_segmental_results.py \
    --config timit_matched \
    --feats_dir ../../data/audio/timit/matched/large_clean/precompute_pca512 \
    --generator_ckpt $generator_ckpt \
    --segmenter_ckpt $segmenter_dir/rl_agent_segmenter_best.pt \
    --no_logit_segment \
    --output_dir $output_dir/raw \
    --split $split
# postprocess boundaries
python postprocess_boundaries.py \
    --bds_fpath $output_dir/raw/$split.bds \
    --raw_output_fpath $output_dir/raw/$split.txt \
    --new_bds_fpath $output_dir/raw/$split.postprocessed.bds \
    --length_fpath ../../data/audio/timit/matched/large_clean/precompute_pca512/$split.lengths

echo "" >> $output_dir/result.txt
echo "# $split" >> $output_dir/result.txt
python eval_results.py --hyp $output_dir/logit_segmented/$split.txt --ref $golden_dir/$split.phones.txt >> $output_dir/result.txt
python eval_boundaries.py --hyp $output_dir/raw/$split.postprocessed.bds --ref ../../data/audio/timit/matched/large_clean/GOLDEN/$split.bds >> $output_dir/result.txt
tail -n 13 $output_dir/result.txt
