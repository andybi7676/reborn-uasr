segmenter_dir=../cnn_segmenter/output/local/rl_agent/ky_from_bc_relative_to_viterbi_ppl_norm
generator_ckpt=../../s2p/multirun/cv3_ky/xlsr/ky_unpaired_all/cp2_gp2.0_sw0.75/seed4/checkpoint_best.pt
output_dir=$segmenter_dir
split=test
golden_dir=./golden/ky

# logit segmented
python generate_w2vu_segmental_results.py \
    --config ky \
    --feats_dir ../../data/audio/timit/matched/large_clean/precompute_pca512 \
    --generator_ckpt $generator_ckpt \
    --segmenter_ckpt $segmenter_dir/rl_agent_segmenter_best.pt \
    --output_dir $output_dir/logit_segmented \
    --split $split
# raw
# python generate_w2vu_segmental_results.py \
#     --config timit_matched \
#     --feats_dir ../../data/audio/timit/matched/large_clean/precompute_pca512 \
#     --generator_ckpt $generator_ckpt \
#     --segmenter_ckpt $segmenter_dir/rl_agent_segmenter_best.pt \
#     --no_logit_segment \
#     --output_dir $output_dir/raw \
#     --split $split
# postprocess boundaries
# python postprocess_boundaries.py \
#     --bds_fpath $output_dir/raw/$split.bds \
#     --raw_output_fpath $output_dir/raw/$split.txt \
#     --new_bds_fpath $output_dir/raw/$split.postprocessed.bds \
#     --length_fpath ../../data/audio/timit/matched/large_clean/precompute_pca512/$split.lengths

echo "# $split" >> $output_dir/result.txt
python eval_results.py --hyp $output_dir/logit_segmented/$split.txt --ref $golden_dir/$split.phones.txt >> $output_dir/result.txt
# python eval_boundaries.py --hyp $output_dir/raw/$split.postprocessed.bds --ref ../../data/audio/timit/matched/large_clean/GOLDEN/$split.bds >> $output_dir/result.txt
echo "" >> $output_dir/result.txt
tail -n 13 $output_dir/result.txt
