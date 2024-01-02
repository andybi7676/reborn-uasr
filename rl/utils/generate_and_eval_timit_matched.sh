segmenter_dir=../cnn_segmenter/output/local/rl_agent/timit_matched_from_bc_relative_to_wfst
generator_ckpt=../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/cp4_gp1.5_sw0.5/seed3/checkpoint_best.pt
output_dir=$segmenter_dir/post_processed
split=test
golden_dir=./golden/timit/matched

python generate_w2vu_segmental_results.py \
    --config timit_matched \
    --feats_dir ../../data/audio/timit/matched/large_clean/precompute_pca512 \
    --generator_ckpt $generator_ckpt \
    --segmenter_ckpt $segmenter_dir/rl_agent_segmenter_best.pt \
    --output_dir $output_dir \
    --split $split

python eval_results.py --hyp $output_dir/$split.txt --ref $golden_dir/$split.phones.txt
