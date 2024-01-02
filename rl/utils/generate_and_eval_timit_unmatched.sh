segmenter_dir=../cnn_segmenter/output/local/rl_agent/timit_unmatched_randinit
output_dir=$segmenter_dir/post_processed
split=test
golden_dir=./golden/timit/unmatched

# python generate_w2vu_segmental_results.py \
#     --config timit_unmatched \
#     --feats_dir ../../data/audio/timit/unmatched/large_clean/precompute_pca512 \
#     --generator_ckpt ../../s2p/multirun/timit_unmatched/large_clean/timit_unpaired_1k/cp4_gp2.0_sw0.5/seed2/checkpoint_best.pt \
#     --segmenter_ckpt $segmenter_dir/rl_agent_segmenter_best.pt \
#     --output_dir $output_dir \
#     --split $split

python eval_results.py --hyp $output_dir/$split.txt --ref $golden_dir/$split.phones.txt
