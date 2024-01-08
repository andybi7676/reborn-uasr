segmenter_dir=../cnn_segmenter/output/rl_agent/uttwise_reward_pplclip_tokerr0.7_lenratio0.7
segmenter_ckpt=$segmenter_dir/rl_agent_segmenter.pt
generator_ckpt=../../s2p/multirun/ls_100h/large_clean_postITER1/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed1/checkpoint_best.pt
output_dir=$segmenter_dir/libri_timit
split=core-dev
golden_dir=./golden/timit/libri_timit

# python generate_w2vu_segmental_results.py \
#     --config libri_timit \
#     --feats_dir ../../data/audio/ls_100h_clean/large_clean/precompute_pca512 \
#     --generator_ckpt $generator_ckpt \
#     --segmenter_ckpt $segmenter_ckpt \
#     --output_dir $output_dir \
#     --split $split

# python libri_timit_postprocess.py < $output_dir/$split.txt > $output_dir/$split.postprocessed.txt

python eval_results.py --hyp $output_dir/$split.postprocessed.txt --ref $golden_dir/$split.phones.new.txt
