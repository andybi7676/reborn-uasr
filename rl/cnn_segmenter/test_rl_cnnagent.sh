export PYTHONPATH=$PYTHONPATH:/home/enpei/EnPei/RL/final_usar/uasr-rl/fairseq

python3 ../utils/generate_w2vu_segmental_results.py --feats_dir ../../data/audio/ls_100h_clean/large_clean/precompute_pca512 --segmenter_ckpt ./output/rl_agent/uttwise_reward_ppl_tokerr/rl_agent_segmenter.pt --logit_segment True --postprocess_code silence --output_dir ./output/rl_agent/uttwise_reward_ppl_tokerr

python3 ../utils/eval_results.py --hyp ./output/rl_agent/uttwise_reward_ppl_tokerr/test.txt --ref ../utils/golden/test.phones.txt


### Wave2Vec2.0
# python3 ../utils/eval_results.py --hyp ../../s2p/multirun/ls_100h/large_clean/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed3/test_viterbi_500-5.0.phones/test.txt --ref ../utils/golden/test.phones.txt


### Pretrained ckpt
# python3 ../utils/generate_w2vu_segmental_results_pretrain.py --feats_dir ../../data/audio/ls_100h_clean/large_clean/precompute_pca512 --segmenter_ckpt ./output/cnn_segmenter/pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_29_0.pt --logit_segment True --postprocess_code silence --output_dir ./output/cnn_segmenter/pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR

# python3 ../utils/eval_results.py --hyp ./output/cnn_segmenter/pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/test.txt --ref ../utils/golden/test.phones.txt