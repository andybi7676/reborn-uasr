source ~/.bashrc
conda activate wav2vecu

export PYTHONPATH=$PYTHONPATH:/home/r11921042/uasr-rl/fairseq

# config_name: str = "timit_matched" # "librispeech" or "timit_matched" or "timit_unmatched"
# data_dir: str = f"{WORK_DIR}/data/audio/timit/matched/large_clean"
# kenlm_fpath: str = f"{WORK_DIR}/data/text/timit/matched/phones/train_text_phn.04.bin"
# dict_fpath: str = "../dict/timit_matched/dict.txt"
# pretrain_segmenter_path: str = "./output/cnn_segmenter/lhz/timit_matched_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_10.pt"
# pretrain_wav2vecu_path: str = "../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/cp4_gp1.5_sw0.5/seed5/checkpoint_best.pt"
# save_dir: str = f"{WORK_DIR}/output/rl_agent/timit_matched_pplNorm1.0_tokerr0.0_lenratio0.2_lr1e-5_epoch500_seed3"
# w2vu_postfix: str = "w2vu_logit_segmented"


# env: str = "../../env.yaml"
# gamma: float = 1.0
# ter_tolerance: float = 0.0
# length_tolerance: float = 0.0
# logit_segment: bool = True
# apply_merge_penalty: bool = False
# wandb_log: bool = True
# utterwise_lm_ppl_coeff: float = 1.0
# utterwise_token_error_rate_coeff: float = 1.0
# length_ratio_coeff: float = 0.0
# num_epochs: int = 500
# learning_rate: float = 1e-5

# python3 train_rl_cnnagent_discriminator_loss.py

coef_ter_list="0.0 0.2 0.4 0.6 0.8 1.0"
coef_len_list="0.0 0.2 0.4 0.6 0.8 1.0"
# lr_list="1e-5 1e-4"
lr_list="1e-4"

WORK_DIR=/work/r11921042

for coef_ter in ${coef_ter_list}
do
for coef_len in ${coef_len_list}
do
for lr in ${lr_list}
do

# Check if ${WORK_DIR}/output/rl_agent/timit_matched_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch500_seed3 exists, if yes, skip
if [ -d ${WORK_DIR}/output/rl_agent/timit_matched_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch500_seed3 ]; then
    echo "Directory exists: ${WORK_DIR}/output/rl_agent/timit_matched_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch500_seed3"
    continue
fi

python3 train_rl_cnnagent_enpei.py \
    --config timit_matched \
    --data_dir ${WORK_DIR}/data/audio/timit/matched/large_clean \
    --kenlm_fpath ${WORK_DIR}/data/text/timit/matched/phones/train_text_phn.04.bin \
    --dict_fpath ../dict/timit_matched/dict.txt \
    --pretrain_segmenter_path ./output/cnn_segmenter/lhz/timit_matched_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_10.pt \
    --pretrain_wav2vecu_path ../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/cp4_gp1.5_sw0.5/seed5/checkpoint_best.pt \
    --w2vu_postfix w2vu_logit_segmented \
    --env ../../env.yaml \
    --gamma 1.0 \
    --ter_tolerance 0.0 \
    --length_tolerance 0.0 \
    --logit_segment True \
    --apply_merge_penalty False \
    --wandb_log True \
    --utterwise_lm_ppl_coeff 1.0 \
    --utterwise_token_error_rate_coeff ${coef_ter} \
    --length_ratio_coeff ${coef_len} \
    --num_epochs 500 \
    --learning_rate ${lr} \
    --seed 3 \
    --save_dir ${WORK_DIR}/output/rl_agent/timit_matched_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch500_seed3 \
    --save_interval 10

done
done
done
    