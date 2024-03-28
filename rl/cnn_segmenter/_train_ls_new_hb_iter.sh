source ~/.bashrc
conda activate wav2vecu

export PYTHONPATH=$PYTHONPATH:/home/r11921042/uasr-rl/fairseq

WORK_DIR=/work/r11921042

tag=ls
TAG=LS
lang=en
LANG=EN
dataset=ls100h
dataset_name=ls_100h_new
g_type=hb_
hr_type=ll60k
unpair_name=ls860

coef_ter_list="0.2 0.0"
coef_len_list="0.2 0.0"
iter=1

lr_list="1e-4"
seed_list="3"

for seed in ${seed_list}
do
for coef_ter in ${coef_ter_list}
do
for coef_len in ${coef_len_list}
do
for lr in ${lr_list}
do

if [ $coef_ter = "0.0" ] && [ $coef_len != "0.0" ]; then
    continue
fi

# Check if ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed3 exists
if [ -d ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${g_type}${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}_postITER${iter} ]; then
    echo "Directory exists: ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${g_type}${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}_postITER${iter}"
    continue
fi

python3 train_rl_cnnagent_enpei.py \
    --config ${lang}_${dataset} \
    --data_dir ${WORK_DIR}/data/${dataset_name}/${g_type}${hr_type} \
    --kenlm_fpath ${WORK_DIR}/data/${dataset_name}/text/prep/phones/lm.phones.filtered.04.bin \
    --dict_fpath ../dict/${lang}_${dataset}/dict.txt \
    --pretrain_segmenter_path ./output/local/cnn_segmenter/${g_type}${tag}_postITER${iter}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt \
    --pretrain_wav2vecu_path ../../s2p/multirun/${lang}_${dataset}/${g_type}${hr_type}_postITER${iter}/${unpair_name}_unpaired_all/best_unsup/checkpoint_best.pt \
    --w2vu_postfix w2vu_logit_segmented_postITER${iter} \
    --env ../../env.yaml \
    --gamma 1.0 \
    --ter_tolerance 0.0 \
    --length_tolerance 0.0 \
    --logit_segment True \
    --no-apply_merge_penalty \
    --wandb_log True \
    --utterwise_lm_ppl_coeff 1.0 \
    --utterwise_token_error_rate_coeff ${coef_ter} \
    --length_ratio_coeff ${coef_len} \
    --num_epochs 40 \
    --learning_rate ${lr} \
    --seed ${seed} \
    --save_dir ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${g_type}${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}_postITER${iter} \
    --save_interval 4 \
    --ter_rm_sil
    # --rm_sil False # [Only for wosil LM]
    # --pretrain_segmenter_path /work/r11921042/output/rl_agent/ls_en/LS_EN_pplNorm1.0_tokerr0.2_lenratio0.2_lr1e-4_epoch40_seed3/rl_agent_segmenter_epoch40.pt \

done
done
done
done