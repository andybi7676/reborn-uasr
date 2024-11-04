# source ~/.bashrc
# conda activate wav2vecu

# export PYTHONPATH=$PYTHONPATH:/home/r11921042/uasr-rl/fairseq

# WORK_DIR=/work/r11921042

data_dir=/livingrooms/public/uasr_rl
reborn_dir=/home/dmnph/reborn-uasr
output_dir=/home/dmnph/reborn_output

source ${reborn_dir}/path.sh

tag=ls
TAG=LS
lang=en
LANG=EN
dataset=ls100h
dataset_name=ls_100h_new
g_type_=wavlm_
g_type=wavlm
hr_type="" # "ll60k" or ""
unpair_name=ls860

coef_ter_list="0.2"
coef_len_list="0.2 0.0" # 0.4 0.6 0.8
posttag="_LMsil_Tsil"
seeds="3"

lr_list="1e-4"

for seed in ${seeds}
do
for coef_ter in ${coef_ter_list}
do
for coef_len in ${coef_len_list}
do
for lr in ${lr_list}
do

# Check if ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed3 exists
if [ -d ${output_dir}/rl_agent/${tag}_${lang}/${g_type_}${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} ]; then
    echo "Directory exists: ${output_dir}/rl_agent/${tag}_${lang}/${g_type_}${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}"
    continue
fi

python3 train_rl_cnnagent_enpei.py \
    --config ${lang}_${dataset} \
    --data_dir ${data_dir}/${dataset_name}/${g_type}${hr_type} \
    --kenlm_fpath ${data_dir}/${dataset_name}/text/prep/phones/lm.phones.filtered.04.bin \
    --dict_fpath ../dict/${lang}_${dataset}/dict.txt \
    --pretrain_segmenter_path ./output/cnn_segmenter/${g_type_}${tag}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt \
    --pretrain_wav2vecu_path ../../s2p/multirun/${lang}_${dataset}/${g_type}${hr_type}/${unpair_name}_unpaired_all/best_unsup/checkpoint_best.pt \
    --w2vu_postfix w2vu_logit_segmented_units \
    --env ../../env.yaml \
    --gamma 1.0 \
    --ter_tolerance 0.0 \
    --length_tolerance 0.0 \
    --logit_segment \
    --no-apply_merge_penalty \
    --utterwise_lm_ppl_coeff 1.0 \
    --utterwise_token_error_rate_coeff ${coef_ter} \
    --length_ratio_coeff ${coef_len} \
    --num_epochs 40 \
    --learning_rate ${lr} \
    --seed ${seed} \
    --save_dir ${output_dir}/rl_agent/${tag}_${lang}/${g_type_}${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} \
    --save_interval 4 \
    --wandb_log \
    --clus_num 64
    # --rm_sil # [Only for wosil LM]
    # --pretrain_segmenter_path None \

done
done
done
done