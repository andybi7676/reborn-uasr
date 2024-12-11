data_dir=/livingrooms/public/uasr_rl
reborn_dir=/home/dmnph/reborn-uasr
output_dir=/home/dmnph/reborn-output

source ${reborn_dir}/path.sh

cd ${reborn_dir}/rl/cnn_segmenter

## For LibriSpeech
tag=ls
TAG=LS
lang=en
LANG=EN
prep_postfix=""
dataset=ls100h
dataset_name=ls_100h_new
hr_type=ll60k
unpair_name=ls860

## For MLS
# tag=mls
# TAG=MLS
# lang=de     # [de, es, fr, it, nl, pt]
# LANG=DE     # [DE, ES, FR, IT, NL, PT]
# prep_postfix="" # "_sep" only for it, "" for others
# dataset=mls
# dataset_name=${lang}_${dataset}
# hr_type=xlsr_100hr
# unpair_name=${lang}


audio_dir=${data_dir}/${dataset_name}/${hr_type}
kenlm_fpath=${data_dir}/${dataset_name}/text/prep${prep_postfix}/phones/lm.phones.filtered.04.bin
if [ $tag == "ls" ]; then
Pretrain_segmenter_path=${output_dir}/cnn_segmenter/${tag}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt
else
Pretrain_segmenter_path=${output_dir}/cnn_segmenter/${lang}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt
fi
Pretrain_wav2vecu_path=../../s2p/multirun/${lang}_${dataset}/${hr_type}/${unpair_name}_unpaired_all${prep_postfix}/best_unsup/checkpoint_best.pt


coef_ter_list="0.2"
coef_len_list="0.2"
posttag=""
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
if [ -d ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} ]; then
    echo "Directory exists: ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}"
    continue
fi

python3 train_rl_cnnagent.py \
    --config ${lang}_${dataset} \
    --data_dir ${audio_dir} \
    --kenlm_fpath ${kenlm_fpath} \
    --dict_fpath ../dict/${lang}_${dataset}/dict.txt \
    --pretrain_segmenter_path ${Pretrain_segmenter_path} \
    --pretrain_wav2vecu_path ${Pretrain_wav2vecu_path} \
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
    --save_dir ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} \
    --save_interval 4 \
    --wandb_log 

done
done
done
done