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
hr_type=ll60k
unpair_name=ls860

if [ $tag == "ls" ]; then
Pretrain_segmenter_path=./output/cnn_segmenter/${tag}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt
else
Pretrain_segmenter_path=./output/cnn_segmenter/${lang}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt
fi

coef_ter_list="0.2 0.0"
coef_len_list="0.2 0.0"

# 4 settings: LM with or without sil, Transcription with or without sil
# posttag: ["_LMnosil_Tnosil", "_LMnosil_Tsil", "_LMsil_Tnosil", "_LMsil_Tsil"]
posttag="_LMsil_Tnosil"
seeds="11"

lr_list="1e-4"

for seed in ${seeds}
do
for coef_ter in ${coef_ter_list}
do
for coef_len in ${coef_len_list}
do
for lr in ${lr_list}
do

# # Check if ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed3 exists
# if [ -d ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} ]; then
#     echo "Directory exists: ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}"
#     continue
# fi

# Check if ${WORK_DIR}/output/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/rl_agent_segmenter_epoch40.pt exists
if [ -f ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/rl_agent_segmenter_epoch40.pt ]; then
    echo "File exists: ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/rl_agent_segmenter_epoch40.pt"
    continue
fi

# Check if ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/running exists
if [ -d ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/running ]; then
    echo "Directory exists: ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/running"
    continue
fi

# Add a folder to indicate that the run is in progress
mkdir ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/running

max_epoch=0
# If ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} exists, search the rl_agent_segmenter_epoch{}.pt with the highest epoch number
if [ -d ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag} ]; then
    echo "Directory exists: ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}"
    # Find the rl_agent_segmenter_epoch{}.pt with the highest epoch number
    for file in ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/rl_agent_segmenter_epoch*.pt;
    do
        epoch=$(echo $file | sed -e "s/.*rl_agent_segmenter_epoch//" -e "s/.pt//")
        if [ $epoch -gt $max_epoch ]; then
            max_epoch=$epoch
            Pretrain_segmenter_path=$file
        fi
    done
    echo "max_epoch: $max_epoch"
    echo "Pretrain_segmenter_path: $Pretrain_segmenter_path"
fi


# 1. 
# Modify [pretrain_segmenter_path] for without or with bc
# pretrain_segmenter_path: [None, ./output/cnn_segmenter/${tag}_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt]

# 2.
# Modify [kenlm_fpath], [lm_rm_sil], [ter_rm_sil] for without or with sil
# 2-1. LM
# kenlm_fpath: [lm.phones.filtered.nosil.04.bin, lm.phones.filtered.04.bin]
# lm_rm_sil: [True, False]

# 2-2. Token Error Rate / Length Ratio
# ter_rm_sil: [True, False]

## w2vu_postfix: now all use w2vu_logit_segmented_units (with sil)

python3 train_rl_cnnagent_enpei.py \
    --config ${lang}_${dataset} \
    --data_dir ${data_dir}/${dataset_name}/${hr_type} \
    --kenlm_fpath ${data_dir}/${dataset_name}/text/prep/phones/lm.phones.filtered.04.bin \
    --dict_fpath ../dict/${lang}_${dataset}/dict.txt \
    --pretrain_segmenter_path ${Pretrain_segmenter_path} \
    --pretrain_wav2vecu_path ../../s2p/multirun/${lang}_${dataset}/${hr_type}/${unpair_name}_unpaired_all/best_unsup/checkpoint_best.pt \
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
    --wandb_log \
    --ter_rm_sil 
    # --pretrain_segmenter_path None \

# remove the folder to indicate that the run is finished
rm -r ${output_dir}/rl_agent/${tag}_${lang}/${TAG}_${LANG}_pplNorm1.0_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${posttag}/running

done
done
done
done