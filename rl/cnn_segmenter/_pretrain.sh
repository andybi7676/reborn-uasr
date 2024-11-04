source ~/.bashrc
conda activate wav2vecu

# export PYTHONPATH=$PYTHONPATH:/home/r11921042/uasr-rl/fairseq

# WORK_DIR=/work/r11921042

data_dir=/livingrooms/public/uasr_rl
reborn_dir=/home/dmnph/reborn-uasr
output_dir=/home/dmnph/reborn_output

source ${reborn_dir}/path.sh

## For LibriSpeech
tag=ls
TAG=LS
lang=en
LANG=EN
dataset=ls100h
dataset_name=ls_100h_new
g_type=wavlm # "hb_" or "" or "wavlm"
hr_type="" # "ll60k" or ""
unpair_name=ls860
save_tag=wavlm_ls # "hb_ls" or "ls" or "wavlm_ls"

## For MLS
# lang=de # [de, es, fr, it, nl, pt]
# save_tag=${lang}
# dataset=mls
# dataset_name=${lang}_${dataset}
# hr_type=xlsr_100hr
# unpair_name=${lang}


python3 pretrain_cnn_segmenter_enpei.py \
    --audio_dir ${data_dir}/${dataset_name}/${g_type}${hr_type} \
    --save_tag ${save_tag} \
    --clus_num 64  # 64 only for wavlm

