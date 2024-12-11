data_dir=/livingrooms/public/uasr_rl
reborn_dir=/home/dmnph/reborn-uasr
output_dir=/home/dmnph/reborn-output

source ${reborn_dir}/path.sh

cd ${reborn_dir}/rl/cnn_segmenter

# For LibriSpeech
tag=ls
TAG=LS
lang=en
LANG=EN
dataset=ls100h
dataset_name=ls_100h_new
g_type=""                           # ["",      "hb_",      "wavlm"]
hr_type="ll60k"                     # ["ll60k", "ll60k",    ""]
save_tag=ls                         # ["ls",    "hb_ls",    "wavlm_ls"]

# For MLS
# lang=de                             # [de, es, fr, it, nl, pt]
# save_tag=${lang}
# dataset=mls
# dataset_name=${lang}_${dataset}
# g_type=""
# hr_type=xlsr_100hr

audio_dir=${data_dir}/${dataset_name}/${g_type}${hr_type}

python3 pretrain_cnn_segmenter.py \
    --audio_dir ${audio_dir} \
    --save_tag ${save_tag} \
    --output_dir ${output_dir}/cnn_segmenter \
    --clus_num 128  # 64 only for wavlm

