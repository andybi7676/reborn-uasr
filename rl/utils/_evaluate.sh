source ~/.bashrc
conda activate wav2vecu

# export PYTHONPATH=$PYTHONPATH:/home/r11921042/uasr-rl/fairseq

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
config_name=${lang}_${dataset}

OUTPUT_DIR=${output_dir}/rl_agent/${tag}_${lang}
# output_list=$(ls -d $output_dir/MLS_${LANG}*)

# for further iteration: ["_postITER${iter}"]
postfix=""

# for with or without BC: posttag= ["_noBC", ""]
# 4 settings: LM with or without sil, Transcription with or without sil
# posttag: ["_LMnosil_Tnosil", "_LMnosil_Tsil", "_LMsil_Tnosil", "_LMsil_Tsil"]
posttag="_LMsil_Tnosil"
generator_ckpt=../../s2p/multirun/${lang}_${dataset}/${hr_type}${postfix}/${unpair_name}_unpaired_all/best_unsup/checkpoint_best.pt
golden_dir=${data_dir}/${dataset_name}/labels
feats_dir=${data_dir}/${dataset_name}/${hr_type}/precompute_pca512
data_root=${data_dir}/${dataset_name}/${hr_type}

# SPLIT_list
## LibriSpeech: "test test-other valid dev-other"
## TIMIT: "core-test core-dev all-test"
SPLIT_list="test test-other valid dev-other"

# "epoch0 epoch4 epoch8 epoch12 epoch16 epoch20 epoch24 epoch28 epoch32 epoch36 epoch40 best"
ckpt_typeS="best epoch40" 
rerun_best="false"

coef_ppl_list="1.0"
coef_ter_list="0.0"
coef_len_list="0.0 0.2"
seed_list="3"
lr_list="1e-4"

for split in ${SPLIT_list}
do

echo "Processing $split..."

for ckpt_type in $ckpt_typeS
do
for coef_ppl in ${coef_ppl_list}
do
for coef_ter in ${coef_ter_list}
do
for coef_len in ${coef_len_list}
do
for lr in ${lr_list}
do
for seed in ${seed_list}
do

output_name=${OUTPUT_DIR}/${TAG}_${LANG}_pplNorm${coef_ppl}_tokerr${coef_ter}_lenratio${coef_len}_lr${lr}_epoch40_seed${seed}${postfix}${posttag}
# for output_name in $output_list
# do

segmenter_dir=$output_name
output_dir=$segmenter_dir
result_dir=$output_dir/results

# make result dir
mkdir -p $result_dir

# If ckpt_type is best, check if epoch40 exists, if not, skip
if [ $ckpt_type = "best" ]; then
    if [ ! -f $segmenter_dir/rl_agent_segmenter_epoch40.pt ]; then
        echo "The run hasn't reached epoch40: $segmenter_dir/rl_agent_segmenter_epoch40.pt"
        # If result file exists, delete it
        if [ -f $result_dir/result_${split}_${ckpt_type}.txt ]; then
            rm $result_dir/result_${split}_${ckpt_type}.txt
            echo "File deleted: $result_dir/result_${split}_${ckpt_type}.txt"
        fi
        continue
    fi
fi

# Check if the result file exists
if [ -f $result_dir/result_${split}_${ckpt_type}.txt ]; then
    # Check if the result is complete
    if [ $(cat $result_dir/result_${split}_${ckpt_type}.txt | wc -l) -gt 7 ]; then
        # Check if rerun_best is true and ckpt_type is best, then do not skip
        if [ $rerun_best = "true" ] && [ $ckpt_type = "best" ]; then
            echo "File exists but rerun_best is true: $result_dir/result_${split}_${ckpt_type}.txt"
        else
            echo "File exists and complete: $result_dir/result_${split}_${ckpt_type}.txt"
            tail -n 9 $result_dir/result_${split}_${ckpt_type}.txt
            continue
        fi
    else
        # Echo that the result is incomplete, will be overwritten
        echo "File exists but incomplete: $result_dir/result_${split}_${ckpt_type}.txt"
    fi
    # Echo the old result and delete it
    cat $result_dir/result_${split}_${ckpt_type}.txt
    rm $result_dir/result_${split}_${ckpt_type}.txt
    echo "File deleted: $result_dir/result_${split}_${ckpt_type}.txt"
fi

# Check if the ckpt file exists
if [ ! -f $segmenter_dir/rl_agent_segmenter_${ckpt_type}.pt ]; then
    echo "File not found: $segmenter_dir/rl_agent_segmenter_${ckpt_type}.pt"
    continue
fi

    segmenter_ckpt=$segmenter_dir/rl_agent_segmenter_${ckpt_type}.pt

    # logit segmented
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --segmenter_ckpt $segmenter_ckpt \
        --output_dir $output_dir/logit_segmented \
        --split $split \
        --output_fname ${split}_${ckpt_type}

    # Do postprocessing (optional)
    # # raw
    # python generate_w2vu_segmental_results.py \
    #     --config $config_name \
    #     --feats_dir $feats_dir \
    #     --generator_ckpt $generator_ckpt \
    #     --segmenter_ckpt $segmenter_ckpt \
    #     --no_logit_segment \
    #     --output_dir $output_dir/raw \
    #     --split $split \
    #     --output_fname ${split}_${ckpt_type}
    # # postprocess boundaries
    # python postprocess_boundaries.py \
    #     --bds_fpath $output_dir/raw/${split}_${ckpt_type}.bds \
    #     --raw_output_fpath $output_dir/raw/${split}_${ckpt_type}.txt \
    #     --new_bds_fpath $output_dir/raw/${split}_${ckpt_type}.postprocessed.bds \
    #     --length_fpath $feats_dir/$split.lengths

    echo "# $split" >> $result_dir/result_${split}_${ckpt_type}.txt
    python eval_results.py --hyp $output_dir/logit_segmented/${split}_${ckpt_type}.txt --ref $golden_dir/$split.phn >> $result_dir/result_${split}_${ckpt_type}.txt

    # Only for TIMIT
    # python eval_boundaries.py --hyp $output_dir/raw/$split_${ckpt_type}.postprocessed.bds --ref ../../data/audio/timit/matched/large_clean/GOLDEN/$split.bds >> $result_dir/result_${split}_${ckpt_type}.txt
    
    echo "" >> $result_dir/result_${split}_${ckpt_type}.txt
    tail -n 13 $result_dir/result_${split}_${ckpt_type}.txt

done
done
done
done
done
done
done