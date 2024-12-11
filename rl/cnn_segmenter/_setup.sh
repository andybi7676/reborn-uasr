# Set up the environment
data_dir=/livingrooms/public/uasr_rl
reborn_dir=/home/dmnph/reborn-uasr

## For LibriSpeech
lang=ls860
data_config=ls_100h_new
g_type=""                   # ["",      "hb_",      "wavlm"]
hr_type=ll60k""             # ["ll60k", "ll60k",    ""]
prep_postfix=""
ckpt_config=en_ls100h
# iter=1
ckpt_postfix=""             # ["", "_postITER${iter}"]"
pair_config=ls860_unpaired_all
clus_num=128                # 64 only for wavlm, 128 for others

## For TIMIT
# iter=1
# ckpt_config=timit_matched
# hr_type=large_clean
# pair_config=timit_paired_no_SA
# data_config=audio/timit/matched
# ckpt_postfix=_postITER${iter}

## For MLS
# lang=it # [de, es, fr, it, nl, pt]
# data_config=${lang}_mls
# g_type=""
# hr_type=xlsr_100hr
# prep_postfix=_sep # Only for it
# # prep_postfix=""
# ckpt_config=${lang}_mls
# # iter=2
# # ckpt_postfix=_postITER${iter}
# ckpt_postfix=""
# pair_config=${lang}_unpaired_all${prep_postfix}

# 1. Set up rl/config (no need to do this if you have already done it)

# 2. Set up phoneme dictionary (no need to do this if you have already done it)
# echo "Set up phoneme dictionary"
# mkdir ${reborn_dir}/rl/dict/${data_config}/
# cp ${data_dir}/${data_config}/text/prep${prep_postfix}/phones/dict* ${reborn_dir}/rl/dict/${data_config}/

# 3. Set up boundary
echo "Set up boundary"
for file in $(ls ${data_dir}/${data_config}/${g_type}${hr_type}/CLUS${clus_num}/*.src)
do
    echo $file
    set=$(echo $file | sed -e "s/.*CLUS${clus_num}\///" -e "s/.src//")
    echo $set
    # Check if ${data_dir}/${data_config}/${hr_type}/CLUS${clus_num}/${set}.bds exists
    if [ -f ${data_dir}/${data_config}/${g_type}${hr_type}/CLUS${clus_num}/${set}.bds ]; then
        echo "File exists: ${data_dir}/${data_config}/${g_type}${hr_type}/CLUS${clus_num}/${set}.bds"
        continue
    fi
    python3 ${reborn_dir}/s2p/scripts/find_prep_align.py ${data_dir}/${data_config}/${g_type}${hr_type}/CLUS${clus_num}/${set}.src adj
    echo "generating ${set}.bds"
done

# 4. Set up w2vu_logit_segmented data
# List out every folder in ${reborn_dir}/s2p/multirun/${ckpt_config}/${g_type}${hr_type}${ckpt_postfix}/${pair_config}/best_unsup/ and filter out the folder names that contain "viterbi_500-5.0.phones"
for folder in $(ls ${reborn_dir}/s2p/multirun/${ckpt_config}/${g_type}${hr_type}${ckpt_postfix}/${pair_config}/best_unsup/ | grep "viterbi_500-5.0.phones")
do
    echo $folder
    set=$(echo $folder | sed -e "s/_viterbi_500-5.0.phones//")
    echo $set
    if [ -f ${data_dir}/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented${ckpt_postfix} ]; then
        echo "File exists: ${data_dir}/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented${ckpt_postfix}"
    else
        cp ${reborn_dir}/s2p/multirun/${ckpt_config}/${g_type}${hr_type}${ckpt_postfix}/${pair_config}/best_unsup/${set}_viterbi_500-5.0.phones/${set}.txt ${data_dir}/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented${ckpt_postfix}
        echo "copying ${set}.txt to ${set}.w2vu_logit_segmented${ckpt_postfix}"
    fi

    # Also copy ${set}_units.txt (with <SIL> tokens) to ${set}.w2vu_logit_segmented_units${ckpt_postfix}
    if [ -f ${data_dir}/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented_units${ckpt_postfix} ]; then
        echo "File exists: ${data_dir}/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented_units${ckpt_postfix}"
    else
        cp ${reborn_dir}/s2p/multirun/${ckpt_config}/${g_type}${hr_type}${ckpt_postfix}/${pair_config}/best_unsup/${set}_viterbi_500-5.0.phones/${set}_units.txt ${data_dir}/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented_units${ckpt_postfix}
        echo "copying ${set}_units.txt to ${set}.w2vu_logit_segmented_units${ckpt_postfix}"
    fi
done



