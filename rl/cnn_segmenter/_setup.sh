# Set up the environment

iter=1
lang=ls860
data_config=ls_100h_new
g_type="hb_" # or ""
hr_type=ll60k
ckpt_config=en_ls100h
ckpt_postfix=_postITER${iter}
# ckpt_postfix=""
pair_config=ls860_unpaired_all

# iter=1
# ckpt_config=timit_matched
# hr_type=large_clean
# pair_config=timit_paired_no_SA
# data_config=audio/timit/matched
# ckpt_postfix=_postITER${iter}


# lang=de
# data_config=${lang}_mls
# hr_type=xlsr_100hr
# # prep_postfix=_sep
# prep_postfix=""
# ckpt_config=${lang}_mls
# pair_config=${lang}_unpaired_all${prep_postfix}
# iter=2
# # ckpt_postfix=""
# ckpt_postfix=_postITER${iter}

# 1. Set up rl/config (no need to do this if you have already done it)


# 2. Set up boundary
# echo "Set up boundary"
# # list out every /home/r11921042/uasr-rl/s2p/scripts/find_prep_align.py /work/r11921042/data/${config}/xlsr_100hr/CLUS128/*.src
# # for file in $(ls /work/r11921042/data/${config}/xlsr_100hr/CLUS128/*.src)

# for file in $(ls /work/r11921042/data/${data_config}/${hr_type}/CLUS128/*.src)
# do
#     echo $file
#     set=$(echo $file | sed -e "s/.*CLUS128\///" -e "s/.src//")
#     echo $set
#     # Check if ${WORK_DIR}/data/${config}/xlsr_100hr/CLUS128/${set}.bds exists
#     if [ -f /work/r11921042/data/${data_config}/${hr_type}/CLUS128/${set}.bds ]; then
#         echo "File exists: /work/r11921042/data/${data_config}/${hr_type}/CLUS128/${set}.bds"
#         continue
#     fi
#     python3 /home/r11921042/uasr-rl/s2p/scripts/find_prep_align.py /work/r11921042/data/${data_config}/${hr_type}/CLUS128/${set}.src adj
#     echo "generating ${set}.bds"
# done

# 3. Set up phoneme dictionary
# echo "Set up phoneme dictionary"
# mkdir /home/r11921042/uasr-rl/rl/dict/${data_config}/
# cp /work/r11921042/data/${data_config}/text/prep${prep_postfix}/phones/dict* /home/r11921042/uasr-rl/rl/dict/${data_config}/


# 4. Set up w2vu_logit_segmented data
# List out every folder in /home/r11921042/uasr-rl/s2p/multirun/${config}/xlsr_100hr/${lang}_unpaired_all/best_unsup/ and filter out the folder names that contain "viterbi_500-5.0.phones"
for folder in $(ls /home/r11921042/uasr-rl/s2p/multirun/${ckpt_config}/${g_type}${hr_type}${ckpt_postfix}/${pair_config}/best_unsup/ | grep "viterbi_500-5.0.phones")
do
    echo $folder
    set=$(echo $folder | sed -e "s/_viterbi_500-5.0.phones//")
    echo $set
    # Check if ${WORK_DIR}/data/${config}/xlsr_100hr/precompute_pca512/${set}.w2vu_logit_segmented exists
    if [ -f /work/r11921042/data/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented${ckpt_postfix} ]; then
        echo "File exists: /work/r11921042/data/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented${ckpt_postfix}"
        continue
    fi
    cp /home/r11921042/uasr-rl/s2p/multirun/${ckpt_config}/${g_type}${hr_type}${ckpt_postfix}/${pair_config}/best_unsup/${set}_viterbi_500-5.0.phones/${set}.txt /work/r11921042/data/${data_config}/${g_type}${hr_type}/precompute_pca512/${set}.w2vu_logit_segmented${ckpt_postfix}
    echo "copying ${set}.txt to ${set}.w2vu_logit_segmented${ckpt_postfix}"
done



