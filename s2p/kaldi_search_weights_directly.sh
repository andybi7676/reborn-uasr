export HYDRA_FULL_ERROR=1
# CHECKPOINT=/work/b07502072/pretrained_models/w2v_large_lv_fsh_swbd_cv.pt # large_noisy
# CHECKPOINT=/work/b07502072/pretrained_models/wav2vec2_large_west_germanic_v2.pt # large_vox
CHECKPOINT=/work/b07502072/pretrained_models/xlsr_53_56k.pt # xlsr
# CHECKPOINT=/work/b07502072/pretrained_models/wav2vec_vox_new.pt # large_clean
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/fr_feats/cv4/xlsr
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/es_feats/cv4/xlsr
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/de_feats/cv4/xlsr
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/LJ_speech/large_clean
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/es/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/fr/prep_sil_0-5/phones
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/fr/prep_new_reduced_sil_0-5/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/ls_wo_lv/prep_g2p/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/ls_wo_lv/prep/phones

SUBSET=valid_small

SAVE_DIR=cv4_fr/xlsr/cv_wiki_new_reduced_sil_0-5_300k/cp4_gp2.0_sw0.5/seed1
# SAVE_DIR=LJ_speech/large_clean/ls_wo_lv/cp4_gp2.0_sw0.5/seed2
# SAVE_DIR=cv4_es/xlsr/cv_wiki_all/cp4_gp1.5_sw0.5/seed2
# SAVE_DIR=cv4_es/xlsr/cv_wiki_all/cp4_gp1.5_sw0.5/seed2
# SAVE_DIR=voxpopuli_de/xlsr_new/vox_trans/cp4_gp2.0_sw0.5/seed1
DECODE_METHOD=kaldi
DECODE_TYPE=phones
BEAM=50
# --------determine whether to set sil token to blank or not----------#
SIL_IS_BLANK=true 
LM=lm4
gram=4
adj_pool=true
# min_lm_ppl=7.884893153191544 #ls_wo_lv/prep_g2p
# min_lm_ppl=7.663504812120307 #ls_wo_lv/prep
# min_lm_ppl=7.279692976965774 #cv_wiki/fr/prep_sil_0-5
min_lm_ppl=7.269818099634827 #cv_wiki/fr/prep_new_reduced_sil_0-5
min_vt_uer=0.05
#----------------------kaldi decoder config---------------------------#
LM_WEIGHT=1.0
TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_fr/train_70h
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_de
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_es/train_all
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/LJ_speech
CKPT_SELECTION=best
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/mls_en
# words or phones
VIT_TRANS=""
if test "$DECODE_TYPE" = 'words'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.words.txt
    LM_PATH=$TEXT_DATA/../kenlm.wrd.o40003.bin
    LEXICON_PATH=$TEXT_DATA/../lexicon_filtered.lst
    if test "$SIL_IS_BLANK" = "true"; then
        HLG_PATH=$TEXT_DATA/../fst/phn_to_words_sil/HLG.phn.kenlm.wrd.o40003.fst
        OUTPUT_DICT=$TEXT_DATA/../fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt
    else
        HLG_PATH=$TEXT_DATA/../fst/phn_to_words/HLG.phn.kenlm.wrd.o40003.fst
        OUTPUT_DICT=$TEXT_DATA/../fst/phn_to_words/kaldi_dict.kenlm.wrd.o40003.txt
    fi
elif test "$DECODE_TYPE" = 'phones'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.phones.txt
    LM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin
    LEXICON_PATH=$TEXT_DATA/lexicon.phones.lst
    HLG_PATH=$TEXT_DATA/../fst/phn_to_phn_sil_$LM/HLG.phn.lm.phones.filtered.0$gram.fst
    OUTPUT_DICT=$TEXT_DATA/../fst/phn_to_phn_sil_$LM/kaldi_dict.lm.phones.filtered.0$gram.txt
    VIT_TRANS=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/ckpt_${CKPT_SELECTION}_${SUBSET}_viterbi_500-5.0.phones/${SUBSET}.txt
else
    echo "Invalid decode type, please choose from {words/phones}"
fi

echo "AUDIO_DATA: $TASK_DATA"
echo "TEXT_DATA: $TEXT_DATA"
echo "SAVE_DIR: $SAVE_DIR"
echo "SUBSET: $SUBSET"
echo "DECODE_METHOD: $DECODE_METHOD"
echo "DECODE_TYPE: $DECODE_TYPE"
echo "LM_PATH: $LM_PATH"
echo "LEXICON_PATH: $LEXICON_PATH"
echo "TARGET_DATA: $TARGET_DATA"
echo "BEAM: $BEAM"
echo "LM_WEIGHT: $LM_WEIGHT"
echo "VIT_TRANS: $VIT_TRANS"

cp /home/b07502072/u-speech2speech/s2p/kalid_search_weights_directly.sh /home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/kaldi_decode/search_weights/${DECODE_TYPE}_${LM}_adj_pool_${adj_pool}
cp $TEXT_DATA/* $TASK_DATA
for bw in $(seq -3 1.0 5); do
    for aw in $(seq 0.4 0.2 2.0); do
        if test "$aw" != '0.0'; then
            python w2vu_generate_directly.py --config-dir config/generate/directly --config-name ${DECODE_METHOD} \
            beam=${BEAM} \
            lm_weight=${LM_WEIGHT} \
            lm_model=${LM_PATH} \
            lexicon=${LEXICON_PATH} \
            targets=${TARGET_DATA} \
            blank_weight=${bw} \
            viterbi_transcript=${VIT_TRANS} \
            min_lm_ppl=${min_lm_ppl} \
            min_vt_uer=${min_vt_uer} \
            kaldi_decoder_config.acoustic_scale=${aw} \
            kaldi_decoder_config.hlg_graph_path=${HLG_PATH} \
            kaldi_decoder_config.output_dict=${OUTPUT_DICT} \
            fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
            fairseq.task.data=${TASK_DATA} \
            fairseq.task.directly.checkpoint=${CHECKPOINT} \
            fairseq.task.directly.adjacent_pooling=${adj_pool} \
            fairseq.common_eval.path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/checkpoint_${CKPT_SELECTION}.pt \
            fairseq.dataset.gen_subset=${SUBSET} results_path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/kaldi_decode/search_weights/${DECODE_TYPE}_${LM}_adj_pool_${adj_pool}/details/ckpt_${CKPT_SELECTION}_${SUBSET}_${BEAM}_${aw}_${bw}
        fi
        rm outputs/*/*/core.*
    done
done
# rm $TASK_DATA/lm* $TASK_DATA/*log

python /home/b07502072/u-speech2speech/s2p/scripts/get_kaldi_best_score.py /home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/kaldi_decode/search_weights${DECODE_TYPE}_${LM}_adj_pool_${adj_pool}

wait
