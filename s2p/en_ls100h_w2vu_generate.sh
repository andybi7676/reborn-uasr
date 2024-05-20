export HYDRA_FULL_ERROR=1
export FAIRSEQ_ROOT=/home/andybi7676/Desktop/reborn-uasr/fairseq
export PYTHONPATH=$FAIRSEQ_ROOT

TASK_DATA=/home/andybi7676/Desktop/reborn-uasr/data2/ls_100h_new/wavlm/precompute_pca512_cls64_mean_pooled
TEXT_DATA=/home/andybi7676/Desktop/reborn-uasr/data/ls_100h_new/text/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/mls_en/train/prep/phones
SUBSET=test
# SAVE_DIR=en_ls100h/hb_ll60k_postITER1/ls860_unpaired_all/best_unsup
SAVE_DIR=en_ls100h/wavlm/ls860_unpaired_all/best_unsup
DECODE_METHOD=viterbi
DECODE_TYPE=phones
BEAM=500
LM_WEIGHT=5.0
TARGET_DATA_DIR=/home/andybi7676/Desktop/reborn-uasr/data/ls_100h_new/labels
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/mls_en
# words or phones
if test "$DECODE_TYPE" = 'words'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.wrd
    LM_PATH=$TEXT_DATA/../kenlm.wrd.o40003.bin
    LEXICON_PATH=$TEXT_DATA/../lexicon_filtered.lst
elif test "$DECODE_TYPE" = 'phones'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.phn
    LM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin
    LEXICON_PATH=$TEXT_DATA/lexicon.phones.lst
else
    echo "Invalid decode type, please choose from {words/phones}"
fi
# TARGET_DATA=/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/asr_test.phn

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

cp $TEXT_DATA/dict* $TASK_DATA
python w2vu_generate.py --config-dir config/generate --config-name ${DECODE_METHOD} \
    beam=${BEAM} \
    lm_weight=${LM_WEIGHT} \
    lm_model=${LM_PATH} \
    lexicon=${LEXICON_PATH} \
    targets=${TARGET_DATA} \
    post_process=silence \
    fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    fairseq.task.data=${TASK_DATA} \
    fairseq.common_eval.path=/home/andybi7676/Desktop/reborn-uasr/s2p/multirun/${SAVE_DIR}/checkpoint_best.pt \
    fairseq.dataset.gen_subset=${SUBSET} results_path=/home/andybi7676/Desktop/reborn-uasr/s2p/multirun/${SAVE_DIR}/${SUBSET}_${DECODE_METHOD}_${BEAM}-${LM_WEIGHT}.${DECODE_TYPE}

# rm $TASK_DATA/dict*
# rm $TASK_DATA/lm* $TASK_DATA/dict* $TASK_DATA/*log $TASK_DATA/train.bin $TASK_DATA/train.idx
