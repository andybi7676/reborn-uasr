export HYDRA_FULL_ERROR=1
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/es_feats/cv4/mhubert/precompute_pca512_cls128_mean_pooled
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/de_feats/cv4/xlsr/precompute_pca512_cls128_mean_pooled
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/mls_en/spk1hr/large_clean_new/precompute_pca512_cls128_mean_pooled
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/en/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep/phones
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cc100/de/w2vu/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/fr/prep_sil_0-5/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/es/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/mls_en/train/prep/phones
SUBSET=asr_test
# SAVE_DIR=voxpopuli_de/large_vox/vox_trans/0
# SAVE_DIR=mls_en/large_clean_new/mls_trans/cp4_gp1.5_sw0.5/seed1
SAVE_DIR=cv4_de/xlsr/cc100_30k/cp4_gp1.5_sw0.5/seed1
# SAVE_DIR=2022-05-03/18-51-26/0
DECODE_METHOD=test
DECODE_TYPE=phones
BEAM=500
LM_WEIGHT=5.0
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/de
TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_de
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/mls_en
# words or phones
if test "$DECODE_TYPE" = 'words'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.words.txt
    LM_PATH=$TEXT_DATA/../kenlm.wrd.o40003.bin
    LEXICON_PATH=$TEXT_DATA/../lexicon_filtered.lst
elif test "$DECODE_TYPE" = 'phones'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.phones.txt
    LM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin
    LEXICON_PATH=$TEXT_DATA/lexicon.phones.lst
else
    echo "Invalid decode type, please choose from {words/phones}"
fi
# TARGET_DATA=/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/asr_test.phones.txt

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

cp $TEXT_DATA/* $TASK_DATA
python w2vu_generate_mask.py --config-dir config/generate/masked --config-name ${DECODE_METHOD} \
beam=${BEAM} \
lm_weight=${LM_WEIGHT} \
lm_model=${LM_PATH} \
lexicon=${LEXICON_PATH} \
targets=${TARGET_DATA} \
post_process=silence \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.common_eval.path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/checkpoint_best.pt \
fairseq.dataset.gen_subset=${SUBSET} results_path=/home/b07502072/u-speech2speech/s2p/tmp

rm $TASK_DATA/lm* $TASK_DATA/dict* $TASK_DATA/*log $TASK_DATA/train.bin $TASK_DATA/train.idx
