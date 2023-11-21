rm core.*
PREFIX=w2v_unsup_gan_xp
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/es_feats/cv4/mhubert/precompute_pca512_cls128_mean_pooled
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/libriTTS/precompute_pca512_cls128_mean_pooled
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/LJ_speech/large_clean/precompute_pca512_cls128_mean_pooled
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/librispeech/large_clean/precompute_pca512_cls128_mean_pooled
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/cvss-t/large_clean
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/de_feats/cv4_10h/xlsr/precompute_pca512_cls128_mean_pooled
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep/phones/train_3k
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/es/prep/phones/train_all
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/fr/prep_skhc/phones/train_300k
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/ls_wo_lv/prep_g2p/phones/train_all
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cc100/de/w2vu/phones/train_30k
KENLM_PATH=$TEXT_DATA/../lm.phones.filtered.04.bin
EXP_NAME=ls_100h/large_clean/ls_wo_lv_g2p_all
# EXP_NAME={voxpopuli_{en,de}, mls_{en,de}}/{xlsr, large_{clean,noisy,vox}}/{vox_trans,wiki_{1,2,3,1-5},mls_trans}
export HYDRA_FULL_ERROR=1

# if [ -d ./multirun/$EXP_NAME ] 
# then
#     echo "Directory $EXP_NAME already exists." 
#     exit 9999 # die with error code 9999
# fi
echo "Exp name(save dir): $EXP_NAME"
mkdir -p ./multirun/$EXP_NAME
cp /home/b07502072/u-speech2speech/s2p/multi_train.sh ./multirun/$EXP_NAME

# cp="4"
# seed="3"
for gp in 1.5 2.0; do
    for cp in 4; do
        for seed in 1 2 3; do
            # PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            python $FAIRSEQ_ROOT/fairseq_cli/hydra_train.py \
                -m --config-dir config/gan \
                --config-name w2vu \
                hydra.sweep.dir=multirun/${EXP_NAME} \
                task.data=${TASK_DATA} \
                task.text_data=${TEXT_DATA} \
                task.kenlm_path=${KENLM_PATH} \
                dataset.num_workers=0 \
                common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
                model.code_penalty=$cp model.gradient_penalty=$gp \
                model.smoothness_weight=0.5 common.seed=${seed} \
                distributed_training.distributed_world_size=1 \
                optimization.max_update=120000 \
                +description=${EXP_NAME}
        done
    done
done
wait 