rm core.* 
PREFIX=w2v_unsup_gan_xp
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/librispeech/large_clean_mfa/precompute_pca512_cls128_mean
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/ls_wo_lv/prep_g2p/phones/train_all
KENLM_PATH=$TEXT_DATA/../lm.phones.filtered.04.bin
EXP_NAME=ls_100h/large_clean_mfa/ls_wo_lv_g2p_all
export HYDRA_FULL_ERROR=1

# if [ -d ./multirun/$EXP_NAME ] 
# then
#     echo "Directory $EXP_NAME already exists." 
#     exit 9999 # die with error code 9999
# fi
echo "Exp name(save dir): $EXP_NAME"
mkdir -p ./multirun/$EXP_NAME
cp /home/b07502072/u-speech2speech/s2p/multi_train.sh ./multirun/$EXP_NAME

for gp in 1.5; do
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