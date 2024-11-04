source ~/.bashrc
conda activate wav2vecu

# use the first GPU
CUDA_VISIBLE_DEVICES=0 bash _train_ls_new_wavlm_yy.sh &
CUDA_VISIBLE_DEVICES=1 bash _train_ls_new_wavlm_yy_2.sh &

wait