source ~/.bashrc
conda activate wav2vecu

# use the first GPU
CUDA_VISIBLE_DEVICES=0 bash _train_ls_new_yy.sh &

wait