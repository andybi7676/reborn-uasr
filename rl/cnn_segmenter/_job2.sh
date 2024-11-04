source ~/.bashrc
conda activate wav2vecu

# # use the first GPU
# CUDA_VISIBLE_DEVICES=0 bash _train_ls_new_nn.sh &
# # use the second GPU
# CUDA_VISIBLE_DEVICES=1 bash _train_ls_new_ny.sh &

CUDA_VISIBLE_DEVICES=0 bash _train_mls_yy_5.sh &
CUDA_VISIBLE_DEVICES=1 bash _train_mls_yy_6.sh &

wait