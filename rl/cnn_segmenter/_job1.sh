source ~/.bashrc
conda activate wav2vecu

# use the first GPU
# CUDA_VISIBLE_DEVICES=0 bash _train_ls_new_yy.sh &
# use the second GPU
# CUDA_VISIBLE_DEVICES=1 bash _train_ls_new_yn.sh &
# # use the third GPU
# CUDA_VISIBLE_DEVICES=2 bash _train_ls_new_ny.sh &
# # use the fourth GPU
# CUDA_VISIBLE_DEVICES=3 bash _train_ls_new_nn.sh &

# CUDA_VISIBLE_DEVICES=0 bash _train_mls_yy.sh &
CUDA_VISIBLE_DEVICES=0 bash _train_ls_new_yy_standardPPL.sh &
CUDA_VISIBLE_DEVICES=1 bash _train_ls_new_yy_standardPPL_2.sh &
# CUDA_VISIBLE_DEVICES=1 bash _train_mls_yy_2.sh &

wait