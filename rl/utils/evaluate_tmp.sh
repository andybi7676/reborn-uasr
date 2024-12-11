source ~/.bashrc                                                                                                                                                       
conda activate wav2vecu                                                            
                                                                                   
# export PYTHONPATH=$PYTHONPATH:/home/r11921042/uasr-rl/fairseq                                                                                                        
                                                                                                                                                                       
data_dir=/livingrooms/public/uasr_rl                                               
reborn_dir=/home/dmnph/reborn-uasr                                                 
output_dir=/home/dmnph/reborn_output                                               
                                                                                   
                                                                                   
source ${reborn_dir}/path.sh                 

python eval_boundaries.py --hyp  ../cnn_segmenter/output/rl_agent/timit_matched/timit_matched_pplNorm1.0_tokerr0.2_lenratio0.0_lr1e-4_epoch1000_seed3_postITER1/raw/all-test.bds --ref /livingrooms/public/uasr_rl/audio/timit/matched/large_clean/GOLDEN/all-test.bds
