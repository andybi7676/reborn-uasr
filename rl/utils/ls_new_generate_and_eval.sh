segmenter_dir=../cnn_segmenter/output/rl_agent/ls_en/LS_EN_pplNorm1.0_tokerr0.2_lenratio0.0_lr1e-4_epoch40_seed3_postITER1
segmenter_ckpt=$segmenter_dir/rl_agent_segmenter_best.pt
generator_ckpt=../../s2p/multirun/en_ls100h/ll60k_postITER1/ls860_unpaired_all/best_unsup/checkpoint_best.pt
output_dir=$segmenter_dir
config_name=en_ls100h
feats_dir=../../data/ls_100h_new/ll60k/precompute_pca512
golden_dir=../../data/ls_100h_new/labels
all_splits="train valid valid_small test dev-other test-other"

for split in $all_splits; do
    echo "Processing $split..."
    # logit segmented
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --segmenter_ckpt $segmenter_ckpt \
        --output_dir $output_dir/logit_segmented \
        --split $split
    # raw
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --segmenter_ckpt $segmenter_ckpt \
        --no_logit_segment \
        --output_dir $output_dir/raw \
        --split $split
    # postprocess boundaries
    python postprocess_boundaries.py \
        --bds_fpath $output_dir/raw/$split.bds \
        --raw_output_fpath $output_dir/raw/$split.txt \
        --new_bds_fpath $output_dir/raw/$split.postprocessed.bds \
        --length_fpath $feats_dir/$split.lengths

    echo "# $split" >> $output_dir/result.txt
    python eval_results.py --hyp $output_dir/logit_segmented/$split.txt --ref $golden_dir/$split.phn >> $output_dir/result.txt
    # python eval_boundaries.py --hyp $output_dir/raw/$split.postprocessed.bds --ref ../../data/audio/timit/matched/large_clean/GOLDEN/$split.bds >> $output_dir/result.txt
    echo "" >> $output_dir/result.txt
    tail -n 13 $output_dir/result.txt
done

ITER1_bds_dir=$feats_dir/../ITER2
postITER1_bds_dir=$feats_dir/../postITER2
mkdir -p $ITER1_bds_dir
mkdir -p $postITER1_bds_dir
for split in $all_splits; do
    echo "Processing $split for bds -> ids..."
    cp $output_dir/raw/$split.bds $ITER1_bds_dir/$split.bds
    cp $output_dir/raw/$split.postprocessed.bds $postITER1_bds_dir/$split.bds
    python bds_to_ids.py --bds_fpath $ITER1_bds_dir/$split.bds      --ids_fpath $ITER1_bds_dir/$split.src       --length_fpath $feats_dir/$split.lengths
    python bds_to_ids.py --bds_fpath $postITER1_bds_dir/$split.bds  --ids_fpath $postITER1_bds_dir/$split.src   --length_fpath $feats_dir/$split.lengths
done
