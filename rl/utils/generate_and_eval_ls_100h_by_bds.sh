set -eu

boundary_dir=../../data/ls_100h_new/ll60k/LUKE
generator_ckpt=../../s2p/multirun/en_ls100h/ll60k/ls860_unpaired_all/best_unsup/checkpoint_best.pt
output_dir=$boundary_dir/eval_on_oracle_output
config_name=en_ls100h
feats_dir=../../data/ls_100h_new/ll60k/precompute_pca512
golden_dir=../../data/ls_100h_new/labels
all_splits="test-bds"

for split in $all_splits; do
    echo "Processing $split..."
    boundary_fpath=$boundary_dir/$split.bds
    # logit segmented
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --boundary_fpath $boundary_fpath \
        --output_dir $output_dir/logit_segmented \
        --split $split
    # raw
    python generate_w2vu_segmental_results.py \
        --config $config_name \
        --feats_dir $feats_dir \
        --generator_ckpt $generator_ckpt \
        --boundary_fpath $boundary_fpath \
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
    python eval_boundaries.py --hyp $output_dir/raw/$split.bds --ref $feats_dir/../GOLDEN/$split.bds >> $output_dir/result.txt
    echo "" >> $output_dir/result.txt
    tail -n 13 $output_dir/result.txt
done
