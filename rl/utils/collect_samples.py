import os
import os.path as osp
import argparse
import editdistance

def read_file(fpath):
    with open(fpath) as f:
        lines = [l.strip() for l in f]
    return lines

def main(args):
    print(args)
    source_logit_segmented_fpath = osp.join(args.source, "logit_segmented", f"{args.split}.txt")
    source_raw_fpath = osp.join(args.source, "raw", f"{args.split}.txt")
    source_raw_bds_fpath = osp.join(args.source, "raw", f"{args.split}.bds")
    target_logit_segmented_fpath = osp.join(args.target, "logit_segmented", f"{args.split}.txt")
    target_raw_fpath = osp.join(args.target, "raw", f"{args.split}.txt")
    target_raw_bds_fpath = osp.join(args.target, "raw", f"{args.split}.bds")
    ref_phn_fpath = osp.join(args.ref_dir, f"{args.split}.phn")
    ref_wrd_fpath = osp.join(args.ref_dir, f"{args.split}.trans")
    output_fpath = osp.join(args.output_dir, f"{args.split}.txt")
    
    phn_refs = read_file(ref_phn_fpath)
    wrd_refs = read_file(ref_wrd_fpath)
    src_hyps = read_file(source_logit_segmented_fpath)
    src_raws = read_file(source_raw_fpath)
    src_bds  = read_file(source_raw_bds_fpath)
    tgt_hyps = read_file(target_logit_segmented_fpath)
    tgt_raws = read_file(target_raw_fpath)
    tgt_bds  = read_file(target_raw_bds_fpath)

    assert len(phn_refs) == len(wrd_refs) == len(src_hyps) == len(src_raws) == len(tgt_hyps) == len(tgt_raws)

    summary = []
    for i, src_hyp, src_raw, src_bd, tgt_hyp, tgt_raw, tgt_bd, phn_ref, wrd_ref in zip(range(len(phn_refs)), src_hyps, src_raws, src_bds, tgt_hyps, tgt_raws, tgt_bds, phn_refs, wrd_refs):
        src_ter = editdistance.eval(src_hyp.split(), phn_ref.split()) / len(phn_ref.split())
        tgt_ter = editdistance.eval(tgt_hyp.split(), phn_ref.split()) / len(phn_ref.split())
        new_item = (i, src_ter, tgt_ter, src_hyp, tgt_hyp, src_raw, tgt_raw, phn_ref, wrd_ref, src_bd, tgt_bd)
        summary.append(new_item)
    summary.sort(key=lambda x: x[2]-x[1], reverse=True)
    with open(output_fpath, "w") as fw:
        for item in summary:
            print(f"row_id={item[0]}\tsrc_ter={item[1]:.4f}\ttgt_ter={item[2]:.4f}\nsrc_hyp={item[3]}\ntgt_hyp={item[4]}\nphn_ref={item[7]}\nwrd_ref={item[8]}\nsrc_raw={item[5]}\ntgt_raw={item[6]}\nsrc_bds={item[9]}\ntgt_bds={item[10]}\n", file=fw, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="",
        help="source directory for the samples, including logit_segmented and raw directory",
    )
    parser.add_argument(
        "--target",
        default="target directory for easy comparison",
        help="a sample arg",
    )
    parser.add_argument(
        "--ref_dir",
        default="",
    )
    parser.add_argument(
        "--split",
        default="test-bds",
    )
    parser.add_argument(
        "--output_dir",
        default="",
    )
    args = parser.parse_args()

    main(args)