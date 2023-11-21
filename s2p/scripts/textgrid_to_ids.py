import os
import glob
import tqdm
import numpy as np
import os.path as osp
import argparse
import textgrid

MAX_SEG_ID = 500

# tg_fpath = "/work/b07502072/corpus/u-s2s/audio/LibriSpeech/no_sil/mfa/train-clean-100/19/198/19-198-0000.TextGrid"
# tg = textgrid.TextGrid.fromFile(tg_fpath)
# print(tg[0].name)
# print(tg[0][0])
# print(tg[0][0].minTime)
# print(tg[0][0].maxTime)
# print(tg[0][0].mark)
# print(type(tg[0][0].minTime))

def get_fids(tsv_fpath):
    with open(tsv_fpath, 'r') as fr:
        _ = fr.readline()
        fids = [l.strip().split('\t')[0].split('/')[-1].split('.')[0] for l in fr]
    return fids

def get_boundary_from_textgrid(mfa_fpath):
    tg = textgrid.TextGrid.fromFile(mfa_fpath)
    phns_tg = tg[1]
    boundary = [0.0]
    for seg in phns_tg:
        s, e = seg.minTime, seg.maxTime
        if s != boundary[-1]:
            boundary.append(s)
        if e != s:
            boundary.append(e)
    return np.array(boundary)

def get_boundaries_with_fids(mfa_dir, fids):
    search_pattern = osp.join(mfa_dir, "**", "*.TextGrid")
    mfa_fpaths = glob.glob(search_pattern, recursive=True)
    fid_to_mfa_fpath = {}
    for mfa_fpath in mfa_fpaths:
        fid = mfa_fpath.split('/')[-1].split('.')[0]
        fid_to_mfa_fpath[fid] = mfa_fpath
    # print(fid_to_mfa_fpath) 
    # fids = fids[:10]
    boundaries = [get_boundary_from_textgrid(fid_to_mfa_fpath[fid]) for fid in tqdm.tqdm(fids, total=len(fids))]
    
    return boundaries

def get_target_lengs(lengs_fpath):
    lengs = []
    with open(lengs_fpath, 'r') as fr:
        for l in fr:
            leng = int(l.strip())
            lengs.append(leng)
    return lengs

def generate_segments_and_write_to_disk(out_fpath, boundaries, target_lengs):
    # target_lengs = target_lengs[:10]
    assert len(boundaries) == len(target_lengs), f"{len(boundaries)} {len(target_lengs)}"
    with open(out_fpath, 'w') as fw:
        for bds, leng in tqdm.tqdm(zip(boundaries, target_lengs), total=len(target_lengs)):
            resample_ratio = (leng) / bds[-1]
            resampled_bds = bds * resample_ratio
            rounded_bds = (np.rint(resampled_bds)).astype(int)
            
            segments = np.zeros(leng, dtype=int)
            for i, (ss, se) in enumerate(zip(rounded_bds[:-1], rounded_bds[1:])):
                segments[ss: se] = i % MAX_SEG_ID
            segments_repr = ' '.join(map(lambda x: str(x), segments))
            fw.write(segments_repr + "\n")

def main(args):
    fids = get_fids(args.tsv)
    boundaries = get_boundaries_with_fids(args.mfa_dir, fids)
    target_lengs = get_target_lengs(args.lengs_fpath)
    os.makedirs(args.out_dir, exist_ok=True)
    out_fpath = osp.join(args.out_dir, f"{args.split}.src")
    generate_segments_and_write_to_disk(out_fpath, boundaries, target_lengs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        default="",
        help="a sample arg",
    )
    parser.add_argument(
        "--mfa_dir"
    )
    parser.add_argument(
        "--out_dir"
    )
    parser.add_argument(
        "--split"
    )
    parser.add_argument(
        "--lengs_fpath"
    )
    args = parser.parse_args()

    main(args)