#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from copy import deepcopy
from scipy.signal import lfilter

import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa as lb
import os.path as osp
import multiprocessing as mp
sys.path.append("/home/b07502072/u-speech2speech/s2p/rVADfast")
import speechproc
import random
import time

root=""
stride = 160

def get_parser():
    parser = argparse.ArgumentParser(description="compute vad segments")
    parser.add_argument(
        "--manifest",
        "-m",
        help="path of the manifest of sound files.(.tsv)",
        required=True,
    )
    parser.add_argument(
        "--rvad-home",
        "-r",
        help="path to rvad home (see https://github.com/zhenghuatan/rVADfast)",
        required=True,
    )
    parser.add_argument(
        "--start",
        '-s',
        type=int,
        default=1,
        help="index to start vad segmentation"
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=-1,
        help="index to end vad segmentation"
    )

    return parser

# def test(fpath):
#     global stride
#     time.sleep(random.randint(1, 3))
#     # pbar.update(1)
#     return osp.join(root, fpath.split()[0])

def rvad(fpath):
    path = osp.join(root, fpath.split()[0])
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    data, fs = lb.load(path, sr=16_000, dtype='float64')
    assert fs == 16_000, "sample rate must be 16khz"
    ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)

    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    # --pass 1--
    noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )

    # sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )

    start = None
    vad_segs = []
    for i, v in enumerate(vad_seg):
        if start is None and v == 1:
            start = i * stride
        elif start is not None and v == 0:
            vad_segs.append((start, i * stride))
            start = None
    if start is not None:
        vad_segs.append((start, len(data)))

    return " ".join(f"{v[0]}:{v[1]}" for v in vad_segs) # will output to train.vads. Each line looks like => start_1:end_1 start_2:end_2 start_3:end_3


def main():
    parser = get_parser()
    args = parser.parse_args()

    sys.path.append(args.rvad_home)

    lines = []
    with open(args.manifest, 'r') as mfr:
        lines = mfr.readlines()
    # lines = sys.stdin.readlines()
    global root
    root = lines[0].rstrip()

    i_start = max(1, min(len(lines)-1, args.start))
    i_end = len(lines) if args.end==-1 else max(1, min(len(lines), args.end))
    lines = lines[i_start:i_end]

    pool = mp.Pool()
    vad_segs = list(tqdm(pool.imap(rvad, lines), total=len(lines)))

    with open(args.manifest.split('.')[0]+f"_{i_start}-{i_end}.vads", 'w') as vads_outf:
        for vad_seg in vad_segs:
            vads_outf.write(vad_seg+'\n')
    print(vad_segs[0:10])


if __name__ == "__main__":
    main()