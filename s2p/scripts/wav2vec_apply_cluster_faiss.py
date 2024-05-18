#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np
import tqdm
import torch
import sys

import faiss
import torch.nn.functional as F

from wav2vec_cluster_faiss import parse_faiss_specs
from wav2vec_extract_features import Wav2VecFeatureReader, HubertFeatureReader


def get_parser():
    parser = argparse.ArgumentParser(description="apply clusters")
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='split to process', required=True)
    parser.add_argument('--labels', help='split to process', default="phn")
    parser.add_argument('--path', help='path to pca and centroids', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec model (if using wav2vec features)', required=True)
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=14)
    parser.add_argument('--max-tsz', type=int, help='batch kmeans up to this much', default=14)
    # fmt: on

    return parser


def get_iterator(args):
    label_path = osp.join(args.data, f"{args.split}.{args.labels}")
    if osp.exists(label_path):
        lp = open(label_path, "r")
    else:
        lp = None
    f_fpath = osp.join(args.data, f"{args.split}.npy")
    len_fpath = osp.join(args.data, f"{args.split}.lengths")
    tsv_fpath = osp.join(args.data, f"{args.split}.tsv")
    if osp.exists(f_fpath) and osp.exists(len_fpath):
        feats = np.load(f_fpath)
        with open(len_fpath, "r") as f_len, open(tsv_fpath, "r") as f_tsv:
            lengths = [int(x.strip()) for x in f_len.readlines()]
            # skip root
            root = f_tsv.readline().strip()
            fnames = [x.strip() for x in f_tsv.readlines()]
        total_frames = len(feats)
        total_lengths = sum(lengths)
        assert total_frames == total_lengths, f"{total_frames} != {total_lengths}"
        assert len(lengths) == len(fnames), f"{len(lengths)} != {len(fnames)}"
        num = len(lengths)
        print("Use precomputed features!")

        def iterate():
            offset = 0
            for i, (length, fname) in enumerate(zip(lengths, fnames)):
                _feats = feats[offset : offset + length]
                offset += length
                yield torch.from_numpy(_feats).cuda(), fname, None

        return iterate, num, root

    with open(osp.join(args.data, f"{args.split}.tsv"), "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [line.rstrip() for line in lines if len(line) > 0]

        if lp is not None:
            lbls = [line.rstrip() for line in lp]
        else:
            lbls = [None] * len(files)

        num = len(files)
        if "hubert" in args.checkpoint:
            print("Use HubertFeatureReader")
            reader = HubertFeatureReader(args.checkpoint, args.layer)
        else:
            reader = Wav2VecFeatureReader(args.checkpoint, args.layer)
            
        def iterate():
            for fname, lbl in zip(files, lbls):
                file = osp.join(root, fname.split("\t")[0])
                feats = reader.get_feats(file)
                yield feats.data, fname, lbl

        return iterate, num, root


def main():
    parser = get_parser()
    args = parser.parse_args()

    spec = osp.basename(args.path)

    try:
        faiss_spec = parse_faiss_specs(spec.rstrip("/"))[0]
    except:
        print(spec)
        raise

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca:
        A = torch.from_numpy(np.load(osp.join(args.path, "pca_A.npy"))).cuda()
        b = torch.from_numpy(np.load(osp.join(args.path, "pca_b.npy"))).cuda()
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(osp.join(args.path, "centroids.npy"))
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    res = faiss.StandardGpuResources()
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1])
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    )
    faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    faiss_index.add(centroids)

    generator, num, root = get_iterator(args)
    iterator = generator()

    had_labels = False
    label_path = osp.join(args.path, f"{args.split}.{args.labels}")

    with torch.no_grad():
        with open(osp.join(args.path, f"{args.split}.src"), "w") as fp, open(
            osp.join(args.path, f"{args.split}.tsv"), "w"
        ) as pp, open(label_path, "w") as lp:
            print(root, file=pp)
            for f, fname, lbl in tqdm.tqdm(iterator, total=num):
                if faiss_spec.pca:
                    f = torch.mm(f, A) + b
                if faiss_spec.norm:
                    f = F.normalize(f, p=2, dim=-1)

                f = f.cpu().numpy()

                _, z = faiss_index.search(f, 1)

                print(" ".join(str(x.item()) for x in z), file=fp)
                print(fname, file=pp)

                if lbl is not None:
                    print(lbl, file=lp)
                    had_labels = True
    if not had_labels:
        os.remove(label_path)


if __name__ == "__main__":
    main()
