import os
import os.path as osp
import argparse
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(args):
    print(args)
    scores = np.load(args.score, allow_pickle=True)
    errors = np.load(args.error, allow_pickle=True)
    print(len(scores), len(errors))
    cosine_sim_ary = []
    def norm_npy(npy):
        return (npy - npy.min()) / (npy.max()-npy.min())
    def standardize_npy(npy):
        avg = np.mean(npy)
        std = np.std(npy)
        return (npy-avg) / std
    def draw(i, y1, y2):
        fig, ax1 = plt.subplots(1, 1)
        assert len(y1) == len(y2)
        x = np.arange(0, len(y1))
        ax1.plot(x, y1, x, y2)
        fig.savefig(f'./tmp/drawings/{i}.png')
        plt.close()
    for i, (scr, err) in tqdm(enumerate(zip(scores, errors)), total=len(scores)):
        if scr.shape != err.shape:
            print(i, scr.shape, err.shape)
            continue
        err = err.astype(scr.dtype)
        scr = norm_npy(standardize_npy(scr))
        err = norm_npy(standardize_npy(err))
        # draw(i, scr, err)
        cos = np.dot(scr, err) / (norm(scr)*norm(err))
        if cos <= 1.1:
            cosine_sim_ary.append(cos)
        
        # print(cos)
    cosine_sim_npy = np.array(cosine_sim_ary)
    cos_avg = np.average(cosine_sim_npy)
    cos_std = np.std(cosine_sim_npy)
    print(f"Avg: {cos_avg}, Std: {cos_std}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score",
        default="",
    )
    parser.add_argument(
        "--error",
        default="",
    )
    args = parser.parse_args()

    main(args)