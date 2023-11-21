import argparse
import sys
import random
import os
import scipy
import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage
import os.path as osp
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json

class HierarchicalMerging():
    def __init__(self, args, **kwargs) -> None:
        pass
        self.phn_dict_fpath = args.phn_dict
        self.merge_count = args.merge_count
        self.merge_level = args.merge_level
        self.output_dir = args.output_dir
        self.load_phn_dict()
        self.current_mapping = defaultdict(lambda: None)

    def load_phn_dict(self):
        self.phn_dict = defaultdict(lambda: None)
        with open(self.phn_dict_fpath, 'r') as fr:
            for l in fr:
                items = l.split()
                if len(items) != 2:
                    print(f"Found invalid line in phn dict: items={items}")
                else:
                    phn, count = items
                    count = int(count)
                    self.phn_dict[phn] = count
            self.phn_size = len(self.phn_dict)
    
    def obtain_similarity_matrix(self): 
        # define similarity matrix for hierarchical clustering
        pass

    # def obtain_skipgram_repr(input_fpath, current_mapping=None):
    #     phn_to_skipgram_dict = defaultdict(lambda: np.zeros())
    
    def run(self):
        pass

class SkipgramHierarchicalMerging(HierarchicalMerging):
    def __init__(self, args, text_input_fpath, skip_size=1, **kwargs) -> None:
        super().__init__(args, **kwargs)
        self.text_input_fpath = text_input_fpath
        self.skip_size = skip_size
    
    def obtain_phn_vectors(self):
        self.phn_in_order = list(self.phn_dict.items())
        self.phn_in_order.sort(key=lambda x: (x[1], x[0]), reverse=True)
        # print(self.phn_in_order)
        self.phn_to_idx = {phn: idx for idx, (phn, count) in enumerate(self.phn_in_order) }
        self.phn_skipgram_count_dict = defaultdict(lambda: np.zeros((self.skip_size*2, self.phn_size))) # +- skip_size, phn_dict_size
        with open(self.text_input_fpath, 'r') as fr:
            lines = fr.readlines()
            for line in tqdm.tqdm(lines, total=len(lines), desc="Calculating skipgram from text file...", dynamic_ncols=True):
                line = line.rstrip()
                cur_phns = line.split()
                for s in range(1, self.skip_size+1):
                    skip_s_pairs = zip(cur_phns[:-s], cur_phns[s:])
                    for front_phn, end_phn in skip_s_pairs:
                        # perform +s skipgram counting
                        direction = 0
                        self.phn_skipgram_count_dict[front_phn][2*(s-1) + direction][self.phn_to_idx[end_phn]] += 1
                        # perform -s skipgram counting
                        direction = 1
                        self.phn_skipgram_count_dict[end_phn][2*(s-1) + direction][self.phn_to_idx[front_phn]] += 1
        # apply normalization to and get skipgram vectors 
        self.X_for_hier_clus = np.zeros((self.phn_size, 2*self.skip_size*self.phn_size))
        for i, (phn_x, _) in enumerate(self.phn_in_order):
            skipgram_count_npy = self.phn_skipgram_count_dict[phn_x]
            # norm_factor = skipgram_count_npy.sum()
            norm_factor = np.linalg.norm(skipgram_count_npy) # l2 norm
            phn_x_vector = (skipgram_count_npy / norm_factor).flatten()
            self.X_for_hier_clus[i] = phn_x_vector

    def perform_hierarchical_clustering(self):
        self.Z_linkage = linkage(self.X_for_hier_clus, method='single', metric='cosine', optimal_ordering=False) # perform basic hierarchical clustering
        print(self.Z_linkage)
        fig, ax = plt.subplots(1, 1, figsize=(25, 10))
        fig.set_dpi(400)
        dn = dendrogram(self.Z_linkage, labels=[(i, phn) for i, (phn, _) in enumerate(self.phn_in_order)])
        ax.set_xlabel('phn classes', fontsize=15)
        ax.tick_params(axis='both', labelsize=12)
        dendrogram_output_fpath = osp.join(self.output_dir, "dendrogram.png")
        plt.savefig(dendrogram_output_fpath)
        cur_merge_count = 0
        for link in self.Z_linkage:
            phn_idx_a, phn_idx_b, distance, level = map(lambda x: int(x), link)
            if level > self.merge_level:
                continue
            from_phn = self.phn_in_order[phn_idx_b][0]
            to_phn   = self.phn_in_order[phn_idx_a][0]
            self.current_mapping[from_phn] = to_phn
            cur_merge_count += 1
            if cur_merge_count == self.merge_count:
                break
        mapping_output_fpath = osp.join(self.output_dir, "phn_dict.map")
        with open(mapping_output_fpath, 'w') as mfw:
            json.dump(self.current_mapping, fp=mfw, indent=4, ensure_ascii=False)
    
    def calculate_similarity_matrix_and_visualize(self):
        # calculate similarity matrix (dot products of skipgram distributions)
        # self.phn_similarity_matrix = np.zeros((self.phn_size, self.phn_size))
        # for i, (phn_x, _) in enumerate(self.phn_in_order):
        #     for j, (phn_y, _) in enumerate(self.phn_in_order):
        #         skipgram_x = self.phn_skipgram_count_dict[phn_x]
        #         skipgram_y = self.phn_skipgram_count_dict[phn_y]
        #         similarity = (skipgram_x.flatten())@(skipgram_y.flatten())
        #         self.phn_similarity_matrix[i][j] = similarity
        # print(self.phn_similarity_matrix)
        # visualizing similarity matrix
        # fig, ax = plt.subplots()
        # ax.imshow(self.phn_similarity_matrix, extent=(0, self.phn_size, 0, self.phn_size))
        # ax.set_xlim(0, self.phn_size)
        # ax.set_ylim(0, self.phn_size)
        # ax.set_aspect('equal')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.title(f"Skipgram similarity matrix visualization (skip_size={self.skip_size})")
        # plt.savefig(f"./skipgram_similarity_matrix(skip_size={self.skip_size},norm_factor=sum).png")
        pass # deprecated and use scipy hierarchy functionalities

    def run(self):
        self.obtain_phn_vectors()
        self.perform_hierarchical_clustering()


def main():
    parser = get_parser()
    args = parser.parse_args()
    skipgramMerger = SkipgramHierarchicalMerging(args, **vars(args))
    skipgramMerger.run()

    # phn_to_skipgram_dict = obtain_skipgram_repr(args.input, current_mapping=current_mapping, skip_size=args.skip_size)


def get_parser():
    parser = argparse.ArgumentParser(
        description="generate bpe dict for advanced wav2vec-U"
    )
    # fmt: off
    parser.add_argument('--phn_dict', default='', required=True)
    parser.add_argument('--merge_count', default=10, help='merge size for skipgram merging')
    parser.add_argument('--merge_level', default=2, help='merge level for hierarchical clustering mapping')
    parser.add_argument('--text_input_fpath', help='input file for spm to train and encode with')
    parser.add_argument('--skip_size', default=1)
    parser.add_argument('--output_dir', default='.')
    return parser

if __name__ == "__main__":
    main()