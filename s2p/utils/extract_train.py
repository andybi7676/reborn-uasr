import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Extract train set into smaller set.')
parser.add_argument('--input_dir', '-i', default='./')
parser.add_argument('--split', '-s', default='train')
parser.add_argument('--output_dir', '-o', default='./smaller')
args = parser.parse_args()
if args.input_dir != './' and args.output_dir == './smaller':
    args.output_dir = osp.join(args.input_dir, 'smaller')

os.makedirs(args.output_dir, exist_ok=True)

in_npy_file = osp.join(args.input_dir, args.split+'.npy')
in_length_file = osp.join(args.input_dir, args.split+'.lengths')
in_tsv_file = osp.join(args.input_dir, args.split+'.tsv')
out_npy_file = osp.join(args.output_dir, args.split+'_smaller.npy')
out_length_file = osp.join(args.output_dir, args.split+'_smaller.lengths')
out_tsv_file = osp.join(args.output_dir, args.split+'_smaller.tsv')
ratio = 0.3

train = np.load(in_npy_file)

total_l = train.shape[0]
ratio_l = round(total_l*ratio)
print(f"total_length: {total_l}, ratio_length: {ratio_l}")
out_l = 0
l_count = 0
with open(in_length_file, 'r') as fr_l:
    for line in fr_l:
        out_l += int(line.strip())
        l_count += 1
        if out_l > ratio_l: break

print(f"out_length: {out_l}, line_count: {l_count}")
train_smaller = train[:out_l]
np.save(out_npy_file, train_smaller)
with open(out_length_file, 'w') as fw_l:
    with open(in_length_file, 'r') as fr_l:
        i = 0
        for l in fr_l:
            if i == l_count: break
            fw_l.write(l)
            i += 1

with open(out_tsv_file, 'w') as fw_l:
    with open(in_tsv_file, 'r') as fr_l:
        i = 0
        for l in fr_l:
            if i == l_count+1: break
            fw_l.write(l)
            i += 1