import os
import random

trans_fpath = '/work/b07502072/corpus/u-s2s/text/mls_en/train/transcripts.txt'
threshold = 100000
out_fpath = '/work/b07502072/corpus/u-s2s/text/mls_en/train/plain_text.txt'


trans_fr = open(trans_fpath, 'r')
trans = [line.split('\t')[-1] for line in trans_fr.readlines()]
trans_fr.close()

with open(out_fpath, 'w') as out_fw:
    choices = random.sample(range(len(trans)), threshold)
    for c in choices:
        out_fw.write(trans[c])
