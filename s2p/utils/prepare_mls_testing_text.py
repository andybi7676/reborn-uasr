import os
import pandas as pd

tsv_fpath = '/work/b07502072/corpus/u-s2s/audio/wo_sil/mls_en/asr_test/manifest/asr_test.tsv'
trans_fpath = '/work/b07502072/corpus/u-s2s/text/mls_en/test/transcripts.txt'
out_dir = '/work/b07502072/corpus/u-s2s/text/mls_en/test'
os.makedirs(out_dir, exist_ok=True)
out_fname = 'asr_test.txt'

audio_trans_dict = {}

trans_fr = open(trans_fpath, 'r')
lines = trans_fr.readlines()
for l in lines:
    audio_fname, trans = l.split('\t')
    audio_trans_dict[audio_fname] = trans
trans_fr.close()

tsv_fr = open(tsv_fpath, 'r')
with open(os.path.join(out_dir, out_fname), 'w') as out_fw:
    audio_root = tsv_fr.readline()
    lines = tsv_fr.readlines()
    for l in lines:
        audio_name = l.split('\t')[0].split('/')[-1].split('.')[0]
        out_fw.write(audio_trans_dict[audio_name])

tsv_fr.close()
