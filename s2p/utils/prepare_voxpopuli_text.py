import os
import pandas as pd

file_path = '/work/b07502072/corpus/u-s2s/audio/transcribed_data/de/asr_train.tsv'
out_dir = '/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/de_new'
os.makedirs(out_dir, exist_ok=True)
out_fname='asr_train.txt'

trans_df = pd.read_csv(file_path, sep='\t')
# print(trans_df['normalized_text'][][:])

with open(os.path.join(out_dir, out_fname), 'w') as fw:
    print(f'# of sentences: {len(trans_df)}')
    for trans in trans_df['normalized_text']:
        fw.write(str(trans)+'\n')
    