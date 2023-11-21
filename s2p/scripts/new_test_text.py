import os
from tqdm import tqdm

asr_files = ['/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/de/asr_test.tsv',  \
            ]
        # '/work/b07502072/corpus/u-s2s/audio/transcribed_data/en/asr_dev.tsv', \
        # '/work/b07502072/corpus/u-s2s/audio/transcribed_data/en/asr_train.tsv']
test_file = '/home/b07502072/u-speech2speech/s2p/data/manifest/wo_sil/voxpopuli/de/asr_test.tsv'

keys = []
with open(test_file, 'r') as tf:
    test_lines = tf.readlines()
    for li in tqdm(test_lines[1:]):
        keys.append(li.split('\t')[0].split('/')[-1].split('.')[0])
    print(keys[:5])

id_to_trans = {}
for f in asr_files:
    # with open('/home/b07502072/u-speech2speech/s2p/data/manifest/wo_sil/new_asr_trans.tsv')
    with open(f, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            splits = line.split('\t')
            id = splits[0]
            if id in keys:
                norm_trans = splits[2]
                id_to_trans[id] = norm_trans

with open('/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/de/asr_test_trans.txt', 'w') as fw:
    for key in keys:
        fw.write(f"{id_to_trans[key]}\n")
            