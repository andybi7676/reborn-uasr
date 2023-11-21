import os
import random

random.seed(1337)
input_fpath = '/home/b07502072/u-speech2speech/s2p/data/manifest/wo_sil/voxpopuli/en/all/train.tsv'
output_fpath = '/home/b07502072/u-speech2speech/s2p/data/manifest/wo_sil/voxpopuli/en/smaller_train/train_shuffled.tsv'

with open(input_fpath, 'r') as fr:
    contents = fr.readlines()
    audio_root = contents[0]
    audio_fpath = contents[1:]
    print("original first ten lines: ")
    for i in range(10):
        print(audio_fpath[i], end='')
    random.shuffle(audio_fpath)
    print("After shuffling first ten lines: ")
    for i in range(10):
        print(audio_fpath[i], end='')
    with open(output_fpath, 'w') as fw:
        fw.write(audio_root)
        for audio in audio_fpath:
            fw.write(audio)
    