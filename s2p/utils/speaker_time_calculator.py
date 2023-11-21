import os
import os.path as osp
from collections import defaultdict

def main():
    root = '/home/andytseng/corpus/u-s2s/audio/wo_sil/mls_en/spk1hr/manifest'
    input_tsv = osp.join(root, 'valid.tsv')
    speaker_time_dict = defaultdict(lambda: 0.0)
    sr = 16000

    tsvf = open(input_tsv, 'r')
    audio_root = tsvf.readline()
    lines = tsvf.readlines()
    tsvf.close()
    for line in lines:
        fpath, length = line.split('\t')
        secs = int(length) / sr
        speaker = fpath.split('/')[1]
        speaker_time_dict[speaker] += secs

    def parse_time(secs):
        hrs = round(secs // 3600)
        mins = round(secs % 3600 // 60)
        secs = round(secs % 60)
        return f"{hrs} hr {mins} min {secs} sec"


    for key in speaker_time_dict.keys():
        print(key, end=': ')
        print(parse_time(speaker_time_dict[key]))

    build_new_tsv = False
    if not build_new_tsv: return 0
    threshold = 3600.0
    valid_speakers = list(filter(lambda k: speaker_time_dict[k]>threshold, list(speaker_time_dict.keys())))
    print(type(valid_speakers[0]))

    with open(osp.join(root, 'spk1hr.txt'), 'w') as fw:
        for valid_spk in valid_speakers:
            fw.write(valid_spk)
            fw.write('\n')

    new_train_tsv = osp.join(root, 'train_spk1hr.tsv')
    accumulate_speaker_time_dict = defaultdict(lambda: 0.0)
    with open(new_train_tsv, 'w') as fw:
        fw.write(audio_root)
        for line in lines:
            fpath, length = line.split('\t')
            secs = int(length) / sr
            speaker = fpath.split('/')[1]
            if speaker in valid_speakers and accumulate_speaker_time_dict[speaker] < threshold:
                print(line, end='')
                fw.write(line)
                accumulate_speaker_time_dict[speaker] += secs
        
if __name__ == '__main__':
    main()