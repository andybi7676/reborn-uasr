import os
import yaml
from glob import glob
import pandas as pd

SAMPLE_RATE = 16e3
# len_for_bucket_root = '/home/b07502072/u-speech2speech/s2p/data/manifest/en/shuffled'
len_for_bucket_root = '/home/andybi7676/Desktop/uasr-rl/data/audio/timit/matched/large_clean'

def get_csv_fpath(root, csv_pths):
    for path in glob(os.path.join(root, "*")):
        if ".csv" in path:
            csv_pths.append(path.split(len_for_bucket_root)[1][1:])
        else:
            get_csv_fpath(path, csv_pths)

def secs_to_time(secs):
    hrs = secs // 3600
    mins = secs % 3600 // 60
    secs = secs % 60
    return f"{hrs} hrs {mins} mins {secs} secs"

def calculate_length(csv_pth):
    sr = SAMPLE_RATE
    # total_secs = 0.0
    f_path = os.path.join(len_for_bucket_root, csv_pth)
    with open(f_path, 'r') as fr:
        lines = fr.readlines()
        total_secs = sum([ int(line.split('\t')[1])/sr for line in lines[1:] ])

    total_secs = round(total_secs)
    return total_secs

def main():
    tsv_pths = [ f.split('/')[-1] for f in glob(os.path.join(len_for_bucket_root, "*.tsv")) ]
    # get_csv_fpath(len_for_bucket_root, csv_pths)
    print(tsv_pths)
    info_dict = {}
    prev_dict = info_dict
    total_time = 0
    for tsv_pth in tsv_pths:
        total_secs = calculate_length(tsv_pth)
        total_time += total_secs
        time_repr = secs_to_time(total_secs)
        names = tsv_pth.split('/')
        prev_dict = info_dict
        for name in names:
            if ".tsv" not in name:
                if name not in prev_dict.keys():
                    prev_dict[name] = {}
                prev_dict = prev_dict[name]
            else:
                prev_dict[name.split('.')[0]] = time_repr
    # print(info_dict)
    info_dict['total'] = secs_to_time(total_time)
    print(yaml.dump(info_dict, default_flow_style=False))
    # print(len_for_bucket_root.split('/'))

if __name__ == "__main__":
    main()
