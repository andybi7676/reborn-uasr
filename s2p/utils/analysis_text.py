import os

f_path = '/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/asr_train.txt'
f_path = '/work/b07502072/corpus/u-s2s/text/wiki/en/wiki_2_7.txt'
f_path = '/work/b07502072/corpus/u-s2s/text/wiki/en/wiki_1_7/wiki_1_7.txt'
f_path = '/work/b07502072/corpus/test_kaggle/u-s2p/text/wiki_3/phones/lm.phones.filtered.txt'
f_path = '/work/b07502072/corpus/test_kaggle/u-s2p/text/wiki_1-5/phones/lm.phones.filtered.txt'
f_path = '/work/b07502072/corpus/test_kaggle/u-s2p/text/wiki_2/phones/lm.phones.filtered.txt'
f_path = '/work/b07502072/corpus/test_kaggle/u-s2p/text/marc/phones/lm.phones.filtered.txt'
f_path = '/work/b07502072/corpus/test_kaggle/u-s2p/text/voxpopuli_trans/phones/lm.phones.filtered.txt'
f_path = '/work/b07502072/corpus/u-s2s/text/8M_wiki/de/train.de'
f_path = '/work/b07502072/corpus/u-s2s/text/cv4_others/de/other.uniq.txt'
f_path = '/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_de/asr_test.trans.txt'

with open(f_path, 'r') as fr:
    total_line = 0
    words_counts = []
    for line in fr:
        total_line += 1
        words_counts.append(len(line.split(' ')))
    total_words = sum(words_counts)
    mean = total_words / total_line
    std = (sum([(wc-mean)**2 for wc in words_counts]) / total_line)**0.5

    print(f"file name: {f_path}")
    print(f"total lines: {total_line}")
    print(f"mean of words: {mean}")
    print(f"std of words: {std}")
