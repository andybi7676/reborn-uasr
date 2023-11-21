import os
import string

puncs = string.punctuation.replace('\'', '')
print(puncs)
in_fname = 'asr_test_trans.txt'
out_fname = 'asr_test.words.txt'
with open(out_fname, 'w') as fw:
    with open(in_fname, 'r') as fr:
        for line in fr.readlines():
            line = line.strip().translate(str.maketrans('', '', puncs))
            # print(line)
            fw.write(line+'\n')