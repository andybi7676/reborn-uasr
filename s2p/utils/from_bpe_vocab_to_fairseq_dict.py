import sys

skip_count = 3
for i, line in enumerate(sys.stdin):
    line = line.strip()
    if i < skip_count:
        continue
    phn, prob = line.split('\t')
    count = 100000 -i  # dead count
    print(f"{phn} {count}")