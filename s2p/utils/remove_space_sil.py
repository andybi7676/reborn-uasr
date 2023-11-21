import sys

for line in sys.stdin:
    line = line.strip().replace("<SIL>", "")
    line = line.replace(" ", "")
    print(line)