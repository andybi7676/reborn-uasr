import sys

for l in sys.stdin:
    l = l.strip()
    l = l.replace(" sil", "")
    l = l.replace("sil ", "")
    l = l.replace("dx", "t")
    print(l)