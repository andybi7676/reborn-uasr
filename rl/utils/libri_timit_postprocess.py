import sys

for l in sys.stdin:
    l = l.strip()
    l = l.replace("ao", "aa")
    l = l.replace(" sil", "")
    l = l.replace("sil ", "")
    print(l)