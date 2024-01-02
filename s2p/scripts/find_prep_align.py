# Used for finding which raw features are pooled together after wav2vec-u preprocessing.
# Usage: python wav2vecu_scripts/find_prep_align.py CLUS128/train.src adj

# EXAMPLE
# K-mean IDs:  24  127 115 115 79  79  65  65  65  46
# Output:      0   1   0   0   0   1   0   0   0   1
import sys

file = open(sys.argv[1], 'r')
out_file = open(sys.argv[1].replace(".src", ".bds"), 'w')
adj = sys.argv[2] == "adj" if len(sys.argv) > 2 else False

for line in file:
    km_ids = line.strip().split()
    # Boundaries after merging same clusters
    b_km = [1 if km_ids[i] != km_ids[i+1] else 0 for i in range(len(km_ids)-1)]
    b_km.append(1)
    assert len(b_km) == len(km_ids)

    if adj:
        # Boundaries after adjacent pooling
        b_adj = []
        odd = True
        for i in b_km:
            if i == 1:
                if odd:
                    b_adj.append(0)
                    odd = False
                else:
                    b_adj.append(1)
                    odd = True
            else:
                b_adj.append(0)
        assert len(b_adj) == len(km_ids)
        # print(km_ids, len(km_ids))
        # print(b_km, len(b_km), sum(b_km))
        # print(b_adj, len(b_adj), sum(b_adj))
    
        print(" ".join(str(i) for i in b_adj), file=out_file)
    else:
        print(" ".join(str(i) for i in b_km), file=out_file)

file.close()
out_file.close()
