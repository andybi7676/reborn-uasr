import os
import os.path as osp

correct_file = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_de/xlsr/cv_wiki_3k/cp4_gp2.0_sw0.5/seed3/combined_asr_mt_result.test.txt"
wrong_file = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_de/xlsr/cv_wiki_3k/cp4_gp2.0_sw0.5/seed3/translation.tsv"
out_file = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_de/xlsr/cv_wiki_3k/cp4_gp2.0_sw0.5/seed3/wfid_to_cfid.tsv"

sent_to_wfid = {}
cfid_to_sent = {}
with open(correct_file, 'r') as refr, open(
    wrong_file, 'r'
) as wfr:
    for l in wfr:
        items = l.strip().split('\t')
        if len(items) < 2: continue
        wfid = items[0].split('.')[0]
        sent = items[1]
        sent_to_wfid[sent] = wfid
    refr.readline()
    for l in refr:
        items = l.strip().split('|')
        cfid = items[0]
        sent = items[-1]
        cfid_to_sent[cfid] = sent

with open(out_file, 'w') as fw:
    print(f"wfid\tcfid\tsent", file=fw)
    for cfid, sent in cfid_to_sent.items():
        try:
            wfid = sent_to_wfid[sent]
            print(f"{wfid}\t{cfid}\t{sent}", file=fw)
        except:
            print(f"{cfid} unsucessed.")



    