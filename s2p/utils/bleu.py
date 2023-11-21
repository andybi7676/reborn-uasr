from sacrebleu.metrics import BLEU
ref_fpath = "./utils/test.en"
hyp_fpath = "./utils/de-en.20"

refs = []
hyps = []
input_fpath = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_de/xlsr/cv_wiki_3k/cp4_gp2.0_sw0.5/seed3/combined_asr_mt_result.txt"
with open(ref_fpath, 'r') as refr:
    for l in refr:
        refs.append(l.strip())
with open(hyp_fpath, 'r') as hypr:
    for l in hypr:
        hyps.append(l.strip())

bleu_metric = BLEU()
print(refs[:5])
print(hyps[:5])
corpus_bleu = bleu_metric.corpus_score(hyps, [refs])
print(corpus_bleu)

mt_refs = []
mt_hyps = []
with open(input_fpath, 'r') as fr:
    fr.readline()
    for line in fr:
        _, _, _, mt_ref, mt_hyp = line.strip().split('|')
        mt_refs.append(mt_ref)
        mt_hyps.append(mt_hyp)
print(mt_refs[:5])
print(mt_hyps[:5])
corpus_bleu = bleu_metric.corpus_score(mt_hyps, [mt_refs])
print(corpus_bleu)

