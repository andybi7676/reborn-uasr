import os
import json
import os.path as osp
from per import cal_per
import editdistance
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import BLEU

input_fpath = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_fr/xlsr/cv_wiki_new_reduced_sil_0-5_300k/cp4_gp2.0_sw0.5/seed1/combine_uasr_mt_words.test.tsv"
threshold = 0.2
out_dir = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_fr/xlsr/cv_wiki_new_reduced_sil_0-5_300k/cp4_gp2.0_sw0.5/seed1"
threshold_str = str(threshold).replace(".", "-")
select_metric = "wer"
out_fname = f"good_examples_{select_metric}_{threshold_str}.json"
out_fpath = osp.join(out_dir, out_fname)
analyze = True

def main():
    # print(editdistance.eval("the program must be run as an administrator".split(), "the program must be executed as an administrator".split()))
    with open(input_fpath, 'r') as fr:
        fr.readline()
        fids = []
        asr_refs = []
        asr_hyps = []
        mt_refs = []
        mt_hyps = []
        good_examples = {
            "threshold": threshold,
            "examples": []
        }
        if analyze: 
            analysis = {}
            analysis['wer_count_in_intervals'] = defaultdict(lambda: 0)
            analysis['bleu_count_in_intervals'] = defaultdict(lambda: 0)
        
        for line in fr:
            # fid, asr_ref, asr_hyp, mt_ref, mt_hyp = line.strip().split('|')
            fid, asr_hyp, asr_ref, mt_hyp, mt_ref = line.strip().split('|')
            mt_hyp_words = mt_hyp.split()
            mt_ref_words = mt_ref.split()
            mt_refs.append(mt_ref)
            mt_hyps.append(mt_hyp)
            wer = editdistance.eval(mt_hyp_words, mt_ref_words) / len(mt_ref_words)
            bleu = sentence_bleu([mt_ref_words], mt_hyp_words, weights=[0.25, 0.25, 0.25, 0.25])
            
            if analyze:
                if wer > 1.0: wer = 1.0
                wer_interval_str = f"{wer:.1f}"
                analysis['wer_count_in_intervals'][wer_interval_str] += 1
                bleu_interval_str = f"{bleu:.1f}"
                analysis['bleu_count_in_intervals'][bleu_interval_str] += 1
            
            if eval(select_metric) < threshold:
                new_good_example = {
                    "fid": fid,
                    "asr_ref": asr_ref,
                    "asr_hyp": asr_hyp,
                    "mt_ref": mt_ref,
                    "mt_hyp": mt_hyp,
                }
                good_examples["examples"].append(new_good_example)
        good_examples["total"] = len(good_examples["examples"])
        if analyze:
            bleu_metric = BLEU()
            corpus_bleu = bleu_metric.corpus_score(mt_hyps, [mt_refs])
            analysis['corpus_bleu'] = str(corpus_bleu)
            print(corpus_bleu)
    with open(out_fpath, 'w') as fw:
        # json.dump(good_examples, fw, indent=2)
        print(json.dumps(good_examples, indent=2, ensure_ascii=False), file=fw)
    if analyze:
        with open(osp.join(out_dir, f"analysis.json"), 'w') as fw:
            json.dump(analysis, fw, indent=2, ensure_ascii=False, sort_keys=True)
if __name__ == "__main__":
    main()   
    