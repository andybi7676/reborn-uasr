import os
import os.path as osp

# src_dir = "/work/b07502072/corpus/u-s2s/audio/de_feats/cv4/xlsr/precompute_pca512_cls128_mean_pooled"
# gold_dir = "/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_de"
# hyp_fpath = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_fr/xlsr/cv_wiki_sil_0-5/cp4_gp2.0_sw0.5/seed2/ckpt_best_asr_test_viterbi_500-5.0.phones/asr_test.txt"
# hyp_sil_fpath = "/home/b07502072/u-speech2speech/s2p/multirun/cv4_fr/xlsr/cv_wiki_sil_0-5/cp4_gp2.0_sw0.5/seed2/ckpt_best_asr_test_viterbi_500-5.0.phones_w_sil/asr_test.txt"

# src_dir = "/work/b07502072/corpus/u-s2s/audio/en_feats/LJ_speech/large_clean/precompute_pca512_cls128_mean_pooled"
# gold_dir = "/home/b07502072/u-speech2speech/s2p/utils/goldens/LJ_speech/g2p"

src_dir = "/work/b07502072/corpus/u-s2s/audio/fr_feats/cv4/xlsr/precompute_pca512_cls128_mean_pooled"
gold_dir = "/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_fr/train_70h"
# src_dir = "/work/b07502072/corpus/u-s2s/audio/fr_feats/cv4_40h/large_clean/precompute_pca512_cls128_mean_pooled"
# gold_dir = "/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_fr/train_40h"

cal_hyps = False
split = "valid"
sil_probs = [0.25, 0.5, 0.75, 1.0]
SR = 16_000

def main():
    tsv_fpath = osp.join(src_dir, f"{split}.tsv")
    len_fpath = osp.join(src_dir, f"{split}.lengths")
    phn_fpath = osp.join(gold_dir, f"{split}.phones.txt")
    wrd_fpath = osp.join(gold_dir, f"{split}.words.txt")

    with open(tsv_fpath, 'r') as tsv_fr, open(
        len_fpath, 'r'
    ) as len_fr, open(
        phn_fpath, 'r'
    ) as phn_fr, open(
        wrd_fpath, 'r'
    ) as wrd_fr:
        audio_root = tsv_fr.readline()
        audio_len = [int(line.strip().split('\t')[1]) / SR for line in tsv_fr]
        feats_len = [int(line.strip()) for line in len_fr]
        phone_len = [len(line.strip().split()) for line in phn_fr]
        words_len = [len(line.strip().split()) for line in wrd_fr]
        audio_sum = sum(audio_len)
        feats_sum = sum(feats_len)
        phone_sum = sum(phone_len)
        # print(f"{feats_sum / audio_sum}")
        # print(f"{phone_sum / audio_sum}")
        print(f"feats_freq\t{feats_sum / audio_sum}")
        print(f"phone_freq\t{phone_sum / audio_sum}")
        for sil_prob in sil_probs:
            additional_len = [ 2 + sil_prob*(wrds-1) for wrds in words_len]
            phone_sum = sum(phone_len) + sum(additional_len)
            print(f"w/ sil={sil_prob:.2f}\t{phone_sum / audio_sum}")
            # print(f"{phone_sum / audio_sum}")
    
        if cal_hyps:
            with open(hyp_fpath, 'r') as hyp_fr, open(
                hyp_sil_fpath, 'r'
            ) as hyp_sil_fr:
                hyp_phn_sum = sum([len(line.strip().split()) for line in hyp_fr])
                hyp_sil_phn_sum = sum([len(line.strip().split()) for line in hyp_sil_fr])
                print(f"hyps_freq\t{hyp_phn_sum / audio_sum}")
                print(f"hyps_w_sil\t{hyp_sil_phn_sum / audio_sum}")



if __name__ == "__main__":
    main()
