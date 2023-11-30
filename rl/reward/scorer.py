import kenlm # just follow the installation instruction of https://github.com/kpu/kenlm
import torch
import argparse
import editdistance
import logging
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from dictionary import Dictionary # import the hacked dictionary cloned from fairseq for greedy-decoding

logger = logging.getLogger(__name__)

@dataclass
class ScorerCfg:
    kenlm_fpath: str = ""
    dict_fpath: str = ""

class Scorer(object):
    def __init__(self, score_cfg: ScorerCfg) -> None:
        # print(score_cfg)
        self.cfg = score_cfg
        self.lm = kenlm.Model(score_cfg.kenlm_fpath)
        self.dictionary = Dictionary.load(score_cfg.dict_fpath)
        self.sil_id = self.dictionary.index("<SIL>")
        self.num_symbols = len(self.dictionary) - self.dictionary.nspecial
        # sentence = "<SIL> W AY <SIL> L AE D IY S EH D HH IY <SIL>"
        # print("sentence-level score: ")
        # print(self.lm.score(sentence)) # if properly loaded, the lm should be able to score the sentence
        # print("word-level scores: ")
        # words = ['<s>'] + sentence.split() + ['</s>']
        # for i, (prob, length, oov) in enumerate(self.lm.full_scores(sentence, bos=False, eos=False)):
        #     print('{3}\t{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2]), i))
        #     if oov:
        #         print('\t"{0}" is an OOV'.format(words[i+1]))
    
    def compute_uttwise_and_framewise_lm_score(self, sentence: str, bos=True, eos=True) -> float:
        _framewise_lm_scores = [log_prob for log_prob,_ , _ in self.lm.full_scores(sentence, bos=bos, eos=eos)]
        uttwise_lm_score = sum(_framewise_lm_scores)
        framewise_lm_scores = _framewise_lm_scores[bos:len(_framewise_lm_scores)-eos] # remove bos or eos scores to match sequence length
        return uttwise_lm_score, framewise_lm_scores
    
    def score(self, result, merge_consecutives=False, rm_sil=False) -> float:
        # pass
        dense_x = result["logits"] # [B, T, C]
        padding_mask = result["padding_mask"] # [B, T]
        dense_x[:,:,:self.dictionary.nspecial] = float("-inf") # special tokens should be dummy during decoding or scoring
        bsz, _, _ = dense_x.size()

        z = dense_x.argmax(-1) # z: [B, T], z is the greedy-decoding result of x, every sequence length of z in the batch == sequence length of dense_x.
        z[padding_mask] = self.dictionary.pad() # the only special token that could be in z is <pad>

        vocab_seen = torch.zeros(self.num_symbols, dtype=torch.bool)

        c_err = 0 # edit distance errs of the batch
        c_len = 0 # target length of the batch
        pred_c_len = 0 # prediction length of the batch
        lm_score_sum = 0 # lm score of the batch
        framewise_lm_scores = [] # framewise lm scores of the batch
        uttwise_lm_scores = [] # uttwise lm scores of the batch
        for i, (x, t) in enumerate(
            zip(
                z,
                result["target"] if "target" in result else [None] * len(z),
            )
        ):
            if t is not None:
                t = t[(t >= self.dictionary.nspecial)]
            x = x[(x >= self.dictionary.nspecial)] # for safety
            if rm_sil:
                x = x[(x != self.sil_id)] # this is dangerous because it will change the length of the sequence

            vocab_seen[x - self.dictionary.nspecial] = True

            pred_units_arr = x
            if merge_consecutives:
                pred_units_arr = torch.unique_consecutive(pred_units_arr) # this will change the length of the sequence.

            pred_units_arr = pred_units_arr.tolist()

            pred_c_len += len(pred_units_arr)

            if t is not None:
                t = t.tolist()
                c_err += editdistance.eval(pred_units_arr, t)
                c_len += len(t)
            else:
                c_len = pred_c_len

            if self.lm is not None:
                pred_str = self.dictionary.string(pred_units_arr)
                uttwise_lm_score, framewise_lm_score = self.compute_uttwise_and_framewise_lm_score(pred_str)
                framewise_lm_scores.append(framewise_lm_score)
                uttwise_lm_scores.append(uttwise_lm_score / (len(pred_units_arr) + 2))
                lm_score_sum += uttwise_lm_score
        
        vocab_seen_percentage = vocab_seen.sum().item() / self.num_symbols
        batchwise_lm_score = lm_score_sum / (pred_c_len + 2 * bsz)
        batchwise_lm_ppl = math.pow(10, -batchwise_lm_score)
        uttwise_lm_ppls = [math.pow(10, -s) for s in uttwise_lm_scores] # uttwise lm ppls[0] should be the same as the batchwise lm ppl when the batch size is 1
        scores = {
            'batchwise_lm_ppl': batchwise_lm_ppl, 
            'uttwise_lm_ppls': uttwise_lm_ppls,
            'framewise_lm_scores': framewise_lm_scores, # the score indicates the framewise log probabilities.
            'vocab_seen_percentage': vocab_seen_percentage, # you can weight the lm score by this percentage to encourage the model to generate more diversly.
            'token_error_rate': c_err / c_len if c_len > 0 else 0.0,
        }
        return scores
        # return self.lm.score(sentence)

def main(args): # for testing
    score_cfg = ScorerCfg(**vars(args))
    scorer = Scorer(score_cfg)
    targets = [
        "<SIL> W AY <SIL> L AE D IY S EH D HH IY <SIL>",
        "<SIL> IH T S AO L M OW S T AO L AW T <SIL>",
        "<SIL> AY W UH D <SIL> AH V <SIL> K AO R S IH N F AH N AH T L IY <SIL> HH AE V <SIL> P R AH F ER D <SIL> T UW S T AA R T AO F W IH DH <SIL> D EH B R AH AA N S AH M JH ER N IY AH V W IH CH W IY D IH D N AA T IY V IH N <SIL> N OW DH AH EH N D B AH T DH AE T W AA Z P ER HH AE P S AH <SIL> F UW L IH SH <SIL> AY D IY AH <SIL> AH N D N AA T W AH N T UW B IY EH N K ER IH JH D W IH DH AH Y AH NG G ER L <SIL> T UW B IY K AH N S IH D ER D <SIL>"
    ]
    tgt_ids = [scorer.dictionary.encode_line(t, append_eos=False, add_if_not_exist=False).long() for t in targets]
    lengths = torch.from_numpy(np.array([len(tgt) for tgt in tgt_ids]))
    max_len = lengths.max().item()
    perfect_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=0)
    padding_mask = torch.arange(max_len).expand(len(targets), max_len) >= lengths.unsqueeze(1)
    pred_logits = torch.nn.functional.one_hot(perfect_ids, num_classes=len(scorer.dictionary)).float()
    scores = scorer.score({
        "logits": pred_logits,
        "padding_mask": padding_mask,
        "target": tgt_ids,
    })
    print(scores['batchwise_lm_ppl'], scores['token_error_rate'], scores['vocab_seen_percentage'], scores['framewise_lm_scores'][0][:5]) # ppl should be low 🙂
    # now add some random processes
    for _ in range(100):
        rand_b, rand_t = np.random.randint(0, len(targets)), np.random.randint(0, max_len)
        pred_logits[rand_b, rand_t] = torch.randn(len(scorer.dictionary))
    scores = scorer.score({
        "logits": pred_logits,
        "padding_mask": padding_mask,
        "target": tgt_ids,
    })
    print(scores['batchwise_lm_ppl'], scores['token_error_rate'], scores['vocab_seen_percentage'], scores['framewise_lm_scores'][0][:5]) # ppl should be higher 😨
    # MORE RANDOM PROCESSES
    for _ in range(1000):
        rand_b, rand_t = np.random.randint(0, len(targets)), np.random.randint(0, max_len)
        pred_logits[rand_b, rand_t] = torch.randn(len(scorer.dictionary))
    scores = scorer.score({
        "logits": pred_logits,
        "padding_mask": padding_mask,
        "target": tgt_ids,
    })
    print(scores['batchwise_lm_ppl'], scores['token_error_rate'], scores['vocab_seen_percentage'], scores['framewise_lm_scores'][0][:5]) # ppl should be very high 😱 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kenlm_fpath", type=str, default="YOUR_KENLM_PATH",
        help="Path to the language model."
    )
    parser.add_argument(
        "--dict_fpath", type=str, default="",
        help="Path to the fairseq-style dictionary."
    )
    args = parser.parse_args()
    main(args)