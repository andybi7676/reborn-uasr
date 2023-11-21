from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer import phonemize
import os
import argparse
import json
import sentencepiece as spm
import os.path as osp

def post_process_fr_fr(phones):
    replace_dict = {
        'ɑ̃': 'ɔ̃'
    }
    reduced_set = [chr(720), chr(771)] # 'ː', '̃ '
    
    def _replace(phns):
        for k, v in replace_dict.items():
            phns = phns.replace(k, v)
        return phns
    def _reduce(phns):
        for re in reduced_set:
            phns = phns.replace(re, "")
        return phns
    
    new_phones = [_reduce(_replace(phns)) for phns in phones]
    return new_phones

def post_process_vowel_consonant(phones):
    vowels = ['ɪ', 'ə', 'æ', 'ɛ', 'ɑ', 'ː', 'ɚ', 'o', 'ʊ', 'i', 'e', 'u', 'ʌ', 'a', 'ᵻ', 'ɜ', 'ɐ', 'ɑ', 'ɔ']
    def _is_vowel(phn):
        for p in phn:
            if p in vowels: return True
        return False
    
    def _map_to_vowel_and_consonant(phns):
        new_phns = []
        for phn in phns.split():
            new_phns.append('v' if _is_vowel(phn) else 'c')
        return ' '.join(new_phns)
    
    new_phones = [_map_to_vowel_and_consonant(phns) for phns in phones]
    return new_phones

def post_process_characterize(phones):
    def _characterize(phns):
        new_phns = []
        for c in phns.strip():
            if c != ' ':
                new_phns.append(c)
        return ' '.join(new_phns)
    
    new_phones = [_characterize(phns) for phns in phones]
    return new_phones

def post_process_hc(phones, load_dir=None):
    phn_mapping_fpath = osp.join(load_dir, "phn_dict.map")
    with open(phn_mapping_fpath, 'r') as fr:
        replace_dict = json.load(fr)
        print(f"phn_mapping_for_hierarchical_clustering: {replace_dict}")
        print("")
    def _replace(phns):
        for k, v in replace_dict.items():
            phns = phns.replace(k, v)
        return phns
    new_phones = [_replace(phns) for phns in phones]
    return new_phones

def post_process_bpe(phones, load_dir=None):
    sp = spm.SentencePieceProcessor(model_file=f"{load_dir}/spm.model")
    def _run_bpe(phns):
        new_phns = [' '.join(sp.encode(phn.strip(), out_type=str)) for phn in phns.strip().split()]
        return ' '.join(new_phns)
    new_phones = [_run_bpe(phns) for phns in phones]
    return new_phones
     
# to see support langs: 
# print(EspeakBackend.supported_languages())
def main():
    parser = get_parser()
    args = parser.parse_args()

    lang = args.lang
    root = args.root
    fname = args.fname
    post_process_code = args.post_process_code
    if post_process_code == 'bpe':
        sep = Separator(phone='', syllable='', word=' ')
    else:
        sep = Separator(phone=' ', syllable='', word='')
    if fname != '':
        word_path = os.path.join(root, fname+".words.txt")
        out_path = os.path.join(root, fname+".phones.txt")
    else:
        word_path = os.path.join(root, "words.txt")
        out_path = os.path.join(root, "phones.txt")
    word_path = "/home/b07502072/u-speech2speech/s2p/asr_test.txt"

    phones = []
    words = []
    with open(word_path, 'r') as fr:
        for line in fr:
            words.append(line.strip())
        # phones = [ ph.strip() for ph in phonemize(words, language=lang, separator=sep, language_switch="remove-flags")]
        # phones = [ ph.replace('  ', ' ') for ph in phones ]
        phones = words
        if post_process_code != "":
            post_process_code = post_process_code.replace('-', '_')
            post_process_fn = eval(f"post_process_{post_process_code}")
            if post_process_code == "bpe" or post_process_code == "hc":
                phones = post_process_fn(phones, load_dir=args.load_dir)
            else:
                phones = post_process_fn(phones)

    with open(out_path, 'w') as fw:
        for phn in phones:
            fw.write(f"{phn}\n")

def get_parser():
    parser = argparse.ArgumentParser(
        description="turn words into espeak phones. "
    )
    # fmt: off
    parser.add_argument('root', help='root dir of input words.txt and phones.txt')
    parser.add_argument('--fname', default='', help='file name of input words.txt and phones.txt')
    parser.add_argument('--lang', help='language to be converted.', default='en-us')
    parser.add_argument('--post_process_code', default='')
    parser.add_argument('--load_dir', default="")

    return parser

if __name__ == "__main__":
    main()