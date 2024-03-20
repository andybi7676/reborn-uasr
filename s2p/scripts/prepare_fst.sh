#!/usr/bin/env zsh
# KENLM_ROOT='/home/b07502072/kenlm/build/bin'
# target_dir=/work/b07502072/corpus/u-s2s/text/ls_wo_lv/prep

# $KENLM_ROOT/lmplz -o 4 -S 50G -T ./tmp < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 > $target_dir/kenlm.wrd.o40003.arpa
# $KENLM_ROOT/build_binary $target_dir/kenlm.wrd.o40003.arpa $target_dir/kenlm.wrd.o40003.bin
lg=$1
target_dir=$2
LANG=C # skip perl: warning: Setting locale failed. (https://stackoverflow.com/questions/2499794/how-to-fix-a-locale-setting-warning-from-perl)

lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn
lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil_lm6 lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil_lm4 lm_arpa=$target_dir/phones/lm.phones.filtered.04.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"