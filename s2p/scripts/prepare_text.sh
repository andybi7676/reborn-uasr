#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
set -u
set -o pipefail

KENLM_ROOT='/home/b07502072/kenlm/build/bin'
lg=$1
text_path=$2
# train_text_path=$3
target_dir=$3
min_phones=$4
phonemizer=$5
sil_prob=$6
post_process_code=$7

lid_path=$8
skip_prep=$9
create_sub=$10

if [ -z "$lid_path" ]; then
  lid_path="lid.187.bin"
fi

if [ -z "$skip_prep" ]; then
  skip_prep="false"
fi

if [ -z "$create_sub" ]; then
  create_sub="false"
fi

ph_lg=${lg}
if test "$lg" = 'fr'; then
  ph_lg='fr-fr'
elif test "$lg" = 'en'; then
  ph_lg='en-us'
elif test "$lg" = 'pt'; then
  ph_lg='pt-br'
fi

USE_ESPEAK=''
if test "$phonemizer" = 'espeak'; then
  ESPEAK_PATH='YES'
elif test "$phonemizer" = 'espeak-ng'; then
  ESPEAK_PATH=$(which espeak-ng)
elif test "$phonemizer" = 'G2P'; then
  ESPEAK_PATH=''
else
  echo "Unknown phonemizer $phonemizer. Valid options are espeak, espean-ng and G2P"
  exit 1
fi

echo "lang: $lg"
echo "phone lang: $ph_lg"
echo "text path: $text_path"
echo "target dir: $target_dir"
echo "use espeak: $ESPEAK_PATH"
echo "skip prep: $skip_prep"
echo "min phone seen threshold is $min_phones"

if test "$skip_prep" = "false"; then
  mkdir -p $target_dir
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py --lang $lg --fasttext-model $lid_path < $text_path | grep -v '\-\-\-' | grep -v '[0-9]' > $target_dir/lm.upper.lid.txt
  python /home/b07502072/u-speech2speech/s2p/scripts/normalize_and_filter_text.py --lang $lg --fasttext-model $lid_path < $text_path | grep -v '\-\-\-' | grep -v '[0-9]' > $target_dir/lm.upper.lid.txt
  # mv $target_dir/lm.upper.lid.txt $target_dir/lm.upper.lid.old.txt
  # cp $train_text_path $target_dir/train.words.txt
  # sort $target_dir/train.words.txt | uniq > $target_dir/train.words.uniq.txt
  # end_line_count=$(wc -l $target_dir/lm.upper.lid.old.txt | cut -d ' ' -f1)

  # cat $target_dir/lm.upper.lid.old.txt $target_dir/train.words.uniq.txt | awk '{ print FNR "\t" $0 }' | sort -k2 | uniq -u -f1 | sort -n > $target_dir/lm.upper.lid.train.uniq.txt
  # nl -n ln $target_dir/lm.upper.lid.old.txt $target_dir/train.words.uniq.txt | sort -k2 | uniq -u -f1 | sort -n > $target_dir/lm.upper.lid.train.uniq.txt
  # awk '{if($1 <= '$end_line_count'){print $0;}}' $target_dir/lm.upper.lid.train.uniq.txt | cut -f2- > $target_dir/lm.upper.lid.txt
  # python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py --lang $lg --fasttext-model $lid_path < $text_path | grep -v '\-\-\-' | grep -v '[0-9]'
  # python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_text.py < $text_path | grep -v '\-\-\-' | grep -v '[0-9]' > $target_dir/lm.upper.lid.txt
  python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/lm.upper.lid.txt --only-source --destdir $target_dir --thresholdsrc 2 --padding-factor 1 --dict-only
  cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' > $target_dir/words.txt

  echo "complete preprocess and create words.txt"

  if [ -z "$ESPEAK_PATH" ]; then
    python /home/b07502072/u-speech2speech/s2p/scripts/g2p_wrd_to_phn.py --compact --only_phonemes < $target_dir/words.txt > $target_dir/phones.txt
  else
    # echoing 1 into corpus will prevent the mismatch lines between lexicon and phones in case the phonemizer fails
    # one=$(echo "1" | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -p ' ' -w '' -l $ph_lg --language-switch remove-flags)
    # sed 's/$/ 1/' $target_dir/words.txt | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -o $target_dir/phones.txt -p ' ' -w '' -l $ph_lg -j 70 --language-switch remove-flags
    # echo "one is ${one}"
    # sed -i "s/${one}$//" $target_dir/phones.txt
    python /home/b07502072/u-speech2speech/s2p/scripts/phonemize_text.py $target_dir --lang $ph_lg --post_process_code $post_process_code --load_dir $target_dir
  fi
  paste $target_dir/words.txt $target_dir/phones.txt > $target_dir/lexicon.lst
else
  echo "-----------------skip preprocess---------------"
fi
echo "Finish root dir data preprocessing, processing subdir"
sleep 2.0

# python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc $min_phones --padding-factor 1 --dict-only

# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst > $target_dir/lexicon_filtered.lst
# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s $sil_prob --surround --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt > $target_dir/phones/lm.phones.filtered.txt
# cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
# echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones/lm.phones.filtered.txt --workers 40 --only-source --destdir $target_dir/phones --srcdict $target_dir/phones/dict.phn.txt

# python ~/u-speech2speech/s2p/utils/generate_lexicon_for_kenlm_decoding.py -i $target_dir/phones/dict.phn.txt -o $target_dir/phones/lexicon.phones.lst

# $KENLM_ROOT/lmplz -o 4 -S 30G -T ./tmp < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 > $target_dir/kenlm.wrd.o40003.arpa
# $KENLM_ROOT/build_binary $target_dir/kenlm.wrd.o40003.arpa $target_dir/kenlm.wrd.o40003.bin

# $KENLM_ROOT/lmplz -o 4 -S 30G -T ./tmp < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.04.arpa
# $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
# $KENLM_ROOT/lmplz -o 6 -S 30G -T ./tmp < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.06.arpa
# $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.06.arpa $target_dir/phones/lm.phones.filtered.06.bin

if test "$create_sub" = "true"; then
  mkdir -p $target_dir/phones/train_all
  cp $target_dir/phones/dict.txt $target_dir/phones/train_all
  mv $target_dir/phones/train.* $target_dir/phones/train_all
  for sub_size in 3k 30k 300k; do
    sub_size_in_num=$(echo $sub_size | cut -d 'k' -f1)000
    mkdir -p $target_dir/phones/train_$sub_size
    head -n $sub_size_in_num $target_dir/phones/lm.phones.filtered.txt > $target_dir/phones/train_$sub_size/train_$sub_size.txt
    # perl -ne 'print if (rand() < .001302599511)' lm.phones.filtered.txt > train_3k.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones/train_$sub_size/train_$sub_size.txt --workers 40 --only-source --destdir $target_dir/phones/train_$sub_size --srcdict $target_dir/phones/dict.phn.txt
  done
fi
# skip using kaldi
# lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
# lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn
# lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
