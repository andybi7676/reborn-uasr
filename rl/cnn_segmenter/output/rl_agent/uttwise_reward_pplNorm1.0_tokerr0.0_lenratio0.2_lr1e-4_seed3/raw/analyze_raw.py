#!/usr/bin/env python3
import numpy as np
import difflib
from collections import defaultdict

def read_phone_file(filename):
  phones = []
  with open(filename) as f:
    for line in f:
      phones.append(line.strip().split())
  return phones

def map_phonemes(ground_truth, prediction):
    s = difflib.SequenceMatcher(None, ground_truth, prediction)
    return s.get_opcodes()

def process_phonemes(ground_truth, predicted):
    # Step 1: Deduplicate predicted phonemes and count duplications
    deduplicated_pred, duplication_count = [], []
    for phoneme in predicted:
        if deduplicated_pred and deduplicated_pred[-1] == phoneme:
            duplication_count[-1] += 1
        else:
            deduplicated_pred.append(phoneme)
            duplication_count.append(1)

    # Step 2: Map ground truth to predicted phonemes
    mapping = map_phonemes(ground_truth, deduplicated_pred)

    # Step 3: Associate ground truth with original duplication counts
    result = []
    pred_cursor = 0
    for tag, i1, i2, j1, j2 in mapping:
        if tag == 'equal' or tag == 'replace':
            for _ in range(i2 - i1):
                if j1 < len(deduplicated_pred):
                    if deduplicated_pred[j1] == ground_truth[i1]:  # Match found
                        result.append((ground_truth[i1], deduplicated_pred[j1], duplication_count[j1]))
                    j1 += 1
                i1 += 1
            pred_cursor = j1
        elif tag == 'insert':
            pred_cursor = j2
        elif tag == 'delete':
            continue
    return_result = defaultdict(list)
    for gt, pred, count in result:
        return_result[gt].append(count)
    return return_result

pred_phone_files = [f"test_epoch{i}.txt" for i in range(16)]

gold_phone = read_phone_file("../test.phones.txt")
full_result = {}
for pred_phone_file in pred_phone_files:
  pred_phone = read_phone_file(pred_phone_file)
  epoch_result = defaultdict(list)
  for pred_phone_utt, gold_phone_utt in zip(pred_phone, gold_phone):
    pred_phone_utt = [x for x in pred_phone_utt if x != "<SIL>"]
    gold_phone_utt = [x for x in gold_phone_utt]
    result = process_phonemes(gold_phone_utt, pred_phone_utt)

    for phone in result:
      epoch_result[phone].extend(result[phone])

  epoch = int(pred_phone_file.split(".")[0].split("epoch")[-1])
  full_result[epoch] = epoch_result

for k, v in full_result.items():
  for phone, freq in v.items():
    full_result[k][phone] = np.mean(freq)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

epochs = list(full_result.keys())
phonemes = set(phoneme for data in full_result.values() for phoneme in data)
data_for_plot = {phoneme: [] for phoneme in phonemes}

print(phonemes)

for epoch in epochs:
    for phoneme in phonemes:
        data_for_plot[phoneme].append(full_result[epoch].get(phoneme, 0))

# Plotting
plt.figure(figsize=(10, 6))
for phoneme, occurrences in data_for_plot.items():
    if phoneme == "":
      phoneme = "sil"
    if "A" in phoneme or "E" in phoneme or "I" in phoneme or "O" in phoneme or "U" in phoneme:
      color = "r"
    # Check if phoneme is capitalized
    elif phoneme[0].islower():
      color = "green"
    else:
      color = "b"

    plt.plot(epochs, occurrences, label=phoneme, color=color)

plt.xlabel('Epoch')
plt.ylabel('Occurrence')
plt.title('Phoneme Occurrence over Epochs')
plt.legend()
plt.show()