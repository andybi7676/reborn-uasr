#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import os

def map_time_to_boundary(x_min, x_max, phones, f_len):
  # Each frame in wav2vec2 feature is 20ms
  frame_time = x_max[-1] / f_len
  dropped_phones = []
  one_idx = []
  for start in x_min:
    one_idx.append(np.round(start / frame_time).astype(int))
  last_idx = np.ceil(x_max[-1] / frame_time)

  boundaries = [0 for _ in range(int(last_idx))]

  # Handle the case of duplicated boundaries
  for i in range(1, len(one_idx)):
    if one_idx[i] != one_idx[i - 1]:
      dropped_phones.append(phones[i - 1])
  dropped_phones.append(phones[-1])

  for one in one_idx:
    boundaries[one] = 1
  assert sum(boundaries) == len(dropped_phones), f"{sum(boundaries)} {len(dropped_phones)}"

  return boundaries, dropped_phones

def get_MFA_boundary(filename):
  with open(filename) as f:
    lines = f.readlines()
  phone_bd_starts = False
  x_min = []
  x_max = []
  phones = []
  for line in lines:
    if "item [2]:" in line:
      phone_bd_starts = True
      continue
    if phone_bd_starts:
      if "xmin" in line:
        x_min.append(float(line.strip().split()[-1]))
      if "xmax" in line:
        x_max.append(float(line.strip().split()[-1]))
      if "text" in line:
        phones.append(eval(line.strip().split()[-1]))
  x_min = x_min[1:]
  x_max = x_max[1:]
  assert len(x_min) == len(x_max) == len(phones), f"{len(x_min)} {len(x_max)} {len(phones)}, {filename}"
  return x_min, x_max, phones

def get_test_file_bds(test_file, feat_len):
  with open(test_file) as f:
    lines = f.readlines()
  lines = lines[1:]
  all_boundaries = []
  all_phones = []

  phone_drop = 0
  assert len(lines) == len(feat_len)
  for line, f_len in tqdm(zip(lines, feat_len)):
    filename = line.strip().split()[0].replace("flac", "TextGrid")
    dir_name = filename.split("-")[0].split('/')[1]
    subdir_name = filename.split("-")[1]
    filename = filename.split("/")[-1]
    x_min, x_max, phones = get_MFA_boundary(os.path.join("test-clean", dir_name, filename))
    boundaries, dropped_phones = map_time_to_boundary(x_min, x_max, phones, f_len)
    if len(phones) != len(dropped_phones):
      phone_drop += 1
    assert sum(boundaries) == len(dropped_phones), f"{sum(boundaries)} {len(dropped_phones)}, {filename}"
    all_boundaries.append(boundaries)
    all_phones.append(dropped_phones)
  
  if phone_drop > 0:
    print(f"Number of dropped phones: {phone_drop}")
  return all_boundaries, all_phones

def get_segment_freq(phones, pred, gt):
  #assert len(gt) == len(pred), f"{len(gt)} {len(pred)}"
  if len(gt) != len(pred):
    min_len = min(len(gt), len(pred))
    gt = gt[:min_len]
    pred = pred[:min_len]
  assert sum(gt) == len(phones), f"{sum(gt)} {len(phones)}"
  one_idx = [i for i, x in enumerate(gt) if x == 1]
  assert len(one_idx) == len(phones), f"{len(one_idx)} {len(phones)}"
  result = {}
  for i, one in enumerate(one_idx):
    phone = phones[i]
    start = one
    end = one_idx[i + 1] if i < len(one_idx) - 1 else len(gt)
    result[phone] = result.get(phone, []) + [pred[start:end].count(1)]
  return result

pred_bd_files = [f"test_epoch{i}.bds" for i in range(16)]
pred_phone_files = [f"test_epoch{i}.txt" for i in range(16)]


def read_bd_file(filename):
  boundaries = []
  with open(filename) as f:
    for line in f:
      boundaries.append([int(x) for x in line.strip().split()])
  return boundaries

def read_phone_file(filename):
  phones = []
  with open(filename) as f:
    for line in f:
      phones.append(line.strip().split())
  return phones

# Get the feature length
predict_bd = read_bd_file(pred_bd_files[0])
feat_len = [len(x) for x in predict_bd]

gold_bd, gold_phone = get_test_file_bds("test.tsv", feat_len)

assert len(gold_bd) == len(gold_phone)

#  for utter_bd, utter_phone in zip(gold_bd, gold_phone):
#    print(f"Number of phones: {len(utter_phone)}")
#    print(f"length of utterance: {sum(utter_bd)}")


full_result = {}

for predict_bd_file, predict_phone_file in zip(pred_bd_files, pred_phone_files):
  predict_bd = read_bd_file(predict_bd_file)
  predict_phone = read_phone_file(predict_phone_file)
  assert len(predict_bd) == len(predict_phone)
  assert len(predict_bd) == len(gold_bd)
  epoch_result = {}
  for predict_bd_utt, predict_phone_utt, gold_bd_utt, gold_phone_utt in zip(
    predict_bd, 
    predict_phone, 
    gold_bd, 
    gold_phone
  ):
    assert len(gold_phone_utt) == sum(gold_bd_utt)
    result = get_segment_freq(gold_phone_utt, predict_bd_utt, gold_bd_utt)
    for k, v in result.items():
      epoch_result[k] = epoch_result.get(k, []) + v
  epoch = int(predict_bd_file.split(".")[0].split("epoch")[-1])
  full_result[epoch] = epoch_result

for k, v in full_result.items():
  for phone, freq in v.items():
    full_result[k][phone] = np.mean(freq)

# Plot a line plot. Each line is a phone. x-axis is epoch, y-axis is the frequency of the phone
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