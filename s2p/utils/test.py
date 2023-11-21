import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairseq.modules import (
    SamePad,
    TransposeLast,
)
import soundfile as sf
import torchaudio
import time

# class Test(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.preprocess = [ TransposeLast() ]
#         bn = nn.BatchNorm1d(32)
#         bn.weight = Parameter(torch.ones(100)*35)
#         bn.running_var *= 35
#         self.preprocess.append(bn)
#         self.proj = nn.Sequential(
#             *self.preprocess
#         )
#     def forward(self, input):
#         output = self.proj(input)

# test = Test()
# print(test)
# print(test.proj[1])
# print(test.proj[1].state_dict().items())
def test_torchaudio_load(fname):
    start_time = time.time()
    for i in range(1000):
        wav, sr = torchaudio.load(fname)
    end_time = time.time()
    print(f"torchaudio load time: {end_time-start_time} secs per 10k files")
def test_sf_load(fname):
    start_time = time.time()
    for i in range(1000):
        wav, sr = sf.read(fname)
        print(wav.shape)
    end_time = time.time()
    print(f"sf load time: {end_time-start_time} secs per 10k files")
fname = '/work/b07502072/corpus/u-s2s/audio/wo_sil/de/2016/20160202-0900-PLENARY-4_de_187.ogg'

wav, sr = sf.read(fname)
source = torch.from_numpy(wav).float()
print(source.shape)
print(source)
wav, sr = torchaudio.load(fname)
print(wav.shape)
print(wav)