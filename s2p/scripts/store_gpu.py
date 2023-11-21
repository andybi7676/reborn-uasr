import os
import os.path as osp
import contextlib

import numpy as np
from npy_append_array import NpyAppendArray
import sys
import torch
import torch.nn.functional as F
import faiss
from shutil import copyfile

import soundfile as sf
import fairseq
import pandas as pd
from fairseq.data import FairseqDataset, data_utils
import random
import math
import time
import tqdm


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source, mask=False, features_only=True, layer=self.layer)
            return m_res["x"].squeeze(0).cpu()

rd = Wav2VecFeatureReader("/work/b07502072/pretrained_models/wav2vec_vox_new.pt", 14)
while True:
    time.sleep(0.25)
    feats = rd.get_feats("/work/b07502072/corpus/u-s2s/audio/LJ_speech/wavs_wo_sil/LJ026-0087.wav")