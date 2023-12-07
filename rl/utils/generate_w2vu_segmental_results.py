from load_w2vu_example import get_model_and_saved_cfg
from fairseq import checkpoint_utils, tasks, utils
from omegaconf import OmegaConf
import torch
import os
import os.path as osp
import tqdm
env = OmegaConf.load("../../env.yaml")
import sys

def register_and_setup_task(task_cfg_fpath, env):
    task_cfg = OmegaConf.load(task_cfg_fpath)
    task_cfg.fairseq.common.user_dir = f"{env.WORK_DIR}/s2p"
    task_cfg.fairseq.task.text_data = f"{env.WORK_DIR}/rl/dummy_data"
    utils.import_user_module(task_cfg.fairseq.common)
    task = tasks.setup_task(task_cfg.fairseq.task)
    return task, task_cfg

def get_extracted_features_dataset(env, split):
    from s2p.data.extracted_features_dataset import ExtractedFeaturesDataset
    dataset = ExtractedFeaturesDataset(
        path=f"{env.WORK_DIR}/data/audio/librispeech/large_clean/precompute_pca512_cls128_mean_pooled",
        split=split,
        min_length=0,
        max_length=None,
        labels=None,
        label_dict=None,
        shuffle=False,
        sort_by_length=False,
        aux_target_postfix=None,
    )
    return dataset

def generate_w2vu_segmental_results(model, dataset, dictionary, device, output_fpath, logit_segment=False):
    with torch.no_grad(), open(output_fpath, "w") as fw:
        for i in tqdm.tqdm(range(len(dataset)), total=len(dataset), desc=f"Generating results...", dynamic_ncols=True):
            feats = dataset[i]["features"] # (T, C)
            feats = feats.unsqueeze(0).to(device) # (B, T, C)
            feats_padding_mask = torch.zeros(feats.shape[:-1], dtype=torch.bool, device=device)
            sample = {
                "features": feats,
                "padding_mask": feats_padding_mask,
                "dense_x_only": True, # set this to True to get generator outputs only
                "segment": logit_segment # set this to True to merge consecutive logits
            }
            model_out = model(**sample)
            emissions = model.get_logits(model_out)
            preds = emissions.transpose(0, 1).argmax(-1)
            hypotheses = dictionary.string(preds)
            print(hypotheses, file=fw, flush=True) # (T, B, Dict_size)

def main():
    task_cfg_fpath = f"{env.WORK_DIR}/rl/config/dummy.yaml"
    task, task_cfg = register_and_setup_task(task_cfg_fpath, env)
    # print(task_cfg)
    sys.path.append(f"{env.WORK_DIR}") # this can only be done after register_and_setup_task
    
    use_cuda = torch.cuda.is_available()
    model, saved_cfg = get_model_and_saved_cfg(env, task, use_cuda=use_cuda)
    model.eval()
    device = next(model.parameters()).device
    
    from rl.reward.dictionary import Dictionary
    from rl.cnn_segmenter.cnn_model import CnnSegmenter, CnnSegmenterConfig, CnnBoundaryConfig
    cnn_segmenter = CnnSegmenter(CnnSegmenterConfig(), CnnBoundaryConfig())
    dictionary = Dictionary.load(f"{env.WORK_DIR}/rl/dummy_data/dict.txt")
    # generate and output results (raw greedy decode output, without logits pooling)
    raw_output_dir = f"{env.WORK_DIR}/rl/utils/w2vu_segmental_results/raw"
    os.makedirs(raw_output_dir, exist_ok=True)
    for split in ['train']:
        dataset = get_extracted_features_dataset(env, split)
        output_fpath = osp.join(raw_output_dir, f"{split}.txt")
        generate_w2vu_segmental_results(model, dataset, dictionary, device, output_fpath, logit_segment=False)
    # generate and output results (logits pooling)
    logit_segment_output_dir = f"{env.WORK_DIR}/rl/utils/w2vu_segmental_results/logit_segment"
    os.makedirs(logit_segment_output_dir, exist_ok=True)
    for split in ['valid', 'test', 'train']:
        dataset = get_extracted_features_dataset(env, split)
        output_fpath = osp.join(logit_segment_output_dir, f"{split}.txt")
        generate_w2vu_segmental_results(model, dataset, dictionary, device, output_fpath, logit_segment=True)
    

if __name__ == "__main__":
    main()
