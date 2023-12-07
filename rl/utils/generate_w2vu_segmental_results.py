from load_w2vu_example import register_and_setup_task, get_model_and_saved_cfg
from fairseq import checkpoint_utils, tasks, utils
from omegaconf import OmegaConf
import torch
import os
import os.path as osp
import tqdm
import argparse
env = OmegaConf.load("../../env.yaml")
import sys

def get_extracted_features_dataset(env, feats_dir, split):
    from s2p.data.extracted_features_dataset import ExtractedFeaturesDataset
    dataset = ExtractedFeaturesDataset(
        path=feats_dir,
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

def get_cnn_segmenter(env, device, ckpt_fpath=None):
    if ckpt_fpath is None:
        return None
    from rl.cnn_segmenter.cnn_model import CnnSegmenter, SegmentationConfig, CnnBoundaryConfig
    cnn_segmenter = CnnSegmenter(SegmentationConfig(), CnnBoundaryConfig())
    segmenter_ckpt = torch.load(ckpt_fpath, map_location="cpu")
    cnn_segmenter.load_state_dict(segmenter_ckpt)
    cnn_segmenter.to(device)
    cnn_segmenter.eval()
    return cnn_segmenter

def generate_w2vu_segmental_results(model, dataset, dictionary, device, output_fpath, logit_segment=False, segmenter=None, postprocess_code=None):
    with torch.no_grad(), open(output_fpath, "w") as fw:
        for i in tqdm.tqdm(range(len(dataset)), total=len(dataset), desc=f"Generating results...", dynamic_ncols=True):
            feats = dataset[i]["features"] # (T, C)
            feats = feats.unsqueeze(0).to(device) # (B, T, C)
            feats_padding_mask = torch.zeros(feats.shape[:-1], dtype=torch.bool, device=device)
            if segmenter:
                feats, feats_padding_mask = segmenter.pre_segment(feats, feats_padding_mask)
            sample = {
                "features": feats,
                "padding_mask": feats_padding_mask,
                "dense_x_only": True, # set this to True to get generator outputs only
                "segment": logit_segment # set this to True to merge consecutive logits
            }
            model_out = model(**sample)
            emissions = model.get_logits(model_out)
            preds = emissions.transpose(0, 1).argmax(-1)
            hypotheses = dictionary.string(preds, bpe_symbol=postprocess_code)
            print(hypotheses, file=fw, flush=True) # (T, B, Dict_size)

def main(args, task):
    print(args)
    sys.path.append(f"{env.WORK_DIR}") # this can only be done after register_and_setup_task
    
    use_cuda = torch.cuda.is_available()
    model, saved_cfg = get_model_and_saved_cfg(env, task, use_cuda=use_cuda)
    model.eval()
    device = next(model.parameters()).device
    
    from rl.reward.dictionary import Dictionary
    segmenter = get_cnn_segmenter(env, device, ckpt_fpath=args.segmenter_ckpt)
    dictionary = Dictionary.load(f"{env.WORK_DIR}/rl/dummy_data/dict.txt")
    postprocess_code = args.postprocess_code
    split = args.split
    feats_dir = args.feats_dir

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"{env.WORK_DIR}/rl/utils/tmp"
    os.makedirs(output_dir, exist_ok=True)
    dataset = get_extracted_features_dataset(env, feats_dir, split)
    output_fpath = osp.join(output_dir, f"{split}.txt")
    generate_w2vu_segmental_results(model, dataset, dictionary, device, output_fpath, logit_segment=args.logit_segment, segmenter=segmenter, postprocess_code=postprocess_code)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate w2vu segmental results",
    )
    parser.add_argument(
        "--env",
        default="../../env.yaml",
        help="custom local env file for github collaboration",
    )
    parser.add_argument(
        "--feats_dir",
        default="../../data/audio/librispeech/large_clean/precompute_pca512_cls128_mean_pooled",
    )
    parser.add_argument(
        "--segmenter_ckpt",
        default=None,
    )
    parser.add_argument(
        "--logit_segment",
        default=False,
    )
    parser.add_argument(
        "--postprocess_code",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument(
        "--split",
        default="test",
    )
    args = parser.parse_args()
    env = OmegaConf.load(args.env)
    task_cfg_fpath = f"{env.WORK_DIR}/rl/config/dummy.yaml"
    task, task_cfg = register_and_setup_task(task_cfg_fpath, env)
    main(args, task)
