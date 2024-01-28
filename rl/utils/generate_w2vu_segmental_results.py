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

def get_cnn_segmenter(env, device, ckpt_fpath=None, boundary_fpath=None):
    if ckpt_fpath is None and boundary_fpath is None:
        return None
    from rl.cnn_segmenter.cnn_model import DummySegmenter, CnnSegmenter, SegmentationConfig, CnnBoundaryConfig
    if ckpt_fpath is None:
        segmenter = DummySegmenter(SegmentationConfig(), boundary_fpath=boundary_fpath)
        return segmenter
    cnn_segmenter = CnnSegmenter(SegmentationConfig(), CnnBoundaryConfig())
    segmenter_ckpt = torch.load(ckpt_fpath, map_location="cpu")
    try:
        cnn_segmenter.load_state_dict(segmenter_ckpt)
    except:
        cnn_segmenter.boundary_predictor.load_state_dict(segmenter_ckpt)
    cnn_segmenter.to(device)
    cnn_segmenter.eval()
    return cnn_segmenter

def generate_w2vu_segmental_results(model, dataset, dictionary, device, output_fpath, logit_segment=False, segmenter=None, postprocess_code=None, return_boundary=True, deterministic=True):
    if return_boundary:
        bds_output_fpath = output_fpath.replace(".txt", ".bds")
    if segmenter:
        if hasattr(segmenter, "boundaries"):
            segment_by_boundary = True
    with torch.no_grad(), open(output_fpath, "w") as fw, open(bds_output_fpath, "w") as bds_fw, open(output_fpath.replace(".txt", ".shape"), "w") as lf:
        for i in tqdm.tqdm(range(len(dataset)), total=len(dataset), desc=f"Generating results...", dynamic_ncols=True):
            feats = dataset[i]["features"] # (T, C)
            feats = feats.unsqueeze(0).to(device) # (B, T, C)
            feats_padding_mask = torch.zeros(feats.shape[:-1], dtype=torch.bool, device=device)
            if segmenter:
                if segment_by_boundary:
                    feats, feats_padding_mask, boundary = segmenter.pre_segment_by_boundary(feats, feats_padding_mask, i, return_boundary=return_boundary)
                else:
                    feats, feats_padding_mask, boundary, boundary_logits = segmenter.pre_segment(feats, feats_padding_mask, return_boundary=return_boundary, deterministic=deterministic)
            for bd in boundary:
                bd = bd.cpu().numpy()
                bd = bd[bd!=-1]
                print(" ".join([str(b) for b in bd]), file=bds_fw, flush=True)
                print(sum(bd==1), file=lf, flush=True, end=" ")
            sample = {
                "features": feats,
                "padding_mask": feats_padding_mask,
                "dense_x_only": True, # set this to True to get generator outputs only
                "segment": logit_segment # set this to True to merge consecutive logits
            }
            model_out = model(**sample)
            emissions = model.get_logits(model_out)
            preds = emissions.transpose(0, 1).argmax(-1)
            print(preds.shape, file=lf, flush=True)
            if not logit_segment:
                hypotheses = " ".join([dictionary.symbols[p] for p in preds[0].cpu().numpy()])
            else:
                hypotheses = dictionary.string(preds, bpe_symbol=postprocess_code)
            print(hypotheses, file=fw, flush=True) # (T, B, Dict_size)

def main(args, task):
    print(args)
    sys.path.append(f"{env.WORK_DIR}") # this can only be done after register_and_setup_task
    
    use_cuda = torch.cuda.is_available()
    model, saved_cfg = get_model_and_saved_cfg(env, task, use_cuda=use_cuda, ckpt_fpath=args.generator_ckpt)
    model.eval()
    device = next(model.parameters()).device
    
    from rl.reward.dictionary import Dictionary
    segmenter = get_cnn_segmenter(env, device, ckpt_fpath=args.segmenter_ckpt, boundary_fpath=args.boundary_fpath)
    dictionary = Dictionary.load(f"{env.WORK_DIR}/rl/dict/{args.config}/dict.txt")
    postprocess_code = args.postprocess_code
    split = args.split
    feats_dir = args.feats_dir

    output_dir = args.output_dir
    output_fname = args.output_fname if args.output_fname else split
    if output_dir is None:
        output_dir = f"{env.WORK_DIR}/rl/utils/tmp"
    os.makedirs(output_dir, exist_ok=True)
    dataset = get_extracted_features_dataset(env, feats_dir, split)
    output_fpath = osp.join(output_dir, f"{output_fname}.txt")
    generate_w2vu_segmental_results(
        model, 
        dataset, 
        dictionary, 
        device, 
        output_fpath, 
        logit_segment=args.logit_segment, 
        segmenter=segmenter, 
        postprocess_code=postprocess_code,
        return_boundary=args.return_boundary,
        deterministic=args.deterministic,
    )
    

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
        "--config",
        default="librispeech",
        help="config name", # "librispeech" or "timit_matched" or "timit_unmatched"
    )
    parser.add_argument(
        "--feats_dir",
        default="../../data/audio/ls_100h_clean/large_clean/precompute_pca512",
    )
    parser.add_argument(
        "--generator_ckpt",
        default=None,
    )
    parser.add_argument(
        "--segmenter_ckpt",
        default=None,
    )
    parser.add_argument(
        "--boundary_fpath",
        default=None,
    )
    parser.add_argument(
        "--logit_segment",
        default=True,
    )
    parser.add_argument(
        "--no_logit_segment",
        dest="logit_segment",
        action="store_false",
    )
    parser.add_argument(
        "--return_boundary",
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        default=True,
    )
    parser.add_argument(
        "--postprocess_code",
        default="silence", # "silence" or "none"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument(
        "--output_fname",
        default="",
    )
    parser.add_argument(
        "--split",
        default="test",
    )
    args = parser.parse_args()
    env = OmegaConf.load(args.env)
    task_cfg_fpath = f"{env.WORK_DIR}/rl/config/{args.config}.yaml"
    task, task_cfg = register_and_setup_task(task_cfg_fpath, env)
    main(args, task)
