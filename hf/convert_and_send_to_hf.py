import argparse
import torch
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
from .configuration_reborn import RebornUASRConfig
from .modeling_reborn import RebornUASRModel
from transformers import PretrainedConfig, PreTrainedModel, AutoModel

def get_extracted_features_dataset(feats_dir, split):
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

def init_model(args):
    def _load_pca(pca_dir, model: RebornUASRModel):
        pca_A_npy = np.load(f"{pca_dir}/512_pca_A.npy")
        pca_b_npy = np.load(f"{pca_dir}/512_pca_b.npy")
        pca_A = torch.from_numpy(pca_A_npy).float().transpose(0, 1)
        pca_b = torch.from_numpy(pca_b_npy).float()
        # to linear layer state_dict
        pca_state_dict = {
            "weight": pca_A,
            "bias": pca_b
        }
        model.pca.load_state_dict(pca_state_dict)
    
    def _load_segmenter_ckpt(ckpt_fpath, model: RebornUASRModel):
        segmenter_state_dict = torch.load(ckpt_fpath, map_location="cpu")
        segmenter_state_dict = {k.replace("boundary_predictor.", ""): v for k, v in segmenter_state_dict.items()}
        model.segmenter.load_state_dict(segmenter_state_dict)
    
    def _load_generator_ckpt(ckpt_fpath, model: RebornUASRModel):
        wav2vecu_state_dict = torch.load(ckpt_fpath, map_location="cpu")
        generator_state_dict = {}
        for k, v in wav2vecu_state_dict['model'].items():
            if "generator" in k:
                generator_state_dict[k.replace("generator.proj.1.weight", "proj.0.weight")] = v
        model.generator.load_state_dict(generator_state_dict)

    model_config = RebornUASRConfig.from_pretrained(args.config_fpath)
    model = RebornUASRModel(model_config)
    if args.pca_dir:
        _load_pca(args.pca_dir, model)
        print("PCA loaded")
    if args.segmenter_ckpt:
        _load_segmenter_ckpt(args.segmenter_ckpt, model)
        print("Segmenter loaded")
    if args.generator_ckpt:
        _load_generator_ckpt(args.generator_ckpt, model)
        print("Generator loaded")

    return model_config, model

def generate_random_example():
    test_input = torch.randn(2, 100, 1024)
    test_padding_mask = torch.zeros(2, 100, dtype=torch.bool)
    test_padding_mask[1, 70:] = True
    example = {
        "input": test_input,
        "padding_mask": test_padding_mask
    }
    return example

def test_model(model, output_fpath, data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dataset = get_extracted_features_dataset(data_root, "test")
    with torch.no_grad(), open(output_fpath, "w") as fw:
        for i in tqdm.tqdm(range(len(dataset)), total=len(dataset), desc=f"Generating results...", dynamic_ncols=True):
            feats = dataset[i]["features"] # (T, C)
            feats = feats.unsqueeze(0).to(device) # (B, T, C)
            feats_padding_mask = torch.zeros(feats.shape[:-1], dtype=torch.bool, device=device)

            hypothesis = model.generate(feats, feats_padding_mask)[0]
            print(hypothesis, file=fw, flush=True)

def send_to_hf(model_config: PretrainedConfig, model: PreTrainedModel, model_card: str, commit_message=""):
    model_config.register_for_auto_class()
    model.register_for_auto_class()
    model.push_to_hub(model_card, use_temp_dir=True, commit_message=commit_message)

def test_hf_model(model_card, output_fpath, data_root):
    print("Testing the model from HF")
    model_from_hf = AutoModel.from_pretrained(f"andybi7676/{model_card}", trust_remote_code=True, revision="main")
    test_model(model_from_hf, output_fpath, data_root)

def main(args):
    print(args)
    model_config, model = init_model(args)
    print("Model config:")
    print(model_config)
    # test local model
    data_root = args.test_data_root
    # test_model(model, "/home/andybi7676/Desktop/uasr-rl/hf/test_results/local_output.txt", data_root)
    # send to HF
    model_card = args.model_card
    if args.send:
        send_to_hf(model_config, model, model_card, commit_message=args.commit_message) # only call this when sending to HF
    # test model from HF
    test_hf_model(model_card, "/home/andybi7676/Desktop/uasr-rl/hf/test_results/remote_output.txt", data_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fpath",
        default="./reborn_uasr/config_ls100h.json"
    )
    parser.add_argument(
        "--model_card",
        default="reborn-uasr_ls100h_iter2-stage1",
        help="the model card for the model push to HF",
    )
    parser.add_argument(
        "--test_data_root",
        type=str,
        default="",
    )
    parser.add_argument(
        "--pca_dir",
        default="",
        help="the fpath for the pca matrices",
    )
    parser.add_argument(
        "--segmenter_ckpt",
        default="",
        help="the ckpt fpath for the segmenter",
    )
    parser.add_argument(
        "--generator_ckpt",
        default="",
        help="the ckpt fpath for the generator",
    )
    parser.add_argument(
        "--commit_message",
        default="",
        help="the commit message for the model push to HF",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    main(args)