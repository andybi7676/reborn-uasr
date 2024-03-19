import os
import torch
import argparse
import editdistance
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, AutoModelForPreTraining
from tqdm import tqdm

def main(args):
# load librispeech dataset for evaluation. Note that the audio is different from the original one.
    split = args.split
    dataset_card = args.dataset_card
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_card, dataset_name, split=split, streaming=True, trust_remote_code=True)

    # load the corresponding version of w2v2 as the upstream feature extractor
    upstream_model_card = args.upstream_model_card
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")
    upstream_model = AutoModelForPreTraining.from_pretrained(upstream_model_card)

    # load the reborn uasr model from the hub, which is composed of the segmenter and the generator
    reborn_model_card = args.reborn_model_card
    reborn_model = AutoModel.from_pretrained(reborn_model_card, trust_remote_code=True, revision="main")

    # model eval mode and to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    upstream_model = upstream_model.to(device)
    reborn_model = reborn_model.to(device)

    # perform evaluation and dump the results to the output directory
    total_errs = 0
    total_len = 0
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad(), open(f"{output_dir}/{split}.hyp", "w") as hyp_fw, open(f"{output_dir}/{split}.ref", "w") as ref_fw:
        for idx, sample in tqdm(enumerate(dataset), desc=f"Generating results...", dynamic_ncols=True):
            audio_feats = processor(sample["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
            audio_feats = audio_feats.to(device)
            
            upstream_output = upstream_model(audio_feats, output_hidden_states=True)
            wav2vecu_feats = upstream_output.hidden_states[15] #(B, T, C)
            feats_padding_mask = torch.zeros(wav2vecu_feats.shape[:-1], dtype=torch.bool, device=device)

            hypothesis = reborn_model.generate(wav2vecu_feats, feats_padding_mask)[0]
            reference = sample["phoneme"]
            print(hypothesis, file=hyp_fw, flush=True)
            print(reference, file=ref_fw, flush=True)
            total_errs += editdistance.eval(hypothesis.split(), reference.split())
            total_len += len(reference.split())

    print(f"PER: {total_errs / total_len * 100:.3f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_card", default="andybi7676/reborn-uasr_librispeech-no-silence-100hr")
    parser.add_argument("--dataset_name", type=str, default=None) # set language when using multilingual librispeech ({german, french, italian, spanish, portuguese, dutch})
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--upstream_model_card", type=str, default="facebook/wav2vec2-large-lv60") # facebook/wav2vec2-large-lv60 or facebook/wav2vec2-large-xlsr-53
    parser.add_argument("--reborn_model_card", type=str, default="andybi7676/reborn-uasr_ls100h_iter2-stage1")
    parser.add_argument("--output_dir", type=str, default="./test_results")

    args = parser.parse_args()
    main(args)