#!/usr/bin/env python3
import torch
import argparse
from torch import nn


def generate_random_sample(bsz, tsz, device='cpu'):
    # bsz: batch_size; tsz: time_size
    random_sample = torch.randn((bsz, tsz, 512), device=device) # (B, T, C), C=512 (repr dim from "w2v2->pca")
    random_sample_size = torch.randint(5, tsz, (bsz,), device=device)
    random_sample_mask = torch.arange(tsz, device=device).expand(bsz, tsz) >= random_sample_size.unsqueeze(1)
    return random_sample, random_sample_mask

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)
    

class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = None

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        
        preds = logits.argmax(dim=-1)

        
        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )
        
        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)


        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            new_logits[b].index_add_(
                dim=0, index=idx.to(new_logits.device), source=logits[b]
            )
            new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Hard-coded parameters
        self.input_dim = 512
        self.output_dim = 44
        self.kernel_size = 4
        self.stride = 1
        self.padding = 2
        self.blank_index = 0
        dropout = 0.1
        bias = False

        # Segmentation
        self.segmenter = JoinSegmenter()

        # Layers
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                self.input_dim, 
                self.output_dim, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=bias
            ),
            TransposeLast()
        )

        # Batch normalization and residual connections are not included in the printed model
        # If needed, they should be added here

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )

        dense_x = dense_x.softmax(-1)

        return dense_x, code_perplexity, prob_perplexity

    def forward(self, features, padding_mask, dense_x_only=True):
        # Presegment
        features, padding_mask = self.segmenter.pre_segment(features, padding_mask)

        dense_padding_mask = padding_mask
        # Define the forward pass based on your requirements
        
        x = self.dropout(features)
        x = self.proj(x)

        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != x.size(1):
            new_padding = dense_padding_mask.new_zeros(x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        # Get the logits from the output
        logits = x
        
        # Presegmentation
        # TODO: uncomment this line to enable presegmentation
        #logits, dense_padding_mask = self.segmenter.logit_segment(logits, dense_padding_mask)
        

        # Normalization
        logits, _, _ = self.normalize(logits)
        
        # The following parts are the get_logits() function from the original model
        if dense_padding_mask.any():
            logits[dense_padding_mask] = float("-inf")
            logits[dense_padding_mask][..., self.blank_index] = float("inf")
        return logits.transpose(0, 1)


def main(args):
    
    ASR = Generator()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the generator
    ASR.load_state_dict(torch.load("./generator.pt"))
    ASR = ASR.to(device=device)
    random_sample, random_sample_mask = generate_random_sample(bsz=5, tsz=100, device=device)
    
    
    input = {
        "features": random_sample,
        "padding_mask": random_sample_mask,
        "dense_x_only": True # set this to True to get generator outputs only
    }
    
    emissions = ASR(**input)
    print(f"emissions.shape: {emissions.shape}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="../env.yaml",
        help="custom local env file for github collaboration",
    )
    args = parser.parse_args()
    main(args)