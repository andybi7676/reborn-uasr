from dataclasses import dataclass
import torch
import torch.nn as nn
from fairseq.dataclass import FairseqDataclass
from enum import Enum, auto

class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()
    CNN_BOUNDARY = auto()

@dataclass
class SegmentationConfig(FairseqDataclass):
    type: SegmentationType = SegmentationType.CNN_BOUNDARY
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    remove_zeros: bool = False


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask

@dataclass
class CnnBoundaryConfig:
    """
    Config for CNN-based boundary predictor
    """
    input_dim: int = 512
    hidden_dim: int = 512
    dropout: float = 0.1
    kernel_size: int = 7

class CnnBoundaryPredictor(nn.Module):
    """
    Boundary predictor for CNN-based agent
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv1 = nn.Conv1d(cfg.input_dim, cfg.hidden_dim, cfg.kernel_size, padding=cfg.kernel_size//2)
        self.conv2 = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(cfg.hidden_dim, 2, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Input:
            x: (B, T, C)
            padding_mask: (B, T) # 0: not padding; 1: padding
        Output:
            boundary: (B, T, 2) # 0: not boundary; 1: boundary
        """

        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(1, 2)
        return x

class CnnSegmenter(Segmenter):
    """
    Segmenter for CNN-based agent
    """
    def __init__(self, cfg: SegmentationConfig, cnn_boundary_cfg: CnnBoundaryConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate
        self.boundary_predictor = CnnBoundaryPredictor(cnn_boundary_cfg)

    def boundary_predict(self, x, padding_mask, deterministic=False):
        """
        Input:
            x: (B, T, C)
            padding_mask: (B, T)
        Output:
            boundary: (B, T) # 0: not boundary; 1: boundary
            boundary_logits: (B, T, 2) # 0: not boundary; 1: boundary
        """
        boundary_logits = self.boundary_predictor(x)
        if deterministic:
            boundary = boundary_logits.argmax(-1)
            boundary[padding_mask] = -1
        else:
            boundary = torch.distributions.Categorical(logits=boundary_logits).sample()
            boundary[padding_mask] = -1
        return boundary, boundary_logits

    def pre_segment(self, logits, padding_mask, return_boundary=False, deterministic=False):
        """
        Input:
            logits: (B, T, C)
            padding_mask: (B, T)
        Output:
            new_logits: (B, T', C)
            new_padding_mask: (B, T')
        """
        
        bsz, tsz, csz = logits.size()
        
        boundary, boundary_logits = self.boundary_predict(logits, padding_mask, deterministic=deterministic)
        
        # max boundary number
        # print("boundary", boundary)
        # print(torch.sum(boundary==1, dim=1))
        new_tsz = int(torch.max(torch.sum(boundary==1, dim=1)).item())+1
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)
        
        
        for b in range(bsz):
            # merge consecutive segments when meeting a boundary (mean_pool_join)
            new_idx = 0
            count = 0
            for t in range(tsz):
                if padding_mask[b, t] == 1:
                    new_pad[b, new_idx] = True
                    continue
                if boundary[b, t] == 1:
                    new_logits[b, new_idx] /= count
                    new_idx += 1
                    count = 0
                new_logits[b, new_idx] += logits[b, t]
                count += 1
            new_logits[b, new_idx] /= count
            if new_idx < new_tsz:
                pad = new_tsz - new_idx
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        if return_boundary:
            return new_logits, new_pad, boundary, boundary_logits
        return new_logits, new_pad
    
    
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

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
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
