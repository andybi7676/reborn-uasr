import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_reborn import RebornUASRConfig
from typing import Optional, Tuple, Union, List

class RebornSegmenter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.segmenter_input_dim, config.segmenter_hidden_dim, config.segmenter_kernel_size, padding=config.segmenter_kernel_size//2)
        self.conv2 = nn.Conv1d(config.segmenter_hidden_dim, config.segmenter_hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(config.segmenter_hidden_dim, 2, 1)
        self.dropout = nn.Dropout(config.segmenter_dropout)
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
    
    def boundary_predict(self, x, padding_mask, deterministic=False):
        """
        Input:
            x: (B, T, C)
            padding_mask: (B, T)
        Output:
            boundary: (B, T) # 0: not boundary; 1: boundary
            boundary_logits: (B, T, 2) # 0: not boundary; 1: boundary
        """
        boundary_logits = self.forward(x)
        if deterministic:
            boundary = boundary_logits.argmax(-1)
            boundary[padding_mask] = -1
        else:
            boundary = torch.distributions.Categorical(logits=boundary_logits).sample()
            boundary[padding_mask] = -1
        return boundary, boundary_logits
    
    def pre_segment(self, logits, padding_mask, return_boundary=False, deterministic=True):
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
        new_tsz = int(torch.max(torch.sum(boundary==1, dim=1)).item())+1 # add <bos>
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)
        
        for b in range(bsz):
            # merge consecutive segments when meeting a boundary (mean_pool_join)
            new_idx = 0
            count = 0
            for t in range(tsz):
                if padding_mask[b, t] == 1:
                    break
                if boundary[b, t] == 1:
                    new_logits[b, new_idx] /= count
                    new_idx += 1
                    count = 0
                new_logits[b, new_idx] += logits[b, t]
                count += 1
            if count > 0:
                # last segment
                new_logits[b, new_idx] /= count
                new_idx += 1
                count = 0
            if new_idx < new_tsz:
                pad = new_tsz - new_idx
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        if return_boundary:
            return new_logits, new_pad, boundary, boundary_logits
        return new_logits, new_pad

class RebornGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_dim = config.generator_output_dim
        self.stride = config.generator_stride
        self.dropout = nn.Dropout(config.generator_dropout)
        cnn_input_dim = config.generator_input_dim
        cnn_output_dim = config.generator_output_dim

        padding = config.generator_kernel // 2
        self.proj = nn.Sequential(
            nn.Conv1d(
                cnn_input_dim,
                cnn_output_dim,
                kernel_size=config.generator_kernel,
                stride=config.generator_stride,
                dilation=config.generator_dilation,
                padding=padding,
                bias=config.generator_bias,
            ),
        )

    def forward(self, dense_x, tokens, dense_padding_mask):
        dense_x = self.dropout(dense_x)
        # (B, T, C) -> (B, C, T)
        dense_x = dense_x.transpose(-2, -1)

        dense_x = self.proj(dense_x)
        # (B, C, T) -> (B, T, C)
        dense_x = dense_x.transpose(-2, -1)
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)
            assert (
                diff > 0
            ), f"{new_padding.shape}, {dense_padding_mask.shape}, {dense_x.shape}, {diff}"
            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        result = {}

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))

        result["dense_x"] = dense_x
        result["token_x"] = token_x
        result["dense_padding_mask"] = dense_padding_mask

        return result

def get_item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == "xla":
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re
        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(' +', ' ', sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence

class SimpleTokenizer(object):
    def __init__(self,
        phones: List[str],
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ) -> None:
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)
        for phone in phones:
            self.add_symbol(phone)
        self.postprocess_code = "silence"
    
    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def get_count(self, idx):
        return self.count[idx]

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(
                    t,
                    bpe_symbol,
                    escape_unk,
                    extra_symbols_to_ignore,
                    include_eos=include_eos,
                )
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        if not include_eos:
            extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = separator.join(
            token_string(i)
            for i in tensor
            if get_item(i) not in extra_symbols_to_ignore
        )

        return post_process(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word
    
    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index


class RebornUASRModel(PreTrainedModel):
    config_class = RebornUASRConfig

    def __init__(self, config):
        super().__init__(config)
        self.pca = nn.Linear(1024, 512)
        self.segmenter = RebornSegmenter(config)
        self.generator = RebornGenerator(config)
        self.tokenizer = None
        if len(config.phones) > 0:
            self.tokenizer = SimpleTokenizer(config.phones)

    def forward(
        self,
        x: Optional[torch.Tensor], # (B, T, C)
        padding_mask: Optional[torch.Tensor], # (B, T)
    ):
        x_reduced = self.pca(x)
        x_segmented, segmented_padding_mask = self.segmenter.pre_segment(x_reduced, padding_mask, deterministic=True)
        x_generated = self.generator(x_segmented, None, segmented_padding_mask)

        return {
            'x_reduced': x_reduced,
            'x_segmented': x_segmented,
            'x_generated': x_generated
        }
    
    def generate(self, x, padding_mask, merge_consecutive=True, remove_silence=True):
        res = self.forward(x, padding_mask)
        y_raw_logits = res['x_generated']['dense_x']
        y_raw_padding = res['x_generated']['dense_padding_mask']
        y_raw_logits[y_raw_padding][..., self.tokenizer.pad_index] = float('inf')
        preds = y_raw_logits.argmax(-1)
        hyps = []
        postprocess_code = "silence" if remove_silence else "none"
        for pred in preds:
            if merge_consecutive:
                # merge consecutive predictions
                pred = torch.unique_consecutive(pred)
            hyp = self.tokenizer.string(pred, bpe_symbol=postprocess_code)
            hyps.append(hyp)
        return hyps

def main():
    model_config = RebornUASRConfig.from_pretrained("/home/andybi7676/Desktop/uasr-rl/reborn_uasr/config.json")
    print(model_config)
    model = RebornUASRModel(model_config)
    print(model.tokenizer.indices)

if __name__ == "__main__":
    main()
