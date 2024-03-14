import os
import json
from transformers import PretrainedConfig

class RebornUASRConfig(PretrainedConfig):
    '''
    We can use this class to define the configuration of the reborn model. 
    The reborn UASR is composed of a segmenter, a discriminator, and a generator.
    We only include the required configurations for the discriminator and the generator from fairseq's wav2vec-U model configuration. 
    '''
    model_type = "reborn_uasr"
    
    def __init__(self, 
        segmenter_type: str = "cnn",
        segmenter_input_dim: int = 512,
        segmenter_hidden_dim: int = 512,
        segmenter_dropout: float = 0.1,
        segmenter_kernel_size: int = 7,

        discriminator_input_dim: int = 512,
        discriminator_kernel: int = 3,
        discriminator_dilation: int = 1,
        discriminator_dim: int = 256,
        discriminator_causal: bool = True,
        discriminator_linear_emb: bool = False,
        discriminator_depth: int = 1,
        discriminator_max_pool: bool = False,
        discriminator_act_after_linear: bool = False,
        discriminator_dropout: float = 0.0,
        discriminator_spectral_norm: bool = False,
        discriminator_weight_norm: bool = False,

        generator_input_dim: int = 512,
        generator_output_dim: int = 40,
        generator_kernel: int = 4,
        generator_dilation: int = 1,
        generator_stride: int = 1,
        generator_bias: bool = False,
        generator_dropout: float = 0.0,
        generator_bn_apply: bool = False,
        generator_bn_init_weight: float = 30.0,

        phones: list = [],
        dict_fpath: str = "",
        special_token_nums: int = 4, # [<s>, <pad>, </s>, <unk>]
        **kwargs
    ):
        super().__init__(**kwargs)
        # read in all the configurations
        self.segmenter_type = segmenter_type
        self.segmenter_input_dim = segmenter_input_dim
        self.segmenter_hidden_dim = segmenter_hidden_dim
        self.segmenter_dropout = segmenter_dropout
        self.segmenter_kernel_size = segmenter_kernel_size

        self.discriminator_input_dim = discriminator_input_dim
        self.discriminator_kernel = discriminator_kernel
        self.discriminator_dilation = discriminator_dilation
        self.discriminator_dim = discriminator_dim
        self.discriminator_causal = discriminator_causal
        self.discriminator_linear_emb = discriminator_linear_emb
        self.discriminator_depth = discriminator_depth
        self.discriminator_max_pool = discriminator_max_pool
        self.discriminator_act_after_linear = discriminator_act_after_linear
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_spectral_norm = discriminator_spectral_norm
        self.discriminator_weight_norm = discriminator_weight_norm

        self.generator_input_dim = generator_input_dim
        self.generator_output_dim = generator_output_dim
        self.generator_kernel = generator_kernel
        self.generator_dilation = generator_dilation
        self.generator_stride = generator_stride
        self.generator_bias = generator_bias
        self.generator_dropout = generator_dropout
        self.generator_bn_apply = generator_bn_apply
        self.generator_bn_init_weight = generator_bn_init_weight

        self.special_token_nums = special_token_nums
        if os.path.isfile(dict_fpath):
            self.phones = self.read_phns_dict_from_fpath(dict_fpath)
        else:
            self.phones = phones
        if len(self.phones) > 0:
            self.generator_output_dim = len(self.phones) + self.special_token_nums
            self.discriminator_input_dim = self.generator_output_dim
    
    def read_phns_dict_from_fpath(self, fpath: str):
        phns = []
        with open(fpath, "r", encoding="utf-8") as f:
            for l in f:
                phn = l.strip().split('\t')[0].split(' ')[0]
                phns.append(phn)
        return phns

def main():
    config = RebornUASRConfig(dict_fpath="/home/andybi7676/Desktop/uasr-rl/data2/pt_mls/text/prep/phones/dict.phn.txt")
    print(config)
    output_fpath = "./reborn_uasr_configs/config_mls-pt.json"
    with open(output_fpath, 'w', encoding='utf-8') as fw:
        config_json_string = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        fw.write(config_json_string)

if __name__ == "__main__":
    main()