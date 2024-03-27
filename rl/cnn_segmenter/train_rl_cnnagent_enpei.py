import sys
from omegaconf import OmegaConf
env = OmegaConf.load("../../env.yaml")
sys.path.append(env.WORK_DIR)
print(f"added {env.WORK_DIR} into sys.path")
import os
from rl.cnn_segmenter.cnn_model import CnnSegmenter, CnnBoundaryPredictor, CnnBoundaryConfig, SegmentationConfig
from rl.cnn_segmenter.load_dataset import ExtractedFeaturesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from rl.reward.scorer import Scorer, ScorerCfg
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from fairseq import checkpoint_utils, tasks, utils
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

WORK_DIR="/work/r11921042"

class RLCnnAgentConfig(object):
    # config_name: str = "librispeech" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = "../../data/audio/ls_100h_clean/large_clean"
    # kenlm_fpath: str = "../../data/text/ls_wo_lv/prep_g2p/phones/lm.phones.filtered.04.bin"
    # dict_fpath: str = "../dummy_data/dict.txt"
    # pretrain_segmenter_path: str = "./output/cnn_segmenter/pretrain_PCA_postITER1_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_29_0.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/ls_100h/large_clean_postITER1/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed1/checkpoint_best.pt"
    # w2vu_postfix: str = "w2vu_logit_segmented"
    # ----------------------------------------------------
    # config_name: str = "timit_matched" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = f"{WORK_DIR}/data/audio/timit/matched/large_clean"
    # kenlm_fpath: str = f"{WORK_DIR}/data/text/timit/matched/phones/train_text_phn.04.bin"
    # dict_fpath: str = "../dict/timit_matched/dict.txt"
    # pretrain_segmenter_path: str = "./output/cnn_segmenter/lhz/timit_matched_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo20_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_10.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/cp4_gp1.5_sw0.5/seed5/checkpoint_best.pt"
    # save_dir: str = f"{WORK_DIR}/output/rl_agent/timit_matched_pplNorm1.0_tokerr0.0_lenratio0.2_lr1e-5_epoch500_seed3"
    # w2vu_postfix: str = "w2vu_logit_segmented"
    # ----------------------------------------------------
    # config_name: str = "timit_unmatched" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = f"{WORK_DIR}/data/audio/timit/unmatched/large_clean"
    # kenlm_fpath: str = f"{WORK_DIR}/data/text/timit/unmatched/phones/train_text_phn.04.bin"
    # dict_fpath: str = "../dict/timit_unmatched/dict.txt"
    # pretrain_segmenter_path: str = "./output/local/cnn_segmenter/timit_unmatched_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo100_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR_ce3.0/cnn_segmenter_20_0.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/timit_unmatched/large_clean/timit_unpaired_1k/cp4_gp2.0_sw0.5/seed2/checkpoint_best.pt"
    # save_dir: str = f"{WORK_DIR}/output/rl_agent/timit_unmatched_wfst_pplNorm1.0_tokerr1.0_lenratio0.0_lr1e-5_epoch500_seed3"
    # w2vu_postfix: str = "wfst_decoded" # "w2vu_logit_segmented" or "wfst_decoded"
    # ----------------------------------------------------
    # config_name: str = "ky" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = "../../data/audio/ky/feats/precompute_pca512"
    # kenlm_fpath: str = "../../data/text/ky/prep/phones/lm.phones.filtered.04.bin"
    # dict_fpath: str = "../dict/ky/dict.txt"
    # pretrain_segmenter_path: str = "./output/local/cnn_segmenter/ky_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo10_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/cv3_ky/xlsr/ky_unpaired_all/cp2_gp2.0_sw0.75/seed4/checkpoint_best.pt"
    # save_dir: str = "./output/local/rl_agent/ky_from_bc_relative_to_viterbi_ppl_norm_len0.2_ter0.2"
    # w2vu_postfix: str = "w2vu_logit_segmented"
    # ----------------------------------------------------
    # config_name: str = "de_mls" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = "../../data/de_mls/xlsr_100hr/precompute_pca512"
    # kenlm_fpath: str = "../../data/de_mls/text/prep/phones/lm.phones.filtered.04.bin"
    # dict_fpath: str = "../dict/de_mls/dict.txt"
    # pretrain_segmenter_path: str = "./output/local/cnn_segmenter/de_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo10_lr0.0005_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/de_mls/xlsr_100hr/de_unpaired_all/cp2_gp2.0_sw0.5/seed5/checkpoint_best.pt"
    # save_dir: str = "./output/local/rl_agent/de_mls_from_bc_rel_to_viterbi_normed_ppl_len0.2"
    # w2vu_postfix: str = "w2vu_logit_segmented"
    # ----------------------------------------------------
    config_name: str = "es_mls" # "librispeech" or "timit_matched" or "timit_unmatched"
    data_dir: str = "../../data/es_mls/xlsr_100hr/precompute_pca512"
    kenlm_fpath: str = "../../data/es_mls/text/prep/phones/lm.phones.filtered.04.bin"
    dict_fpath: str = "../dict/es_mls/dict.txt"
    pretrain_segmenter_path: str = "./output/local/rl_agent/es_mls_from_bc_rel_to_viterbi_clipped_ppl_len0.5_ter0.3_2nd_best_unsup/rl_agent_segmenter.pt"
    pretrain_wav2vecu_path: str = "../../s2p/multirun/es_mls/xlsr_100hr/es_unpaired_all/second_best_unsup/checkpoint_best.pt"
    save_dir: str = "./output/local/rl_agent/es_mls_from_bc_rel_to_viterbi_clipped_ppl_len0.5_ter0.3_2nd_best_unsup_more_epoch"
    w2vu_postfix: str = "new_w2vu_logit_segmented"

    env: str = "../../env.yaml"
    gamma: float = 1.0
    ter_tolerance: float = 0.0
    length_tolerance: float = 0.0
    logit_segment: bool = True
    apply_merge_penalty: bool = False
    wandb_log: bool = True
    utterwise_lm_ppl_coeff: float = 1.0
    utterwise_token_error_rate_coeff: float = 1.0
    length_ratio_coeff: float = 0.0
    seed: int = 3
    start_epoch: int = 0
    num_epochs: int = 500
    learning_rate: float = 1e-5
    save_interval: int = 1
    rm_sil: bool = False
    ter_rm_sil: bool = False

class TrainRlCnnAgent(object):
    def __init__(self, cfg: RLCnnAgentConfig):
        self.score_cfg = ScorerCfg(
            kenlm_fpath=cfg.kenlm_fpath,
            dict_fpath=cfg.dict_fpath,
        )
        self.scorer = Scorer(self.score_cfg)
        self.cfg = cfg
        # os.makedirs(cfg.save_dir, exist_ok=False)
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.log_fw = open(os.path.join(cfg.save_dir, "log.txt"), "a")
        self.apply_merge_penalty = cfg.apply_merge_penalty
        self.logit_segment = cfg.logit_segment
        if self.cfg.wandb_log:
            wandb.init(
                project="uasr-rl",
                name=cfg.save_dir.split('/')[-1],
                config=cfg,
            )
        # Set random seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)


        # use pandas to save csv
        import pandas as pd
        self.val_score_df = pd.DataFrame(
            columns=[
                "run_name",
                "model_name",
                "batchwise_lm_ppl",
                "uttwise_lm_ppl",
                "vocab_seen_percentage",
                "framewise_lm_scores",
                # "merge_ratio",
                "target_uttwise_lm_ppl",
                "uttwise_token_error_rate",
                "uttwise_pred_token_length",
                "uttwise_target_token_length",
                "relative_lm_ppl",
                "ppl_outperform_ratio",
                "relative_length_ratio",
                "length_diff_ratio",
                "boundary_f1",
                "boundary_precision",
                "boundary_recall",
                "boundary_1s_ratio",
                "avg_rewards",
                "eval_score",
            ])

        # Check if the csv file exists and it can be read
        if os.path.exists(os.path.join(cfg.save_dir, "val_scores.csv")):
            try:
                self.val_score_df = pd.read_csv(os.path.join(cfg.save_dir, "val_scores.csv"))
            except:
                print("val_scores.csv exists but cannot be read")
                # self.log("val_scores.csv exists but cannot be read")

        # Create csv file to save validation scores
        self.val_score_csv = open(os.path.join(cfg.save_dir, "val_scores.csv"), "w")
    
    def log(self, msg):
        print(msg, file=self.log_fw, flush=True)

    def get_score(self, pred_logits, padding_mask, target=None):
        """
        Reward function for RL agent
        Return:
            scores:
                batchwise_lm_ppl: float
                token_error_rate: float (only when tgt_ids is not None)
                vocab_seen_percentage: float
                framewise_lm_scores: list of list of float
        """
        
        result = {
            "logits": pred_logits,
            "padding_mask": padding_mask,
        }
        if target is not None:
            result["target"] = target
        
        scores = self.scorer.score(result, rm_sil=self.cfg.rm_sil, ter_rm_sil=self.cfg.ter_rm_sil)

        return scores
    
    def compute_rewards(self, scores, boundary, merge_ratio):
        """
        Compute rewards from scores
        Return:
            rewards: torch.tensor (Flattened framewise rewards)
        """
        # Compute reward: 
        if not self.apply_merge_penalty:
            merge_ratio = np.zeros(merge_ratio.shape)

        uttwise_lm_ppls = scores['uttwise_lm_ppls']
        target_uttwise_lm_ppls = scores['target_uttwise_lm_ppls']
        uttwise_token_error_rates = scores['uttwise_token_error_rates']
        uttwise_pred_token_lengths = scores['uttwise_pred_token_lengths']
        uttwise_target_token_lengths = scores['uttwise_target_token_lengths']
        length_ratio = uttwise_pred_token_lengths / uttwise_target_token_lengths
        

        uttwise_lm_ppls = torch.tensor(uttwise_lm_ppls, dtype=torch.float32).to(self.device)
        uttwise_token_error_rates = torch.tensor(uttwise_token_error_rates, dtype=torch.float32).to(self.device)
        length_ratio = torch.tensor(length_ratio, dtype=torch.float32).to(self.device)
        target_uttwise_lm_ppls = torch.tensor(target_uttwise_lm_ppls, dtype=torch.float32).to(self.device)


        length_ratio_loss = torch.abs(length_ratio - 1)
        length_ratio_loss[length_ratio_loss < self.cfg.length_tolerance] = 0.0
        uttwise_token_error_rates[uttwise_token_error_rates < self.cfg.ter_tolerance] = 0.0
       
        # reward standardization
        if len(target_uttwise_lm_ppls) == len(uttwise_lm_ppls):
            uttwise_lm_ppls = uttwise_lm_ppls - target_uttwise_lm_ppls
            # clip rewards
            # uttwise_lm_ppls = torch.clamp(uttwise_lm_ppls, -5, 5)
        else:
            uttwise_lm_ppls = (uttwise_lm_ppls - uttwise_lm_ppls.mean()) / uttwise_lm_ppls.std()

        # unstandardized rewards
        mean_rewards = - uttwise_lm_ppls.mean().item() * self.cfg.utterwise_lm_ppl_coeff - uttwise_token_error_rates.mean().item() * self.cfg.utterwise_token_error_rate_coeff - length_ratio_loss.mean().item() * self.cfg.length_ratio_coeff

        uttwise_lm_ppls = (uttwise_lm_ppls - uttwise_lm_ppls.mean()) / uttwise_lm_ppls.std()
        uttwise_token_error_rates = (uttwise_token_error_rates - uttwise_token_error_rates.mean()) / uttwise_token_error_rates.std()
        length_ratio_loss = (length_ratio_loss - length_ratio_loss.mean()) / length_ratio_loss.std()
        
        uttwise_rewards = - uttwise_lm_ppls * self.cfg.utterwise_lm_ppl_coeff - uttwise_token_error_rates * self.cfg.utterwise_token_error_rate_coeff - length_ratio_loss * self.cfg.length_ratio_coeff
        
        # reward gained at boundary=1
        rewards = torch.zeros_like(boundary, dtype=torch.float32).to(self.device)

        # for each boundary=1, reward[pos] = framewise_reward[count]
        # count = 0
        boundary_for_reward = -torch.ones(boundary.size(0), boundary.size(1)+1, dtype=boundary.dtype).to(self.device)
        boundary_for_reward[:, :-1] = boundary
        boundary_for_reward[:,-1] = -1
        boundary_padding_mask = boundary_for_reward == -1
        boundary_end_mask = boundary_padding_mask[:, 1:] & (~boundary_padding_mask[:, :-1])
        rewards[boundary_end_mask] = uttwise_rewards
        
        # cumulative reward (gamma=0.99)
        cum_rewards = torch.zeros_like(rewards, dtype=torch.float32).to(self.device)

        reward_len = rewards.size(1)
        cum_rewards[:, reward_len-1] = rewards[:, reward_len-1]
        for i in range(reward_len-2, -1, -1):
            cum_rewards[:, i] = rewards[:, i] + self.cfg.gamma * cum_rewards[:, i+1]
        # print(cum_rewards)

        # return rewards
        return cum_rewards, mean_rewards

    def log_score(self, scores, total_loss, mean_rewards, step, dataset_len, split='train', boundary_scores=None, do_log=True):
        """
        Log scores
        scores:
            batchwise_lm_ppl: float
            uttwise_lm_ppls: list of float
            framewise_lm_scores: list of list of float
            vocab_seen_percentage: float
            token_error_rate: float (only when tgt_ids is not None)
            uttwise_token_error_rates: numpy array with shape (B,)
            uttwise_token_errors: numpy array with shape (B,)
            uttwise_target_token_lengths: numpy array with shape (B,)
            uttwise_pred_token_lengths: numpy array with shape (B,)
        Log:
            batchwise_lm_ppl: float
            mean uttwise_lm_ppl: float
            mean framewise_lm_scores: float
            vocab_seen_percentage: float
            token_error_rate: float (only when tgt_ids is not None)
            mean_uttwise_token_error_rates: float
            mean_length_ratio: float
            mean_pred_token_lengths: float
        """

        batchwise_lm_ppl = scores['batchwise_lm_ppl']
        uttwise_lm_ppls = scores['uttwise_lm_ppls']
        mean_uttwise_lm_ppl = sum(uttwise_lm_ppls) / len(uttwise_lm_ppls)
        framewise_lm_scores = scores['framewise_lm_scores']
        mean_framewise_lm_scores = sum([(sum(sublist) / len(sublist)) for sublist in framewise_lm_scores if len(sublist) > 0]) / len(framewise_lm_scores)
        # print error if there is empty list in framewise_lm_scores
        for sublist in framewise_lm_scores:
            if len(sublist) == 0:
                print(f'Epmty list in framewise_lm_scores {framewise_lm_scores.index(sublist)}')
                self.log(f'Epmty list in framewise_lm_scores {framewise_lm_scores.index(sublist)}')
        vocab_seen_percentage = scores['vocab_seen_percentage']
        token_error_rate = scores['token_error_rate']
        uttwise_token_error_rates = scores['uttwise_token_error_rates']
        mean_uttwise_token_error_rates = sum(uttwise_token_error_rates) / len(uttwise_token_error_rates)
        uttwise_target_token_lengths = scores['uttwise_target_token_lengths']
        uttwise_pred_token_lengths = scores['uttwise_pred_token_lengths']
        mean_length_ratio = sum(uttwise_pred_token_lengths / uttwise_target_token_lengths) / len(uttwise_target_token_lengths)
        mean_pred_token_lengths = sum(uttwise_pred_token_lengths) / len(uttwise_pred_token_lengths)

        log_dict = {
            f'{split}_loss': total_loss,
            f'{split}_mean_reward': mean_rewards,
            f'{split}_batchwise_lm_ppl': batchwise_lm_ppl,
            f'{split}_mean_uttwise_lm_ppl': mean_uttwise_lm_ppl,
            f'{split}_mean_framewise_lm_scores': mean_framewise_lm_scores,
            f'{split}_vocab_seen_percentage': vocab_seen_percentage,
            f'{split}_token_error_rate': token_error_rate,
            f'{split}_mean_uttwise_token_error_rates': mean_uttwise_token_error_rates,
            f'{split}_mean_length_ratio': mean_length_ratio,
            f'{split}_mean_pred_token_lengths': mean_pred_token_lengths,
        }
        if boundary_scores is not None:
            log_dict.update({
                f'{split}_boundary_f1': boundary_scores['f1'],
                f'{split}_boundary_precision': boundary_scores['precision'],
                f'{split}_boundary_recall': boundary_scores['recall'],
                f'{split}_boundary_1s_ratio': boundary_scores['1s_ratio'],
            })

        if do_log == True:
            print(f'Step {step + 1}/{dataset_len}')
            for key in log_dict:
                print(f'{key}: {log_dict[key]}')
            # print(f'Framewise LM scores: {framewise_lm_scores[0][:5]}')
            # check if there is empty list in framewise_lm_scores
            # for sublist in framewise_lm_scores:
            #     if len(sublist) == 0:
            #         print(f'Epmty list in framewise_lm_scores {framewise_lm_scores.index(sublist)}')
            # print(f'Framewise length: {[len(sublist) for sublist in framewise_lm_scores]}')
            print('-' * 10)

            self.log(f'Step {step + 1}/{dataset_len}')
            for key in log_dict:
                self.log(f'{key}: {log_dict[key]}')
                
            self.log('-' * 10)

        if self.cfg.wandb_log:
            wandb.log(log_dict)

    def count_boundary_scores(self, boundary_target, boundary):
        """
        Return:
            f1: float
            precision: float
            recall: float
            1s_ratio (predicted/target): float 
        """
        target = boundary_target.cpu().numpy().reshape(-1)
        prediction = boundary.cpu().numpy().reshape(-1)

        # mask out padding
        mask = target != -1
        target = target[mask]
        prediction = prediction[mask]

        f1 = f1_score(target, prediction)
        precision = precision_score(target, prediction)
        recall = recall_score(target, prediction)

        boundary_1s_ratio = prediction.sum() / target.sum()

        boundary_scores = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            '1s_ratio': boundary_1s_ratio,
        }

        return boundary_scores


    def train_rl_agent_epoch(self, model, dataloader, optimizer, device, scheduler, log_steps, gradient_accumulation_steps):
        """
        Train RL agent for one epoch
        """
        model.segmenter.boundary_predictor.train()
        model.zero_grad()
        example = dataloader.dataset[0]
        if example.get("target", None) is not None:
            print("Training with target")

        for step, sample in enumerate(dataloader):

            # if self.cfg.max_steps_per_epoch is not None and step >= self.cfg.max_steps_per_epoch:
            #     break

            # Get features
            features = sample['net_input']['features']
            features = features.to(device)

            # Get aux targets
            # aux_targets = sample['net_input']['aux_target']
            # aux_targets = aux_targets.to(device)

            # Get padding mask
            padding_mask = sample['net_input']['padding_mask']
            padding_mask = padding_mask.to(device)

            # # Get target
            target = sample.get("target", None)
            # targets = targets.to(device)

            # Get boundary logits
            features, padding_mask, boundary, boundary_logits = model.segmenter.pre_segment(features, padding_mask, return_boundary=True)

            orig_size = features.size(0) * features.size(1) - padding_mask.sum()

            gen_result = self.model.generator(features, None, padding_mask)

            orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
            orig_dense_padding_mask = gen_result["dense_padding_mask"]

            if self.logit_segment:
                dense_x, dense_padding_mask = self.model.segmenter.logit_segment(
                    orig_dense_x, orig_dense_padding_mask
                )
                merge_counts = (~orig_dense_padding_mask).sum(dim=1) - (~dense_padding_mask).sum(dim=1)
                merge_ratio = merge_counts / (~orig_dense_padding_mask).sum(dim=1)
                merge_ratio = merge_ratio.cpu().numpy()
            else:
                dense_x, dense_padding_mask = orig_dense_x, orig_dense_padding_mask
                merge_ratio = np.zeros(orig_dense_padding_mask.size(0))

            batch_size = dense_x.size(0)

            # Get loss
            loss = F.cross_entropy(boundary_logits.reshape(-1, 2), boundary.reshape(-1), ignore_index=-1, reduction='none')
  
            # reshape to batchwise
            loss = loss.reshape(batch_size, -1)

            # Get scores
            scores = self.get_score(dense_x, dense_padding_mask, target=target)

            # Compute reward
            rewards, mean_rewards = self.compute_rewards(scores, boundary, merge_ratio)

            # Get loss * rewards
            loss = loss * rewards

            # Backprop
            loss.sum().backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # Log
            if (step + 1) % log_steps == 0:
                self.log_score(scores, loss.mean().item(), mean_rewards, step, len(dataloader), split='train')

    def validate_rl_agent_epoch(self, model, dataloader, device):
        """
        Validate RL agent for one epoch
        """
        model.eval()
        save_best = False
        total_rewards = 0.0

        # Record scores
        batchwise_lm_ppl = []
        uttwise_lm_ppls = []
        vocab_seen_percentage = []
        framewise_lm_scores = []
        # merge_ratio = []
        target_uttwise_lm_ppls = []
        uttwise_token_error_rates = []
        uttwise_pred_token_lengths = []
        uttwise_target_token_lengths = []
        boundary_f1 = []
        boundary_precision = []
        boundary_recall = []
        boundary_1s_ratio = []
        
        print('Validation')
        self.log('Validation')
        with torch.no_grad():
            for step, sample in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Validating...", dynamic_ncols=True)):

                # if self.cfg.max_val_steps is not None and step >= self.cfg.max_val_steps:
                #     break

                # Get features
                features = sample['net_input']['features']
                features = features.to(device)
                # # Get target
                target = sample.get("target", None)

                if 'aux_target' in sample['net_input']:
                    # Get aux targets
                    aux_targets = sample['net_input']['aux_target']
                    aux_targets = aux_targets.to(device)

                # Get padding mask
                padding_mask = sample['net_input']['padding_mask']
                padding_mask = padding_mask.to(device)

                # Get boundary logits
                features, padding_mask, boundary, boundary_logits = model.segmenter.pre_segment(features, padding_mask, return_boundary=True) # deterministic?

                orig_size = features.size(0) * features.size(1) - padding_mask.sum()

                gen_result = self.model.generator(features, None, padding_mask)

                orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
                orig_dense_padding_mask = gen_result["dense_padding_mask"]

                if self.logit_segment:
                    dense_x, dense_padding_mask = self.model.segmenter.logit_segment(
                        orig_dense_x, orig_dense_padding_mask
                    )
                    merge_counts = (~orig_dense_padding_mask).sum(dim=1) - (~dense_padding_mask).sum(dim=1)
                    merge_ratio = merge_counts / (~orig_dense_padding_mask).sum(dim=1)
                    merge_ratio = merge_ratio.cpu().numpy()
                else:
                    dense_x, dense_padding_mask = orig_dense_x, orig_dense_padding_mask
                    merge_ratio = np.zeros(orig_dense_padding_mask.size(0))

                if 'aux_target' in sample['net_input']:
                    # Count boundary scores
                    boundary_scores = self.count_boundary_scores(aux_targets, boundary)
                # Get scores
                scores = self.get_score(dense_x, dense_padding_mask, target=target)
                # Get rewards
                rewards, mean_rewards = self.compute_rewards(scores, boundary, merge_ratio)

                # Record scores
                total_rewards += mean_rewards
                batchwise_lm_ppl.append(scores["batchwise_lm_ppl"])
                uttwise_lm_ppls.extend(scores["uttwise_lm_ppls"])
                vocab_seen_percentage.append(scores["vocab_seen_percentage"])
                # mean of framewise_lm_scores of each utterance
                framewise_lm_scores.extend([np.mean(x) for x in scores["framewise_lm_scores"]])
                # merge_ratio.append(merge_ratio)
                target_uttwise_lm_ppls.extend(scores["target_uttwise_lm_ppls"])
                uttwise_token_error_rates.extend(scores["uttwise_token_error_rates"])
                uttwise_pred_token_lengths.extend(scores["uttwise_pred_token_lengths"])
                uttwise_target_token_lengths.extend(scores["uttwise_target_token_lengths"])
                if 'aux_target' in sample['net_input']:
                    boundary_f1.append(boundary_scores['f1'])
                    boundary_precision.append(boundary_scores['precision'])
                    boundary_recall.append(boundary_scores['recall'])
                    boundary_1s_ratio.append(boundary_scores['1s_ratio'])

                # Log
                # self.log_score(scores, 0.0, mean_rewards, step, len(dataloader), split='val', boundary_scores=boundary_scores, do_log=False)

        # count custum reward
        relative_lm_ppl = np.array(uttwise_lm_ppls) - np.array(target_uttwise_lm_ppls)
        ppl_outperform_ratio = (relative_lm_ppl < 0).sum() / len(relative_lm_ppl)
        relative_length_ratio = np.array(uttwise_pred_token_lengths) / np.array(uttwise_target_token_lengths)
        length_diff_ratio = np.abs(relative_length_ratio - 1)
        avg_rewards = total_rewards / len(dataloader)

        # evaluation metrics: -batchwise_lm_ppl / vocab_seen_percentage
        eval_score = -np.mean(batchwise_lm_ppl) / np.mean(vocab_seen_percentage)

        avg_boundary_scores = {
            'f1': np.mean(boundary_f1),
            'precision': np.mean(boundary_precision),
            'recall': np.mean(boundary_recall),
            '1s_ratio': np.mean(boundary_1s_ratio),
        }
        # Record average scores
        eval_dict = {
            "run_name": self.cfg.save_dir.split('/')[-1],
            "model_name": self.model_name,
            "batchwise_lm_ppl": np.mean(batchwise_lm_ppl),
            "uttwise_lm_ppl": np.mean(uttwise_lm_ppls),
            "vocab_seen_percentage": np.mean(vocab_seen_percentage),
            "framewise_lm_scores": np.mean(framewise_lm_scores),
            # "merge_ratio": np.mean(merge_ratio),
            "target_uttwise_lm_ppl": np.mean(target_uttwise_lm_ppls),
            "uttwise_token_error_rate": np.mean(uttwise_token_error_rates),
            "uttwise_pred_token_length": np.mean(uttwise_pred_token_lengths),
            "uttwise_target_token_length": np.mean(uttwise_target_token_lengths),
            "relative_lm_ppl": np.mean(relative_lm_ppl),
            "ppl_outperform_ratio": ppl_outperform_ratio,
            "relative_length_ratio": np.mean(relative_length_ratio),
            "length_diff_ratio": np.mean(length_diff_ratio),
            "boundary_f1": avg_boundary_scores['f1'],
            "boundary_precision": avg_boundary_scores['precision'],
            "boundary_recall": avg_boundary_scores['recall'],
            "boundary_1s_ratio": avg_boundary_scores['1s_ratio'],
            "avg_rewards": avg_rewards,
            "eval_score": eval_score,
        }

        if eval_score > self.best_valid_score:
            self.best_valid_score = eval_score
            save_best = True
            self.log(f"Best validation score: {self.best_valid_score}")


            # Record for validation csv ('rl_agent_segmenter_best')
            self.eval_dict_best = eval_dict.copy()
            # Change model name
            self.eval_dict_best['model_name'] = 'rl_agent_segmenter_best'

        
        # Log scores
        print(f'Validation')
        for key in eval_dict:
            print(f'{key}: {eval_dict[key]}')
        print('-' * 10)
        self.log(f'Validation')
        for key in eval_dict:
            self.log(f'{key}: {eval_dict[key]}')
        self.log('-' * 10)

        if self.cfg.wandb_log:
            wandb.log({
                "val/" + key: eval_dict[key] for key in eval_dict
            })

        # Save scores
        self.val_score_df = self.val_score_df.append(eval_dict, ignore_index=True)
        self.val_score_df.to_csv(self.val_score_csv, index=False)
        # Flush csv
        self.val_score_csv.flush()
        
        return save_best
    
    def register_and_setup_task(self, task_cfg_fpath, env):
        task_cfg = OmegaConf.load(task_cfg_fpath)
        task_cfg.fairseq.common.user_dir = f"{env.WORK_DIR}/s2p"
        task_cfg.fairseq.task.text_data = f"{env.WORK_DIR}/rl/dict/{task_cfg.fairseq.task.text_data}" # "librispeech" or "timit_matched" or "timit_unmatched"
        utils.import_user_module(task_cfg.fairseq.common)
        task = tasks.setup_task(task_cfg.fairseq.task)
        return task, task_cfg

    def load_pretrained_model(self):

        env = OmegaConf.load(self.cfg.env)
        task_cfg_fpath = f"{env.WORK_DIR}/rl/config/{self.cfg.config_name}.yaml"
        task, task_cfg = self.register_and_setup_task(task_cfg_fpath, env)
        print(task_cfg)
        # Load model
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
        [self.cfg.pretrain_wav2vecu_path],
            task=task
        )
        self.model = models[0]
        self.model.eval()
        
        # Load wav2vec-u model (including generator and discriminator)
        # print(torch.load(self.cfg.pretrain_wav2vecu_path))
        self.model.load_state_dict(torch.load(self.cfg.pretrain_wav2vecu_path)['model'])
        
        self.model.segmenter = CnnSegmenter(SegmentationConfig(), CnnBoundaryConfig(input_dim=512))

        for p in self.model.generator.parameters():
            p.param_group = "generator"
            # fix the generator
            p.requires_grad = False

        for p in self.model.segmenter.parameters():
            p.param_group = "segmenter"

        # Check if the pre-trained model exists
        if (self.cfg.pretrain_segmenter_path is not None) and (self.cfg.pretrain_segmenter_path != "None") and not os.path.exists(self.cfg.pretrain_segmenter_path):
            raise Exception(f"Cannot find {self.cfg.pretrain_segmenter_path}")

        if (self.cfg.pretrain_segmenter_path is None) or (self.cfg.pretrain_segmenter_path == "None"):
            print("No pre-trained segmenter, use random initialization")
            self.model.segmenter.boundary_predictor.train()
            return

        # Load pre-trained CNN model
        try:
            self.model.segmenter.boundary_predictor.load_state_dict(torch.load(self.cfg.pretrain_segmenter_path))
        except:
            print(f"Cannot load {self.cfg.pretrain_segmenter_path} by boundary predictor, try to load by segmenter")
            self.model.segmenter.load_state_dict(torch.load(self.cfg.pretrain_segmenter_path))

        self.model.segmenter.boundary_predictor.train()

    def train_rl_agent(self):
        """
        Given the pre-trained CNN model, train the RL agent
        """

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f'Using device: {device}')

        # Load pre-trained CNN model
        self.load_pretrained_model()
        self.model.to(device)

        # Audio features path
        dir_path = f'{self.cfg.data_dir}/precompute_pca512'
        # Boundary labels path
        boundary_labels_path = f'{self.cfg.data_dir}/CLUS128'

        # Load Extracted features
        train_dataset = ExtractedFeaturesDataset(
            path=dir_path,
            split='train',
            labels=self.cfg.w2vu_postfix,
            label_dict=self.scorer.dictionary,
            aux_target_postfix='bds',
            aux_target_dir_path=boundary_labels_path,
        )

        valid_dataset = ExtractedFeaturesDataset(
            path=dir_path,
            split='valid',
            labels=self.cfg.w2vu_postfix,
            label_dict=self.scorer.dictionary,
            aux_target_postfix='bds',
            aux_target_dir_path=boundary_labels_path,
        )

        # Hyperparameters
        BATCH_SIZE = 128
        NUM_EPOCHS = self.cfg.num_epochs
        LEARNING_RATE = self.cfg.learning_rate
        WEIGHT_DECAY = 1e-4
        GRADIENT_ACCUMULATION_STEPS = 1
        LOG_STEPS = 1
        SAVE_INTERVAL = self.cfg.save_interval
        STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE + 1
        MAX_STEPS_PER_EPOCH = None
        MAX_VAL_STEPS = len(valid_dataset) // BATCH_SIZE + 1

        # wandb config update
        self.cfg.batch_size = BATCH_SIZE
        # self.cfg.num_epochs = NUM_EPOCHS
        # self.cfg.learning_rate = LEARNING_RATE
        self.cfg.weight_decay = WEIGHT_DECAY
        self.cfg.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
        self.cfg.log_steps = LOG_STEPS
        # self.cfg.save_interval = SAVE_INTERVAL
        self.cfg.steps_per_epoch = STEPS_PER_EPOCH
        self.cfg.max_steps_per_epoch = MAX_STEPS_PER_EPOCH
        self.cfg.max_val_steps = MAX_VAL_STEPS
        self.cfg.num_train_data = len(train_dataset)
        self.cfg.num_val_data = len(valid_dataset)
        if self.cfg.wandb_log:
            wandb.config.update(vars(self.cfg))
            self.log(vars(self.cfg))

        # Log configuration to file
        with open(os.path.join(self.cfg.save_dir, "config.yaml"), "w") as f:
            import yaml
            yaml.dump(vars(self.cfg), f)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS_PER_EPOCH * (NUM_EPOCHS - self.cfg.start_epoch))
        self.best_valid_score = float('-inf')

        # Load data
        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=1,
            collate_fn=train_dataset.collater,
            shuffle=True,
            batch_size=BATCH_SIZE,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            num_workers=1,
            collate_fn=valid_dataset.collater,
            shuffle=False,
            batch_size=1 # for evaluation
        )

        # Create save directory
        if not os.path.exists(self.cfg.save_dir):
            os.makedirs(self.cfg.save_dir)

        # log configuration
        self.log(vars(self.cfg))

        if self.apply_merge_penalty:
            print("Will apply merge penalty.")
        
        # Validate
        self.model_name = f'rl_agent_segmenter_epoch{self.cfg.start_epoch}.pt'
        self.validate_rl_agent_epoch(self.model, self.valid_dataloader, device)

        torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter_epoch{}.pt'.format(self.cfg.start_epoch))

        # Train Policy Gradient
        for epoch in range(self.cfg.start_epoch, NUM_EPOCHS):
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
            print('-' * 10)

            # Train
            self.train_rl_agent_epoch(self.model, self.train_dataloader, optimizer, device, scheduler, LOG_STEPS, GRADIENT_ACCUMULATION_STEPS)

            # Validate
            if (epoch + 1) % SAVE_INTERVAL == 0:

                # Validate
                self.model_name = 'rl_agent_segmenter_epoch{}.pt'.format(epoch+1)
                save_best = self.validate_rl_agent_epoch(self.model, self.valid_dataloader, device)

                # Save model
                torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter_epoch{}.pt'.format(epoch+1))

                # Save model
                if save_best:
                    torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter_best.pt'.format(epoch))
                    print('Save best model at epoch {}'.format(epoch+1))
                    self.log('Save best model at epoch {}'.format(epoch+1))
            

        # Save model
        # torch.save(self.model.state_dict(), self.cfg.save_dir + '/rl_agent.pt')
        torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter.pt')

        # Add best model to val_score_df
        self.val_score_df = self.val_score_df.append(self.eval_dict_best, ignore_index=True)
        self.val_score_df.to_csv(self.val_score_csv, index=False)
        # Flush csv
        self.val_score_csv.flush()
        

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="librispeech")
    parser.add_argument("--data_dir", type=str, default="../../data/audio/ls_100h_clean/large_clean")
    parser.add_argument("--kenlm_fpath", type=str, default="../../data/text/ls_wo_lv/prep_g2p/phones/lm.phones.filtered.04.bin")
    parser.add_argument("--dict_fpath", type=str, default="../dummy_data/dict.txt")
    parser.add_argument("--pretrain_segmenter_path", type=str, default="./output/cnn_segmenter/pretrain_PCA_postITER1_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_29_0.pt")
    parser.add_argument("--pretrain_wav2vecu_path", type=str, default="../../s2p/multirun/ls_100h/large_clean_postITER1/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed1/checkpoint_best.pt")
    parser.add_argument("--w2vu_postfix", type=str, default="w2vu_logit_segmented")
    parser.add_argument("--env", type=str, default="../../env.yaml")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ter_tolerance", type=float, default=0.0)
    parser.add_argument("--length_tolerance", type=float, default=0.0)
    parser.add_argument("--logit_segment", type=bool, default=True)
    parser.add_argument("--apply_merge_penalty", type=bool, default=False) 
    # store_false: apply merge penalty, if no-apply_merge_penalty, then store_false
    parser.add_argument('--no-apply_merge_penalty', dest='apply_merge_penalty', action='store_false')
    parser.add_argument("--wandb_log", type=bool, default=True)
    parser.add_argument("--utterwise_lm_ppl_coeff", type=float, default=1.0)
    parser.add_argument("--utterwise_token_error_rate_coeff", type=float, default=1.0)
    parser.add_argument("--length_ratio_coeff", type=float, default=0.0)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="./output/rl_agent/ls_100h_clean_postITER1/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--rm_sil", type=bool, default=False)
    parser.add_argument("--ter_rm_sil", type=bool, default=False)
    args = parser.parse_args()

    rl_cfg = RLCnnAgentConfig()
    for arg in vars(args):
        setattr(rl_cfg, arg, getattr(args, arg))
    train_rl_agent = TrainRlCnnAgent(rl_cfg)
    train_rl_agent.train_rl_agent() 

    