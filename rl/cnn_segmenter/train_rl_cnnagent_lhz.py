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
import wandb

class RLCnnAgentConfig(object):
    # config_name: str = "librispeech" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = "../../data/audio/ls_100h_clean/large_clean/precompute_pca512"
    # kenlm_fpath: str = "../../data/text/ls_wo_lv/prep_g2p/phones/lm.phones.filtered.04.bin"
    # dict_fpath: str = "../dummy_data/dict.txt"
    # pretrain_segmenter_path: str = "./output/cnn_segmenter/pretrain_PCA_postITER1_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_29_0.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/ls_100h/large_clean_postITER1/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed1/checkpoint_best.pt"
    # w2vu_postfix: str = "w2vu_logit_segmented"
    # ----------------------------------------------------
    # config_name: str = "timit_matched" # "librispeech" or "timit_matched" or "timit_unmatched"
    # data_dir: str = "../../data/audio/timit/matched/large_clean/precompute_pca512"
    # kenlm_fpath: str = "../../data/text/timit/matched/phones/train_text_phn.04.bin"
    # dict_fpath: str = "../dict/timit_matched/dict.txt"
    # pretrain_segmenter_path: str = "./output/local/cnn_segmenter/timit_matched_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo100_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt"
    # pretrain_wav2vecu_path: str = "../../s2p/multirun/timit_matched/large_clean/timit_paired_no_SA/cp4_gp1.5_sw0.5/seed3/checkpoint_best.pt"
    # save_dir: str = "./output/local/rl_agent/timit_matched_from_bc_relative_to_wfst_length_only"
    # w2vu_postfix: str = "wfst_decoded"
    # ----------------------------------------------------
    config_name: str = "timit_unmatched" # "librispeech" or "timit_matched" or "timit_unmatched"
    data_dir: str = "../../data/audio/timit/unmatched/large_clean/precompute_pca512"
    kenlm_fpath: str = "../../data/text/timit/unmatched/phones/train_text_phn.04.bin"
    dict_fpath: str = "../dict/timit_unmatched/dict.txt"
    pretrain_segmenter_path: str = "./output/local/cnn_segmenter/timit_unmatched_pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo80_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter.pt"
    pretrain_wav2vecu_path: str = "../../s2p/multirun/timit_unmatched/large_clean/timit_unpaired_1k/cp4_gp2.0_sw0.5/seed2/checkpoint_best.pt"
    save_dir: str = "./output/local/rl_agent/timit_unmatched_from_bc_relative_to_wfst_length_only"
    w2vu_postfix: str = "wfst_decoded"

    env: str = "../../env.yaml"
    gamma: float = 1.0
    ter_tolerance: float = 0.08
    logit_segment: bool = True
    apply_merge_penalty: bool = False
    wandb_log: bool = True
    utterwise_lm_ppl_coeff: float = 1.0
    utterwise_token_error_rate_coeff: float = 0.0
    length_ratio_coeff: float = 0.5

class TrainRlCnnAgent(object):
    def __init__(self, cfg: RLCnnAgentConfig):
        self.score_cfg = ScorerCfg(
            kenlm_fpath=cfg.kenlm_fpath,
            dict_fpath=cfg.dict_fpath,
        )
        self.scorer = Scorer(self.score_cfg)
        self.cfg = cfg
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
        
        scores = self.scorer.score(result)
        
        # print(scores['batchwise_lm_ppl'], scores['token_error_rate'], scores['vocab_seen_percentage'], scores['framewise_lm_scores'][0][:5]) # ppl should be very high 😱 
        # ##############################
        # print('-' * 10)
        # print('Batchwise LM PPL (should be very high): ', scores['batchwise_lm_ppl'])
        # # Mean of uttwise LM PPL (list of float)
        # print('Mean uttwise LM PPL: ', sum(scores['uttwise_lm_ppls']) / len(scores['uttwise_lm_ppls']))
        # # print('Uttwise LM PPL: ', scores['uttwise_lm_ppls'][:5])
        # if tgt_ids is not None:
        #     print('Token error rate: ', scores['token_error_rate'])
        # print('Vocab seen percentage: ', scores['vocab_seen_percentage'])
        # print('Mean framewise LM scores: ', sum([(sum(sublist) / len(sublist))  for sublist in scores['framewise_lm_scores']]) / len(scores['framewise_lm_scores']))
        # # print('Framewise LM scores: ', scores['framewise_lm_scores'][0][:5])
        # print('-' * 10)
        # ##############################

        return scores
    
    def compute_rewards(self, scores, boundary, merge_ratio):
        """
        Compute rewards from scores
        Return:
            rewards: torch.tensor (Flattened framewise rewards)
        """
        # Compute reward: 
        # batchwise_lm_ppl = scores['batchwise_lm_ppl']
        if not self.apply_merge_penalty:
            merge_ratio = np.zeros(merge_ratio.shape)
        uttwise_lm_ppls = scores['uttwise_lm_ppls']
        target_uttwise_lm_ppls = scores['target_uttwise_lm_ppls']
        uttwise_token_error_rates = scores['uttwise_token_error_rates']
        uttwise_pred_token_lengths = scores['uttwise_pred_token_lengths']
        uttwise_target_token_lengths = scores['uttwise_target_token_lengths']
        length_ratio = uttwise_pred_token_lengths / uttwise_target_token_lengths
        # vocab_seen_percentage = scores['vocab_seen_percentage']
        # framewise_lm_scores = scores['framewise_lm_scores']

        # flatten framewise_lm_scores
        # framewise_lm_scores = [item for sublist in framewise_lm_scores for item in sublist]
        
        # framewise_reward = torch.tensor(framewise_lm_scores).to(self.device)
        # unpenalized_mask = (uttwise_token_error_rates < self.cfg.ter_tolerance)
        # uttwise_token_error_rates[unpenalized_mask] = 0.0
        # ter_penalty = uttwise_token_error_rates
        
        # uttwise_lm_ppls = np.array(uttwise_lm_ppls)
        # target_uttwise_lm_ppls = np.array(target_uttwise_lm_ppls)
        
        # if len(target_uttwise_lm_ppls) == len(uttwise_lm_ppls):
        #     uttwise_rewards = target_uttwise_lm_ppls - uttwise_lm_ppls
        #     # clip rewards
        #     uttwise_rewards = np.clip(uttwise_rewards, -5, 5)
        #     positive_rewards_mask = (uttwise_rewards >= 0)
        #     uttwise_rewards[positive_rewards_mask]  = uttwise_rewards[positive_rewards_mask]  * (1 - ter_penalty[positive_rewards_mask])  * (1 - merge_ratio[positive_rewards_mask])
        #     uttwise_rewards[~positive_rewards_mask] = uttwise_rewards[~positive_rewards_mask] * (1 + ter_penalty[~positive_rewards_mask]) * (1 + merge_ratio[~positive_rewards_mask])
        # else:
        #     normed_uttwise_lm_ppls = (uttwise_lm_ppls - uttwise_lm_ppls.mean()) / uttwise_lm_ppls.std()
        #     uttwise_rewards = -normed_uttwise_lm_ppls
        # uttwise_rewards = torch.tensor(uttwise_rewards, dtype=torch.float32).to(self.device)
        
        uttwise_lm_ppls = torch.tensor(uttwise_lm_ppls, dtype=torch.float32).to(self.device)
        uttwise_token_error_rates = torch.tensor(uttwise_token_error_rates, dtype=torch.float32).to(self.device)
        length_ratio = torch.tensor(length_ratio, dtype=torch.float32).to(self.device)
        target_uttwise_lm_ppls = torch.tensor(target_uttwise_lm_ppls, dtype=torch.float32).to(self.device)

        length_ratio_loss = torch.abs(length_ratio - 1)

        # print(framewise_reward.shape)
        # print(uttwise_lm_ppls.shape)

        # reward standardization
        # framewise_reward = (framewise_reward - framewise_reward.mean()) / framewise_reward.std()
        if len(target_uttwise_lm_ppls) == len(uttwise_lm_ppls):
            uttwise_lm_ppls = uttwise_lm_ppls - target_uttwise_lm_ppls
            # clip rewards
            uttwise_lm_ppls = torch.clamp(uttwise_lm_ppls, -5, 5)
        else:
            uttwise_lm_ppls = (uttwise_lm_ppls - uttwise_lm_ppls.mean()) / uttwise_lm_ppls.std()
        uttwise_token_error_rates = (uttwise_token_error_rates - uttwise_token_error_rates.mean()) / uttwise_token_error_rates.std()
        length_ratio_loss = (length_ratio_loss - length_ratio_loss.mean()) / length_ratio_loss.std()
        
        uttwise_rewards = - uttwise_lm_ppls * self.cfg.utterwise_lm_ppl_coeff - uttwise_token_error_rates * self.cfg.utterwise_token_error_rate_coeff - length_ratio_loss * self.cfg.length_ratio_coeff
        

        # reward standardization
        # framewise_reward = (framewise_reward - framewise_reward.mean()) / framewise_reward.std()
        
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
        # bsz, tsz = boundary.size(0), boundary.size(1)
        # for i in range(bsz):
        #     for j in range(tsz):
        #         # if boundary[i, j] == 1:
        #             # rewards[i, j] = framewise_reward[count]
        #             # count += 1
        #         if boundary[i, j] == -1 or j == tsz - 1:
        #             # rewards[i, j] = framewise_reward[count]
        #             rewards[i, j] = uttwise_rewards[i]
        #             # count += 1
        #             break

        # cumulative reward (gamma=0.99)
        cum_rewards = torch.zeros_like(rewards, dtype=torch.float32).to(self.device)

        reward_len = rewards.size(1)
        cum_rewards[:, reward_len-1] = rewards[:, reward_len-1]
        for i in range(reward_len-2, -1, -1):
            cum_rewards[:, i] = rewards[:, i] + self.cfg.gamma * cum_rewards[:, i+1]
        # print(cum_rewards)

        # return rewards
        return cum_rewards, uttwise_rewards

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

            # # Get target length
            # target_lengths = sample['target_lengths']
            # target_lengths = target_lengths.to(device)

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
            # print(boundary_logits.shape)
            # print(boundary.shape)
            # boundary_logits = boundary_logits.reshape(-1, 2)
            # boundary = boundary.reshape(-1)
            # mask
            # boundary_logits = boundary_logits[boundary != -1]
            # boundary = boundary[boundary != -1]

            # Get loss
            loss = F.cross_entropy(boundary_logits.reshape(-1, 2), boundary.reshape(-1), ignore_index=-1, reduction='none')
            # print(loss.shape)
            # print(loss)
            # reshape to batchwise
            loss = loss.reshape(batch_size, -1)

            # Get scores
            scores = self.get_score(dense_x, dense_padding_mask, target=target)

            # Compute reward
            rewards, _ = self.compute_rewards(scores, boundary, merge_ratio)

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
                batchwise_lm_ppl = scores['batchwise_lm_ppl']
                uttwise_lm_ppls = scores['uttwise_lm_ppls']
                mean_uttwise_lm_ppl = sum(uttwise_lm_ppls) / len(uttwise_lm_ppls)
                vocab_seen_percentage = scores['vocab_seen_percentage']
                framewise_lm_scores = scores['framewise_lm_scores']
                mean_framewise_lm_scores = sum([(sum(sublist) / len(sublist))  for sublist in framewise_lm_scores if len(sublist) > 0]) / len(framewise_lm_scores)

                print(f'Step {step + 1}/{len(dataloader)}')
                print(f'Loss: {loss.sum().item()}')
                print(f'Mean reward: {rewards.mean().item()}')
                print(f'Batchwise LM PPL: {batchwise_lm_ppl}')
                print(f'Mean uttwise LM PPL: {mean_uttwise_lm_ppl}')
                print(f'Vocab seen percentage: {vocab_seen_percentage}')
                print(f'Mean framewise LM scores: {mean_framewise_lm_scores}')
                print(f'Framewise LM scores: {framewise_lm_scores[0][:5]}')
                # check if there is empty list in framewise_lm_scores
                for sublist in framewise_lm_scores:
                    if len(sublist) == 0:
                        print(f'Epmty list in framewise_lm_scores {framewise_lm_scores.index(sublist)}')
                print('-' * 10)
                if self.cfg.wandb_log:
                    wandb.log(
                        {
                            "loss": loss.sum().item(),
                            "reward": rewards.mean().item(),
                            "train": {
                                "batchwise_lm_ppl": batchwise_lm_ppl,
                                "uttwise_lm_ppls": mean_uttwise_lm_ppl,
                                "vocab_seen_percentage": vocab_seen_percentage,
                                "framewise_lm_scores": mean_framewise_lm_scores,
                            }
                        }
                    )

    def validate_rl_agent_epoch(self, model, dataloader, device):
        """
        Validate RL agent for one epoch
        """
        model.eval()
        save_best = False
        total_rewards = 0.0
        with torch.no_grad():
            for step, sample in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Validating...", dynamic_ncols=True)):

                # if self.cfg.max_val_steps is not None and step >= self.cfg.max_val_steps:
                #     break

                # Get features
                features = sample['net_input']['features']
                features = features.to(device)
                # # Get target
                target = sample.get("target", None)
                # Get aux targets
                # aux_targets = sample['net_input']['aux_target']
                # aux_targets = aux_targets.to(device)

                # Get padding mask
                padding_mask = sample['net_input']['padding_mask']
                padding_mask = padding_mask.to(device)

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
                # Log
                # print('Validation')
                # print(f'Step {step + 1}/{len(dataloader)}')

                # Get scores
                scores = self.get_score(dense_x, dense_padding_mask, target=target)
                # Get rewards
                _, uttwise_rewards = self.compute_rewards(scores, boundary, merge_ratio)

                total_rewards += uttwise_rewards.sum().item()

                batchwise_lm_ppl = scores['batchwise_lm_ppl']
                uttwise_lm_ppls = scores['uttwise_lm_ppls']
                mean_uttwise_lm_ppl = sum(uttwise_lm_ppls) / len(uttwise_lm_ppls)
                vocab_seen_percentage = scores['vocab_seen_percentage']
                framewise_lm_scores = scores['framewise_lm_scores']
                mean_framewise_lm_scores = sum([(sum(sublist) / len(sublist))  for sublist in framewise_lm_scores if len(sublist) > 0]) / len(framewise_lm_scores)

                # print(f'Batchwise LM PPL: {batchwise_lm_ppl}')
                # print(f'Mean uttwise LM PPL: {mean_uttwise_lm_ppl}')
                # print(f'Vocab seen percentage: {vocab_seen_percentage}')
                # print(f'Mean framewise LM scores: {mean_framewise_lm_scores}')
                
                # print('-' * 10)
                if self.cfg.wandb_log:
                    wandb.log(
                        { "val": {
                            "merge_ratio": merge_ratio.mean().item(),
                            "batchwise_lm_ppl": batchwise_lm_ppl,
                            "uttwise_lm_ppls": mean_uttwise_lm_ppl,
                            "vocab_seen_percentage": vocab_seen_percentage,
                            "framewise_lm_scores": mean_framewise_lm_scores,
                        }}
                    )

        if total_rewards > self.best_valid_score:
            self.best_valid_score = total_rewards
            save_best = True
        print(f"total_rewards: {total_rewards}")
        self.log(f"total_rewards: {total_rewards}")
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
        dir_path = self.cfg.data_dir
        # Boundary labels path
        boundary_labels_path = f'../../data/audio/ls_100h_clean/large_clean_mfa/CLUS128'

        # Load Extracted features
        train_dataset = ExtractedFeaturesDataset(
            path=dir_path,
            split='train',
            labels=self.cfg.w2vu_postfix,
            label_dict=self.scorer.dictionary,
            aux_target_postfix='boundaries',
            aux_target_dir_path=boundary_labels_path,
        )

        valid_dataset = ExtractedFeaturesDataset(
            path=dir_path,
            split='valid',
            labels=self.cfg.w2vu_postfix,
            label_dict=self.scorer.dictionary,
            aux_target_postfix='boundaries',
            aux_target_dir_path=boundary_labels_path,
        )

        # Hyperparameters
        BATCH_SIZE = 128
        NUM_EPOCHS = 200
        LEARNING_RATE = 1e-5
        WEIGHT_DECAY = 1e-4
        GRADIENT_ACCUMULATION_STEPS = 1
        LOG_STEPS = 1
        STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE + 1
        MAX_STEPS_PER_EPOCH = None
        MAX_VAL_STEPS = len(valid_dataset) // BATCH_SIZE + 1

        # wandb config update
        self.cfg.batch_size = BATCH_SIZE
        self.cfg.num_epochs = NUM_EPOCHS
        self.cfg.learning_rate = LEARNING_RATE
        self.cfg.weight_decay = WEIGHT_DECAY
        self.cfg.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
        self.cfg.log_steps = LOG_STEPS
        self.cfg.steps_per_epoch = STEPS_PER_EPOCH
        self.cfg.max_steps_per_epoch = MAX_STEPS_PER_EPOCH
        self.cfg.max_val_steps = MAX_VAL_STEPS
        self.cfg.num_train_data = len(train_dataset)
        self.cfg.num_val_data = len(valid_dataset)
        if self.cfg.wandb_log:
            wandb.config.update(self.cfg)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS_PER_EPOCH * NUM_EPOCHS)
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
            batch_size=BATCH_SIZE,
        )

        # Create save directory
        if not os.path.exists(self.cfg.save_dir):
            os.makedirs(self.cfg.save_dir)

        if self.apply_merge_penalty:
            print("Will apply merge penalty.")
        # Validate
        # self.validate_rl_agent_epoch(self.model, self.valid_dataloader, device)
        # Train Policy Gradient
        for epoch in range(NUM_EPOCHS):
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
            print('-' * 10)

            # Train
            self.train_rl_agent_epoch(self.model, self.train_dataloader, optimizer, device, scheduler, LOG_STEPS, GRADIENT_ACCUMULATION_STEPS)

            # Validate
            save_best = self.validate_rl_agent_epoch(self.model, self.valid_dataloader, device)

            # Save model
            if save_best:
                torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter_best.pt'.format(epoch))
            if NUM_EPOCHS > 40:
                if epoch % 5 == 0:
                    torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter_epoch{}.pt'.format(epoch))

        # Save model
        torch.save(self.model.state_dict(), self.cfg.save_dir + '/rl_agent.pt')
        torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter.pt')
        


if __name__ == "__main__":
    # for i in range(20):
    #     segmenter_ckpt_path = f"/home/andybi7676/Desktop/uasr-rl/rl/cnn_segmenter/output/local/rl_agent/uttwise_reward_with_ed_fixsample_less_ter_larger_clip_val_with_merge_penalty/rl_agent_segmenter_epoch{i}.pt"
    rl_cfg = RLCnnAgentConfig()
    # rl_cfg.pretrain_segmenter_path = segmenter_ckpt_path
    train_rl_agent = TrainRlCnnAgent(rl_cfg)
    train_rl_agent.train_rl_agent() 

    