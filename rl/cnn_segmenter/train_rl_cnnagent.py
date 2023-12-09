import sys
sys.path.append('../')
sys.path.append('../reward')
import os
from cnn_model import CnnSegmenter, CnnBoundaryPredictor, CnnBoundaryConfig, SegmentationConfig
from load_dataset import ExtractedFeaturesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from reward.scorer import Scorer, ScorerCfg
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from omegaconf import OmegaConf
from fairseq import checkpoint_utils, tasks, utils

import wandb

class RLCnnAgentConfig(object):
    kenlm_fpath: str = "../../data/text/prep_g2p/phones/lm.phones.filtered.04.bin"
    dict_fpath: str = "../dummy_data/dict.txt"
    pretrain_segmenter_path: str = "./output/cnn_segmenter/pretrain_PCA_cnn_segmenter_kernel_size_7_v1_epo30_lr0.0001_wd0.0001_dropout0.1_optimAdamW_schCosineAnnealingLR/cnn_segmenter_29_0.pt"
    pretrain_wav2vecu_path: str = "../../s2p/multirun/ls_100h/large_clean/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed3/checkpoint_best.pt"
    save_dir: str = "./output/rl_agent/uttwise_reward_fixsample"
    env: str = "../../env.yaml"
    gamma: float = 0.99
    wandb_log: bool = True

# class Wav2vec_U_with_CnnSegmenter(Wav2vec_U):
#     def __init__(self, cfg: Wav2vec_UConfig, target_dict):
#         super().__init__(cfg, target_dict)

#         self.cfg = cfg
#         self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
#         self.smoothness_weight = cfg.smoothness_weight

#         output_size = len(target_dict)
#         self.pad = target_dict.pad()
#         self.eos = target_dict.eos()
#         self.ncritic = cfg.ncritic
#         self.smoothing = cfg.smoothing
#         self.smoothing_one_sided = cfg.smoothing_one_sided
#         self.no_softmax = cfg.no_softmax
#         self.gumbel = cfg.gumbel
#         self.hard_gumbel = cfg.hard_gumbel
#         self.last_acc = None

#         self.gradient_penalty = cfg.gradient_penalty
#         self.code_penalty = cfg.code_penalty
#         self.blank_weight = cfg.blank_weight
#         self.blank_mode = cfg.blank_mode
#         self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
#         assert self.blank_index != target_dict.unk()

#         self.pca_A = self.pca_b = None
#         d = cfg.input_dim

#         self.segmenter = CnnSegmenter(SegmentationConfig(), CnnBoundaryConfig())

#         self.generator = Generator(d, output_size, cfg)

#         for p in self.generator.parameters():
#             p.param_group = "generator"
#             # fix the generator
#             p.requires_grad = False

#         for p in self.segmenter.parameters():
#             p.param_group = "segmenter"

#         self.max_temp, self.min_temp, self.temp_decay = cfg.temp
#         self.curr_temp = self.max_temp
#         self.update_num = 0


class TrainRlCnnAgent(object):
    def __init__(self, cfg: RLCnnAgentConfig):
        self.score_cfg = ScorerCfg(
            kenlm_fpath=cfg.kenlm_fpath,
            dict_fpath=cfg.dict_fpath,
        )
        self.scorer = Scorer(self.score_cfg)
        self.cfg = cfg
        if self.cfg.wandb_log:
            wandb.init(
                project="uasr-rl",
                name=cfg.save_dir.split('/')[-1],
                config=cfg,
            )

    def get_score(self, pred_logits, padding_mask, tgt_ids=None):
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
        if tgt_ids is not None:
            result["target"] = tgt_ids
        
        scores = self.scorer.score(result, rm_sil=True)
        
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
    
    def compute_rewards(self, scores, boundary):
        """
        Compute rewards from scores
        Return:
            rewards: torch.tensor (Flattened framewise rewards)
        """
        # Compute reward: 
        # batchwise_lm_ppl = scores['batchwise_lm_ppl']
        uttwise_lm_ppls = scores['uttwise_lm_ppls']
        # vocab_seen_percentage = scores['vocab_seen_percentage']
        # framewise_lm_scores = scores['framewise_lm_scores']

        # flatten framewise_lm_scores
        # framewise_lm_scores = [item for sublist in framewise_lm_scores for item in sublist]
        
        # framewise_reward = torch.tensor(framewise_lm_scores).to(self.device)
        uttwise_lm_ppls = torch.tensor(uttwise_lm_ppls).to(self.device)

        # print(framewise_reward.shape)
        # print(uttwise_lm_ppls.shape)

        # reward standardization
        # framewise_reward = (framewise_reward - framewise_reward.mean()) / framewise_reward.std()
        uttwise_lm_ppls = (uttwise_lm_ppls - uttwise_lm_ppls.mean()) / uttwise_lm_ppls.std()
        
        
        # reward gained at boundary=1
        rewards = torch.zeros_like(boundary, dtype=torch.float32).to(self.device)

        # for each boundary=1, reward[pos] = framewise_reward[count]
        # count = 0
        for i in range(boundary.size(0)):
            for j in range(boundary.size(1)):
                # if boundary[i, j] == 1:
                    # rewards[i, j] = framewise_reward[count]
                    # count += 1
                if boundary[i, j] == -1:
                    # rewards[i, j] = framewise_reward[count]
                    rewards[i, j] = -uttwise_lm_ppls[i]
                #     count += 1
                    break
        # print(count)
        # assert count == framewise_reward.size(0)
        # print(rewards)

        # cumulative reward (gamma=0.99)
        cum_rewards = torch.zeros_like(rewards, dtype=torch.float32).to(self.device)

        reward_len = rewards.size(1)
        cum_rewards[:, reward_len-1] = rewards[:, reward_len-1]
        for i in range(reward_len-2, -1, -1):
            cum_rewards[:, i] = rewards[:, i] + self.cfg.gamma * cum_rewards[:, i+1]
        # print(cum_rewards)

        # return rewards
        return cum_rewards


    def train_rl_agent_epoch(self, model, dataloader, optimizer, device, scheduler, log_steps, gradient_accumulation_steps):
        """
        Train RL agent for one epoch
        """
        model.segmenter.boundary_predictor.train()
        model.zero_grad()

        for step, sample in enumerate(dataloader):

            if self.cfg.max_steps_per_epoch is not None and step >= self.cfg.max_steps_per_epoch:
                break

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
            # targets = sample['target']
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

            dense_x, dense_padding_mask = self.model.segmenter.logit_segment(
                orig_dense_x, orig_dense_padding_mask
            )

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
            scores = self.get_score(dense_x, dense_padding_mask)

            # Compute reward
            rewards = self.compute_rewards(scores, boundary)

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

        with torch.no_grad():
            for step, sample in enumerate(dataloader):

                if self.cfg.max_val_steps is not None and step >= self.cfg.max_val_steps:
                    break

                # Get features
                features = sample['net_input']['features']
                features = features.to(device)

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

                dense_x, dense_padding_mask = self.model.segmenter.logit_segment(
                    orig_dense_x, orig_dense_padding_mask
                )

                # Log
                print('Validation')
                print(f'Step {step + 1}/{len(dataloader)}')

                # Get scores
                scores = self.get_score(dense_x, dense_padding_mask)

                batchwise_lm_ppl = scores['batchwise_lm_ppl']
                uttwise_lm_ppls = scores['uttwise_lm_ppls']
                mean_uttwise_lm_ppl = sum(uttwise_lm_ppls) / len(uttwise_lm_ppls)
                vocab_seen_percentage = scores['vocab_seen_percentage']
                framewise_lm_scores = scores['framewise_lm_scores']
                mean_framewise_lm_scores = sum([(sum(sublist) / len(sublist))  for sublist in framewise_lm_scores if len(sublist) > 0]) / len(framewise_lm_scores)

                print(f'Batchwise LM PPL: {batchwise_lm_ppl}')
                print(f'Mean uttwise LM PPL: {mean_uttwise_lm_ppl}')
                print(f'Vocab seen percentage: {vocab_seen_percentage}')
                print(f'Mean framewise LM scores: {mean_framewise_lm_scores}')
                
                print('-' * 10)
                if self.cfg.wandb_log:
                    wandb.log(
                        { "val": {
                            "batchwise_lm_ppl": batchwise_lm_ppl,
                            "uttwise_lm_ppls": mean_uttwise_lm_ppl,
                            "vocab_seen_percentage": vocab_seen_percentage,
                            "framewise_lm_scores": mean_framewise_lm_scores,
                        }}
                    )


    def register_and_setup_task(self, task_cfg_fpath, env):
        task_cfg = OmegaConf.load(task_cfg_fpath)
        task_cfg.fairseq.common.user_dir = f"{env.WORK_DIR}/s2p"
        task_cfg.fairseq.task.text_data = f"{env.WORK_DIR}/rl/dummy_data"
        utils.import_user_module(task_cfg.fairseq.common)
        task = tasks.setup_task(task_cfg.fairseq.task)
        return task, task_cfg

    def load_pretrained_model(self):

        env = OmegaConf.load(self.cfg.env)
        task_cfg_fpath = f"{env.WORK_DIR}/rl/config/dummy.yaml"
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
        self.model.segmenter.boundary_predictor.load_state_dict(torch.load(self.cfg.pretrain_segmenter_path))

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
        dir_path = '../../data/audio/ls_100h_clean/large_clean/precompute_pca512'
        # Boundary labels path
        boundary_labels_path = f'../../data/audio/ls_100h_clean/large_clean_mfa/CLUS128'

        # Load Extracted features
        train_dataset = ExtractedFeaturesDataset(
            path=dir_path,
            split='train',
            aux_target_postfix='boundaries',
            aux_target_dir_path=boundary_labels_path,
        )

        valid_dataset = ExtractedFeaturesDataset(
            path=dir_path,
            split='valid',
            aux_target_postfix='boundaries',
            aux_target_dir_path=boundary_labels_path,
        )

        # Hyperparameters
        BATCH_SIZE = 128
        NUM_EPOCHS = 10
        LEARNING_RATE = 1e-5
        WEIGHT_DECAY = 1e-4
        GRADIENT_ACCUMULATION_STEPS = 1
        LOG_STEPS = 1
        STEPS_PER_EPOCH = 222
        MAX_STEPS_PER_EPOCH = None
        MAX_VAL_STEPS = 10

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

        # Validate
        self.validate_rl_agent_epoch(self.model, self.valid_dataloader, device)

        # Train Policy Gradient
        for epoch in range(NUM_EPOCHS):
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
            print('-' * 10)

            # Train
            self.train_rl_agent_epoch(self.model, self.train_dataloader, optimizer, device, scheduler, LOG_STEPS, GRADIENT_ACCUMULATION_STEPS)

            # Validate
            self.validate_rl_agent_epoch(self.model, self.valid_dataloader, device)

            # Save model
            torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter_epoch{}.pt'.format(epoch))

        # Save model
        torch.save(self.model.state_dict(), self.cfg.save_dir + '/rl_agent.pt')
        torch.save(self.model.segmenter.state_dict(), self.cfg.save_dir + '/rl_agent_segmenter.pt')
        


if __name__ == "__main__":
    rl_cfg = RLCnnAgentConfig()
    train_rl_agent = TrainRlCnnAgent(rl_cfg)
    train_rl_agent.train_rl_agent() 

    