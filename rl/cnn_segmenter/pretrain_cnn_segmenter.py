import sys
sys.path.append('../')
sys.path.append('../../')
import os
from load_dataset import ExtractedFeaturesDataset
from cnn_model import CnnSegmenter, CnnBoundaryPredictor
from cnn_model import CnnBoundaryConfig
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import f1_score

import wandb
from s2p.scripts.phoneseg_utils import PrecisionRecallMetric
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def evaluate_phonemeseg(pred_file_path, gt_file_path, log_file):
    # predicted and ground truth boundaries files
    pred_file = open(pred_file_path, 'r')
    gt_file = open(gt_file_path, 'r')

    pred_lines = pred_file.readlines()
    gt_lines = gt_file.readlines()
    assert len(pred_lines) == len(gt_lines)

    metric_tracker_harsh = PrecisionRecallMetric(tolerance=1, mode="harsh")
    metric_tracker_lenient = PrecisionRecallMetric(tolerance=1, mode="lenient")

    for pred, gt in tqdm(zip(pred_lines, gt_lines), total=len(pred_lines)):
        pred = pred.strip().split()
        gt = gt.strip().split()
        assert len(pred) == len(gt)

        # location of non-boundary frames
        pred = [[i for i, frame in enumerate(pred) if frame != '0']]
        gt = [[i for i, frame in enumerate(gt) if frame != '0']]

        # Ground truth first, model prediction second
        metric_tracker_harsh.update(gt, pred)
        metric_tracker_lenient.update(gt, pred)

    tracker_metrics_harsh = metric_tracker_harsh.get_stats()
    tracker_metrics_lenient = metric_tracker_lenient.get_stats()

    print(f"{'SCORES:':<15} {'Lenient':>10} {'Harsh':>10}")
    log_file.write(f"{'SCORES:':<15} {'Lenient':>10} {'Harsh':>10}\n")
    for k in tracker_metrics_harsh.keys():
        print("{:<15} {:>10.4f} {:>10.4f}".format(k+":", tracker_metrics_lenient[k], tracker_metrics_harsh[k]))
        log_file.write("{:<15} {:>10.4f} {:>10.4f}\n".format(k+":", tracker_metrics_lenient[k], tracker_metrics_harsh[k]))

def evaluate_cnn_segmenter(cnn_segmenter, valid_dataloader, device, epoch, BATCH_SIZE, name='valid', wandb_log=True, print_result=False, log_file=None, SAVE_DIR=None, bds_postfix='bds'):
    # Evaluate
    cnn_segmenter.eval()
    valid_loss = 0
    f1_scores = 0
    accuracy = 0
    total_1s = 0
    total_1s_gt = 0
        
    if print_result:
        result_file = open(SAVE_DIR + f'/{name}_pred.{bds_postfix}', 'w')
        result_file_gt = open(SAVE_DIR + f'/{name}_target.{bds_postfix}', 'w')

    for step, batch in enumerate(valid_dataloader):
        logits = cnn_segmenter(batch['net_input']['features'].to(device))
        # Flatten logits
        logits = logits.reshape(-1, logits.size(-1))
        target = batch['net_input']['aux_target'].to(device).reshape(-1)
        # Cross entropy loss
        loss = F.cross_entropy(logits, target, ignore_index=-1)
        valid_loss += loss.item()

        mask = target != -1
        logits = logits[mask]
        target = target[mask]
        predictions = torch.argmax(logits, dim=-1)

        # Count number of 1s in predictions and target
        count_1s = (predictions == 1).sum().item()
        count_1s_gt = (target == 1).sum().item()
        total_1s += count_1s
        total_1s_gt += count_1s_gt

        if print_result:
            for i in range(len(target)):
                if i == len(target) - 1:
                    result_file_gt.write(str(target[i].item()) + '\n')
                else:
                    result_file_gt.write(str(target[i].item()) + ' ')
            for i in range(len(predictions)):
                if i == len(predictions) - 1:
                    result_file.write(str(predictions[i].item()) + '\n')
                else:
                    result_file.write(str(predictions[i].item()) + ' ')

        # Count f1 score
        f1_scores += f1_score(target.cpu(), predictions.cpu(), average='binary')

        # Count accuracy
        correct = (predictions == target).sum().item()
        accuracy += correct / (target != -1).sum().item()
    valid_loss /= len(valid_dataloader)
    print(f'Epoch {epoch}: {name} loss = {valid_loss}')
    f1_scores /= len(valid_dataloader)
    print(f'Epoch {epoch}: {name} f1 score = {f1_scores}')
    accuracy /= len(valid_dataloader)
    print(f'Epoch {epoch}: {name} accuracy = {accuracy}')
    print(f'Epoch {epoch}: {name} 1s = {total_1s}')
    print(f'Epoch {epoch}: {name} 1s_gt = {total_1s_gt}')
    print(f'Epoch {epoch}: {name} 1s_ratio = {total_1s / total_1s_gt}')

    if wandb_log:
        wandb.log({f'{name}_loss': valid_loss})
        wandb.log({f'{name}_f1_score': f1_scores})
        wandb.log({f'{name}_accuracy': accuracy})
        wandb.log({f'{name}_1s': total_1s})
        wandb.log({f'{name}_1s_gt': total_1s_gt})
        wandb.log({f'{name}_1s_ratio': total_1s / total_1s_gt})
    log_file.write(f'Epoch {epoch}: {name} loss = {valid_loss}\n')
    log_file.write(f'Epoch {epoch}: {name} f1 score = {f1_scores}\n')
    log_file.write(f'Epoch {epoch}: {name} accuracy = {accuracy}\n')
    log_file.write(f'Epoch {epoch}: {name} 1s = {total_1s}\n')
    log_file.write(f'Epoch {epoch}: {name} 1s_gt = {total_1s_gt}\n')
    log_file.write(f'Epoch {epoch}: {name} 1s_ratio = {total_1s / total_1s_gt}\n')
        


def pretrain_cnn_segmenter(audio_dir, save_tag, output_dir, clus_num=128):
    """
    Pretrain CNN-based segmenter
    Data: 
        - Extracted features
            (shape: (B, T, C))
        - Labels: word boundaries 
            (0: not boundary; 1: boundary)
            (shape: (B, T))
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Init model
    cnn_boundary_cfg = CnnBoundaryConfig()
    # cnn_boundary_cfg.input_dim = 512

    cnn_segmenter = CnnBoundaryPredictor(cnn_boundary_cfg).to(device)

    # Audio features path
    dir_path = f'{audio_dir}/precompute_pca512' # /precompute_pca512
    split = 'train'
    # Boundary labels path
    boundary_labels_path = f'{audio_dir}/CLUS{clus_num}'
    bds_postfix = "bds"

    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 1
    SAVE_EPOCHS = 1
    LOG_STEPS = 10
    MAX_STEPS_PER_EPOCH = 1000
    NAME=f"{save_tag}_pretrain_PCA_cnn_segmenter_kernel_size_{cnn_boundary_cfg.kernel_size}_v1_epo{NUM_EPOCHS}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_dropout{cnn_boundary_cfg.dropout}_optimAdamW_schCosineAnnealingLR"
    SAVE_DIR = f'{output_dir}/{NAME}'
    USE_CE_WEIGHTS = True
    if USE_CE_WEIGHTS:
        ce_weight = torch.tensor([1.0, 5.0]).to(device)
    else:
        ce_weight = torch.tensor([1.0, 1.0]).to(device)

    # Create save dir
    os.makedirs(SAVE_DIR, exist_ok=False)

    log_file = open(SAVE_DIR + '/log.txt', 'w')

    print('Loading dataset...')

    # Load Extracted features
    dataset = ExtractedFeaturesDataset(
        path=dir_path,
        split=split,
        aux_target_postfix=bds_postfix,
        aux_target_dir_path=boundary_labels_path,
    )

    valid_dataset = ExtractedFeaturesDataset(
        path=dir_path,
        split='valid',
        aux_target_postfix=bds_postfix,
        aux_target_dir_path=boundary_labels_path,
    )
    # Load data
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        collate_fn=dataset.collater,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=1,
        collate_fn=dataset.collater,
        shuffle=False,
        batch_size=BATCH_SIZE,
    )


    my_config = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'log_steps': LOG_STEPS,
        'save_dir': SAVE_DIR,
        'cnn_input_dim': cnn_boundary_cfg.input_dim,
        'cnn_hidden_dim': cnn_boundary_cfg.hidden_dim,
        'cnn_dropout': cnn_boundary_cfg.dropout,
        'cnn_kernel_size': cnn_boundary_cfg.kernel_size,
    }

    # init wandb
    wandb.init(
        project="uasr-rl",
        name=NAME,
        config=my_config,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        cnn_segmenter.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=0,
    )

    for epoch in range(NUM_EPOCHS):
        # Evaluate
        evaluate_cnn_segmenter(cnn_segmenter, valid_dataloader, device, epoch, BATCH_SIZE=BATCH_SIZE, log_file=log_file)

        # Train
        cnn_segmenter.train()

        for step, batch in enumerate(dataloader):
            
            if step == MAX_STEPS_PER_EPOCH:
                break

            logits = cnn_segmenter(batch['net_input']['features'].to(device))
            # Flatten logits
            logits = logits.reshape(-1, logits.size(-1))
            # print(logits.shape)
            target = batch['net_input']['aux_target'].to(device).reshape(-1)
            # print(target.shape)
            

            # Cross entropy loss
            loss = F.cross_entropy(logits, target, weight=ce_weight, ignore_index=-1)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log loss
            if step % LOG_STEPS == 0:
                wandb.log({'train_loss': loss.item()})
                print(f'Epoch {epoch}: step {step}: loss = {loss.item()}')
                log_file.write(f'Epoch {epoch}: step {step}: loss = {loss.item()}\n')

        if epoch % SAVE_EPOCHS == 0:
            torch.save(
                cnn_segmenter.state_dict(),
                os.path.join(SAVE_DIR, f'cnn_segmenter_{epoch}.pt'),
            )

    # Save model
    torch.save(
        cnn_segmenter.state_dict(),
        os.path.join(SAVE_DIR, f'cnn_segmenter.pt'),
    )
    # Evaluate
    evaluate_cnn_segmenter(cnn_segmenter, valid_dataloader, device, epoch, BATCH_SIZE=BATCH_SIZE, print_result=True, log_file=log_file, SAVE_DIR=SAVE_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str)
    parser.add_argument('--save_tag', type=str, default='ls')
    parser.add_argument('--clus_num', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()
    pretrain_cnn_segmenter(args.audio_dir, args.save_tag, args.output_dir, args.clus_num)