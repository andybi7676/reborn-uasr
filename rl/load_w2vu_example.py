import fairseq
import torch
from fairseq import checkpoint_utils, tasks, utils
from omegaconf import OmegaConf
import argparse

def register_and_setup_task(task_cfg_fpath, env):
    task_cfg = OmegaConf.load(task_cfg_fpath)
    task_cfg.fairseq.common.user_dir = f"{env.WORK_DIR}/s2p"
    task_cfg.fairseq.task.text_data = f"{env.WORK_DIR}/rl/dummy_data"
    utils.import_user_module(task_cfg.fairseq.common)
    task = tasks.setup_task(task_cfg.fairseq.task)
    return task, task_cfg

def get_model_and_saved_cfg(env, task, use_cuda=False):
    model_fpath = f"{env.WORK_DIR}/s2p/multirun/ls_100h/large_clean/ls_wo_lv_g2p_all/cp4_gp1.5_sw0.5/seed3/checkpoint_best.pt"
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        [model_fpath],
        task=task
    )
    model = models[0]
    model.eval()
    if use_cuda:
        model.cuda()
    return model, saved_cfg

def generate_random_sample(bsz, tsz, device='cpu'):
    # bsz: batch_size; tsz: time_size
    random_sample = torch.randn((bsz, tsz, 512), device=device) # (B, T, C), C=512 (repr dim from "w2v2->pca")
    random_sample_size = torch.randint(5, tsz, (bsz,), device=device)
    random_sample_mask = torch.arange(tsz, device=device).expand(bsz, tsz) >= random_sample_size.unsqueeze(1)
    return random_sample, random_sample_mask

def main(args):
    env = OmegaConf.load(args.env)
    task_cfg_fpath = f"{env.WORK_DIR}/rl/config/dummy.yaml"
    task, task_cfg = register_and_setup_task(task_cfg_fpath, env)
    # print(task_cfg)
    
    use_cuda = torch.cuda.is_available()
    model, saved_cfg = get_model_and_saved_cfg(env, task, use_cuda=use_cuda)
    # print(model)
    device = next(model.parameters()).device
    random_sample, random_sample_mask = generate_random_sample(bsz=5, tsz=100, device=device)
    input = {
        "features": random_sample,
        "padding_mask": random_sample_mask,
        "dense_x_only": True # set this to True to get generator outputs only
    }
    model_out = model(**input)
    emissions = model.get_logits(model_out)
    print(emissions.shape) # (T, B, Dict_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="../env.yaml",
        help="custom local env file for github collaboration",
    )
    args = parser.parse_args()
    main(args)