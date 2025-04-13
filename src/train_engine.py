import math
import random
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

import numpy as np
import utils
from loss import cross_entropy_loss, hist_kl_divergence, hist_ot_loss, l1_loss, calculate_l1_distance, convert_target_to_hist
from dataset import AD_CLASS_TO_ID, LOCO_CLASS_TO_ID, VISA_CLASS_TO_ID
from mask_sampler import ObjectMaskingSampler, RandomMaskingSampler, BlockMaskingSampler, DivisionMaskingSampler, ManualMaskSampler, \
    KMeansMaskingSampler, GroundTruthMaskSampler

from einops import rearrange
import wandb
import logging as log
log.basicConfig(level=log.INFO)

import time

def get_cost_mat(tokenizer: torch.nn.Module):
    codebook = tokenizer.codebook
    codebook_normalized = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
    cost_mat = 1 - codebook_normalized @ codebook_normalized.t()
    return cost_mat

def train_one_epoch(
    model: torch.nn.Module, 
    tokenizer: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    loss_scaler,
    max_norm: float = 0,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    args=None,
):
    model.train()
    
    print_freq = 10
    cost_mat = None
    
    if args.target_type == "code":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "kl":
        loss_fn = hist_kl_divergence
    elif args.loss == "l1":
        loss_fn = l1_loss
    
    loss_meter_main = utils.SmoothedValue(window_size=5)
    loss_meter_aux = utils.SmoothedValue(window_size=5)
    acc_meter_main = utils.SmoothedValue(window_size=5)
    acc_meter_aux = utils.SmoothedValue(window_size=5)
    
    s_dataloader = time.time() 
    for step, inputs in enumerate(dataloader):
        e_dataloader = time.time()
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] if "lr_scale" in param_group else lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
                    
        images_processed, images, bool_masked_pos = inputs["samples"]
        clsnames = inputs["clsnames"]
        if "ad" in args.data_root:
            clsids = [AD_CLASS_TO_ID[clsname] for clsname in clsnames]
        elif "loco" in args.data_root:
            clsids = [LOCO_CLASS_TO_ID[clsname] for clsname in clsnames]
        else:
            raise ValueError("Unknown dataset.")
        
        images = images.to(args.device)
        images_processed = images_processed.to(args.device)
        bool_masked_pos = bool_masked_pos.to(args.device)  # (B, H, W) or (B, N, H, W)
        if len(bool_masked_pos.shape) == 4:
            b, n, h, w = bool_masked_pos.shape
            bool_masked_pos = bool_masked_pos.flatten(2).to(torch.bool)  # (B, N, H*W)
            bool_masked_pos = bool_masked_pos.view(b*n, h*w)  # (B*N, H*W)
            images_processed = torch.repeat_interleave(images_processed, n, dim=0)  # (B*N, C, H, W)
        else:
            b, h, w = bool_masked_pos.shape
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)  # (B, H*W)
        
        s_tokenizer = time.time()
        with torch.no_grad():
            input_ids = tokenizer.get_codebook_indices(images_processed, clsids)  # (B, Num_codebooks, H*W)
            input_ids = torch.split(input_ids, 1, dim=1)  # [(B, 1, HW), ...]
            input_ids = [ids.squeeze(1) for ids in input_ids]  # [(B, HW)]
            features = tokenizer.extract_feature(images_processed)  # (B, C, H, W)
            
            labels = [ids[bool_masked_pos] for ids in input_ids] # [(M), ...]
            labels = [label.long() for label in labels]  
        
        e_tokenizer = time.time()
        
        s_model = time.time()
        with torch.cuda.amp.autocast():
            if args.target_type == "hist":
                outputs = model(features, bool_masked_pos=bool_masked_pos, return_hist_token=True)  # (B, num_codebooks*K)
            elif args.target_type == "code":
                outputs = model(features, bool_masked_pos=bool_masked_pos)  # (B, M, num_codebooks*K)
            e_model = time.time()
            
            s_loss = time.time()
            if isinstance(outputs, list):
                if args.num_codebooks > 1:
                    outputs1 = outputs[0]  # (M, num_codebooks*K)bool
                    outputs1 = torch.split(outputs1, args.codebook_size, dim=-1)  # [(M, K), ...]
                    loss1 = 0
                    for i, logits in enumerate(outputs1):  # (M, K)
                        target = labels[i] # (M)
                        loss1 += loss_fn(logits, target=target)
                    loss_meter_main.update(loss1.item())
                    
                    outputs2 = outputs[1]  # (M, num_codebooks*K)
                    outputs2 = torch.split(outputs2, args.codebook_size, dim=-1)  # [(M, K), ...]
                    loss2 = 0
                    for i, logits in enumerate(outputs2):
                        target = labels[i]
                        loss2 += loss_fn(logits, target=target)
                    loss_meter_aux.update(loss2.item())
                    loss = loss1 + loss2
                else:
                    labels = labels[0]
                    if args.target_type == "code":
                        loss_1 = loss_fn(outputs[0], target=labels)
                        loss_2 = loss_fn(outputs[1], target=labels)
                    elif args.loss == "kl":
                        loss_1 = loss_fn(outputs[0], target=labels, mask=bool_masked_pos)
                        loss_2 = loss_fn(outputs[1], target=labels, mask=bool_masked_pos)
                    elif args.loss == "l1":
                        loss_1 = loss_fn(outputs[0], target=labels, mask=bool_masked_pos)
                        loss_2 = loss_fn(outputs[1], target=labels, mask=bool_masked_pos)
                    loss_meter_main.update(loss_1.item())
                    loss_meter_aux.update(loss_2.item())
                    loss = loss_1 + loss_2
            else:
                if args.num_codebooks > 1:
                    # outputs: (M, num_codebooks*K)
                    outputs = torch.split(outputs, args.codebook_size, dim=-1)  # [(M, K), ...]
                    loss = 0
                    for i, logits in enumerate(outputs):  # (M, K)
                        target = labels[i] # (M)
                        if args.target_type == "hist":
                            loss += loss_fn(logits, target=target, mask=bool_masked_pos)
                        elif args.target_type == "code":
                            loss += loss_fn(logits, target=target)
                else:
                    loss = loss_fn(outputs, target=labels[0], mask=bool_masked_pos)
                    loss_meter_main.update(loss.item())

        e_loss = time.time()
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            log.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order
        )  # optimzier step w/ grad scaling, clipping
        loss_scale_value = loss_scaler.state_dict()["scale"]
        
        if step % print_freq == 0:
            log.info(f"Epoch: {current_epoch}, Step: {step}, Loss: {loss_value}, Grad Norm: {grad_norm}, Loss Scale: {loss_scale_value}")
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        
        if lr_scheduler is not None:
            lr_scheduler.step(
            )
            
        if step % args.log_interval == 0:
            wandb.log({"train/loss": loss_meter_main.avg, "step": step})
            wandb.log({"train/lr": max_lr, "step": step})
            wandb.log({"train/min_lr": min_lr, "step": step})
            wandb.log({"train/weight_decay": weight_decay_value, "step": step})
            wandb.log({"train/grad_norm": grad_norm, "step": step})
            wandb.log({"train/loss_scale": loss_scale_value, "step": step})
        
    return {
        "loss": loss_meter_main.avg,
        "lr": max_lr,
        "min_lr": min_lr,
        "weight_decay": weight_decay_value,
        "grad_norm": grad_norm,
        "loss_scale": loss_scale_value,
    }

def patchify(imgs, p):
    b, c, h, w = imgs.shape
    patched_imgs = rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
    return patched_imgs

def unpatchify(patched_imgs, p):
    b, n, d = imgs.shape
    imgs = rearrange(patched_imgs, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=p, p2=p)
    return imgs
                
def evaluate(
    model: torch.nn.Module, tokenizer: torch.nn.Module, test_dataset, global_step: int, 
    args
):
    mask_config = {
        "sampler": "block_random",
        "input_size": 24,
        "num_masking_patches": 230,
        "min_num_patches": 92,
        "max_num_patches": None,
        "min_aspect": 0.3, 
        "max_aspect": None
    }
    sampler = get_masking_sampler(mask_config)
    
    # test 
    assert args.tokenizer_score_path is not None, "Please provide the tokenizer score path"
    score_json = args.tokenizer_score_path
    hvq_results = []
    n_masks = args.num_masks
    print(f"We randomly sample {n_masks} masks for each image")
    
    # Open the file and read line by line
    with open(score_json, "r") as file:
        for line in file:
            # Convert each line from JSON format to a dictionary
            data = json.loads(line.strip())
            hvq_results.append(data) 
    
    for ds in test_dataset.datasets:
        category = ds.category
        test_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        test_batches = list(iter(test_loader))

        log_test_samples = [batches for batches in test_batches if "logical" in batches["filenames"][0]]
        str_test_samples = [batches for batches in test_batches if "structural" in batches["filenames"][0]]
        normal_test_samples = [batches for batches in test_batches if "good" in batches["filenames"][0]]
        
        all_test_samples = normal_test_samples + log_test_samples + str_test_samples
    
        all_hvq_scores = []
        score_key = "max_score"
        base_dir = args.data_root

        for sample in all_test_samples:
            name = sample["filenames"][0].replace(base_dir, "")
            for hvq_res in hvq_results:
                if name == hvq_res["filename"]:
                    break
            score = hvq_res[score_key]
            all_hvq_scores.append(score)
        
        n_log_scores = np.zeros((len(normal_test_samples + log_test_samples), n_masks))
        str_scores = np.zeros((len(str_test_samples), n_masks))
        n_log_samples = normal_test_samples + log_test_samples
        str_samples = str_test_samples
        
        for i, sample in enumerate(n_log_samples):
            imgs, org_img, clsnames = get_inputs(sample, args)
            img_path = sample["filenames"][0]
            for j in range(n_masks):
                mask = sampler(None, None)
                mask = torch.from_numpy(mask).unsqueeze(0)
                mask = mask_process(mask, args)
                d = 0
                out_pred, out_tar = None, None
                for idx in args.codebook_indices:
                    pred_hist, out_pred = get_pred_dist(imgs, mask, model, tokenizer, idx, out_pred, args)
                    target_hist, out_tar = get_target_dist(imgs, clsnames, mask, tokenizer, idx, out_tar, args, pred_hist)
                    d += calculate_l1_distance(pred_hist, target_hist)
                n_log_scores[i, j] = d
        
        for i, sample in enumerate(str_samples):
            imgs, org_img, clsnames = get_inputs(sample, args)
            img_path = sample["filenames"][0]
            for j in range(n_masks):
                mask = sampler(None, None)
                mask = torch.from_numpy(mask).unsqueeze(0)
                mask = mask_process(mask, args)
                d = 0
                out_pred, out_tar = None, None
                for idx in args.codebook_indices:
                    pred_hist, out_pred = get_pred_dist(imgs, mask, model, tokenizer, idx, out_pred, args)
                    target_hist, out_tar = get_target_dist(imgs, clsnames, mask, tokenizer, idx, out_tar, args, pred_hist)
                    d += calculate_l1_distance(pred_hist, target_hist)
                str_scores[i, j] = d
        
        normal_scores, log_scores = n_log_scores[:len(normal_test_samples)], n_log_scores[len(normal_test_samples):]
        normal_avg_scores = np.mean(normal_scores, axis=-1)
        log_avg_scores = np.mean(log_scores, axis=-1)
        str_avg_scores = np.mean(str_scores, axis=-1)
        all_mlm_scores = np.concatenate([normal_avg_scores, log_avg_scores, str_avg_scores], axis=0)

        scaled_hvq_scores = (np.array(all_hvq_scores) - np.mean(all_hvq_scores)) / np.std(all_hvq_scores)
        scaled_mlm_scores = (np.array(all_mlm_scores) - np.mean(all_mlm_scores)) / np.std(all_mlm_scores)
        
        all_merged_scores = scaled_hvq_scores + scaled_mlm_scores
        
        normal_scores = np.array(all_merged_scores[:len(normal_test_samples)])
        
        log_scores = np.array(all_merged_scores[len(normal_test_samples):len(normal_test_samples)+len(log_test_samples)])
        str_scores = np.array(all_merged_scores[len(normal_test_samples)+len(log_test_samples):])

        n_log_scores = np.concatenate([normal_scores, log_scores])
        n_str_scores = np.concatenate([normal_scores, str_scores])
        n_log_labels = [0] * len(normal_scores) + [1] * len(log_scores)
        n_str_labels = [0] * len(normal_scores) + [1] * len(str_scores)

        from sklearn.metrics import roc_auc_score
        auc_log = roc_auc_score(n_log_labels, n_log_scores)
        auc_str = roc_auc_score(n_str_labels, n_str_scores)
        
        log.info(f"Category: {category}, AUC-Logical: {auc_log}, AUC-Structural: {auc_str}")

def get_inputs(sample, args=None):
    clsnames = sample["clsnames"]
    if "ad" in args.data_root:
        clsids = [AD_CLASS_TO_ID[clsname] for clsname in clsnames]
    elif "loco" in args.data_root:
        clsids = [LOCO_CLASS_TO_ID[clsname] for clsname in clsnames]
    else:
        raise ValueError("Invalid dataset root")
    imgs = sample["samples"][1]
    imgs = imgs.to(args.device)
    org_img = imgs.permute(0, 2, 3, 1).cpu().numpy()[0]
    return imgs, org_img, clsids

def mask_process(mask, args=None):
    mask = mask.flatten().unsqueeze(0).to(args.device).bool()
    return mask

def get_target_dist(imgs, clsnames, mask, tokenizer, codebook_idx=None, out=None, args=None, pred_hist=None):
    
    with torch.no_grad():
        if out is None:
            out_tar = tokenizer.get_codebook_indices(imgs, clsnames)  # (B, V, N)/(B, N)
        else:
            out_tar = out
        if codebook_idx is not None:
            input_ids = out_tar[:, codebook_idx]  # (B, N)
        else:
            input_ids = out_tar
        assert mask.shape == input_ids.shape
        input_ids = input_ids[mask].long()
        
        with torch.amp.autocast('cuda'):
            target_hist = convert_target_to_hist(input_ids, pred_hist.shape[-1]).cpu()
            target_hist = target_hist.to(torch.float64)
            target_hist = target_hist / target_hist.sum()
    return target_hist, out_tar

def get_pred_dist(imgs, mask, model, tokenizer, codebook_idx=None, out=None, args=None):
    if out is not None:
        out_ = torch.split(out, args.codebook_size, dim=-1)[codebook_idx]
        out_ = out_.to(torch.float64)
        pred_hist = F.softmax(out_, dim=-1)
        pred_hist = pred_hist.cpu()[0]
        return pred_hist, out
        
    if args.tokenizer == "hvq":
        feature = tokenizer.extract_feature(imgs)
    else:
        feature = imgs

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out_pred = model(feature, bool_masked_pos=mask, return_hist_token=True)
            out_pred = out_pred.to(torch.float64)
            if codebook_idx is not None:
                out = torch.split(out_pred, args.codebook_size, dim=-1)[codebook_idx]
                pred_hist = F.softmax(out, dim=-1)
                pred_hist = pred_hist.cpu()[0]
            else:
                pred_hist = F.softmax(out, dim=-1)
                pred_hist = pred_hist.cpu()[0]
    return pred_hist, out_pred
    

def get_masking_sampler(mask_config):
    sampler_type = mask_config["sampler"]
    cfg = mask_config.copy()
    cfg.pop("sampler")
    if sampler_type == "block_random":
        return BlockMaskingSampler(**cfg)
    elif sampler_type == "checkerboard":
        from mask_sampler import CheckerboardMaskingSampler
        return CheckerboardMaskingSampler(**cfg)
    elif sampler_type == "random":
        return RandomMaskingSampler(**cfg)
    else: 
        raise ValueError(f"Unknown sampler type: {sampler_type}")
            
    