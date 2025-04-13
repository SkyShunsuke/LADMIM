import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
import json
import yaml
import os
from tqdm import tqdm

import random
from pathlib import Path

from dataset import build_loco_dataset, LOCO_CLASSES
from train_engine import train_one_epoch, validate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

import utils
from tokenizer import build_tokenizer
from model import get_model_with_default

import logging as log
log.basicConfig(level=log.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='LADMIM evaluation')
    
    parser.add_argument('--data_root', default='data/mvtec_loco', type=str, help='path to dataset')
    parser.add_argument('--category', default='bottle', type=str, help='category to evaluate')
    parser.add_argument('--is_mask', action="store_true", help='whether to use ground truth mask')
    parser.add_argument('--input_res', default=224, type=int, help='input resolution')
    parser.add_argument('--window_size', default=24, type=int, help='the size of feature map')
    
    parser.add_argument('--tokenizer', default='hvq', type=str, help='tokenizer type')
    parser.add_argument('--model_config', default='', type=str, help='path to model config file')
    parser.add_argument('--model_ckpt', default='', type=str, help='path to model checkpoint')
    
    # Evaluation config
    parser.add_argument('--distance', default="acc", type=str, help='distance metric')
    
    parser.add_argument('--masking', default='block_random', type=str, help='type of masking strategy')  # [random, block_random, object]
    # Mask configuration for random masking
    parser.add_argument('--num_mask_patches', default=75, type=int, help='number of patches to mask')
    parser.add_argument('--max_mask_patches_per_block', default=1000, type=int, help='max number of patches to mask in a block')
    parser.add_argument('--min_mask_patches_per_block', default=16, type=int, help='size of mask patch')
    parser.add_argument('--num_iterations', default=1, type=int, help='number of iterations')
    
    # Mask configuration for object masking
    parser.add_argument('--num_objects', default=1, type=int, help='number of objects to mask')
    parser.add_argument('--mask_dir', default='', type=str, help='Mask directory path')
    
    # Visualization
    parser.add_argument('--output_dir', default='output', type=str, help='output directory')
    parser.add_argument('--vis', action="store_true", help='whether to visualize')

    # Other 
    parser.add_argument('--device', default='cuda', type=str, help='device')
    
    args = parser.parse_args()
    return args

def get_args_from_config(config_path: str) -> argparse.Namespace:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description='LADMIM')
    arg_list = []
    for key, value in config.items():
        if "wandb_version" in config.keys():
            value = value["value"] if isinstance(value, dict) else value
        if value is not None:    
            arg_list.append(f'--{key}')
            arg_list.append(str(value))
        parser.add_argument(f'--{key}', help=f'{key}')
    args = parser.parse_args(arg_list)
    return args

def calculate_cross_entropy_loss(shape, outputs, input_ids, num_codebooks, blk_id=0, mask=None, reduction="mean"):
    """Calculate cross entropy loss of the model

    Args:
        shape (_type_): (B, h, w)
        outputs (_type_): (B, N, K*Num_codebooks)
        input_ids (_type_): (B, Num_codebooks, H*W)
        blk_id (int, optional): _description_. Defaults to 0.
        mask (_type_, optional): _description_. Defaults to None.
        reduction (str, optional): _description_. Defaults to "mean".
    Returns:
        _type_: _description_
    """
    b, h, w = shape
    
    out_all, out_cls = outputs  # (B, N, K*Num_codebooks)
    out_all = out_all.reshape(b, h, w, num_codebooks, -1)  # (B, H, W, Num_codebooks, K)
    logits_blks = torch.split(out_all, 1, dim=3)  # [(B, H, W, K), ...]
    preds_blk = torch.argmax(logits_blks[blk_id], dim=-1).squeeze(-1)  # (B, H, W)
    target_blk = input_ids[:, blk_id].reshape(b, h, w)  # (B, H, W)
    
    if mask is not None:
        mask = mask.reshape(b, h, w)
        preds_blk = preds_blk[mask]
        preds_blk = preds_blk.reshape(b, -1)
        logits_blk = logits_blks[blk_id].reshape(b, h, w, -1)
        logits_blk = logits_blk[mask]
        logits_blk = logits_blk.reshape(b, -1)
        target_blk = target_blk[mask]
        target_blk = target_blk.reshape(b, -1)
        
    # cross entropy: (B, H, W)
    import pdb; pdb.set_trace()
    loss = torch.nn.functional.cross_entropy(logits_blks[blk_id], target_blk, reduction="none")  # (B, M)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "max":
        return loss.max()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    

def calculate_accuracy(shape, outputs, input_ids, num_codebooks, blk_id=0, mask=None):
    """Calculate accuracy of the model

    Args:
        shape (_type_): (B, h, w)
        outputs (_type_): (B, N, K*Num_codebooks)
        input_ids (_type_): (B, Num_codebooks, H*W)
        blk_id (int, optional): _description_. Defaults to 0.
        mask (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """
    b, h, w = shape
    
    out_all, out_cls = outputs  # (B, N, K*Num_codebooks)
    out_all = out_all.reshape(b, h, w, num_codebooks, -1)  # (B, H, W, Num_codebooks, K)
    logits_blks = torch.split(out_all, 1, dim=3)  # [(B, H, W, K), ...]
    preds_blk = torch.argmax(logits_blks[blk_id], dim=-1).squeeze(-1)  # (B, H, W)
    target_blk = input_ids[:, blk_id].reshape(b, h, w)  # (B, H, W)
    
    if mask is not None:
        mask = mask.reshape(b, h, w)
        preds_blk = preds_blk[mask]
        preds_blk = preds_blk.reshape(b, -1)
        target_blk = target_blk[mask]
        target_blk = target_blk.reshape(b, -1)
        
    pred_results = (preds_blk == target_blk).float()  # (B, H, W)
    pred_results_split = torch.split(pred_results, 1, dim=0)  # [(1, H, W), ...]
    accs = [pred.mean().item() for pred in pred_results_split]
    return accs

def evaluate(args, model_args):
    
    # Load models
    tokenizer = build_tokenizer(model_args)
    num_codebook_size = tokenizer.n_embed
    num_codebooks = 4
    tokenizer.to(args.device)
    tokenizer.eval()
    
    model = get_model_with_default()
    model.to(args.device)
    model.eval()
    model.load_state_dict(torch.load(args.model_ckpt)["model"])
    
    dataset_test = build_loco_dataset(args, split="test")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)
    
    scores = []
    gt_labels = []
    anom_types = []
    
    for i in range(args.num_iterations):
        scores_iter = np.zeros((args.num_iterations, len(dataset_test)))
        
        for j, inputs in tqdm(enumerate(dataloader_test), total=len(dataset_test)):
            images_processed, images, bool_masked_pos = inputs["samples"]
            clsnames = inputs["clsnames"]
            
            images = images.to(args.device)
            images_processed = images_processed.to(args.device)
            bool_masked_pos = bool_masked_pos.to(args.device)
            b, h, w = bool_masked_pos.shape
        
            with torch.no_grad():
                input_ids = tokenizer.get_codebook_indices(images, clsnames)  # (B, Num_codebooks, H*W)
                features = tokenizer.extract_feature(images)  # (B, C, H, W)
                
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)  # (B, H*W)
                ids_mask = bool_masked_pos.unsqueeze(1).expand(-1, num_codebooks, -1)
                labels = input_ids[ids_mask]
                labels = labels.reshape(b, num_codebooks, -1)
                labels = labels.long()
                
                outputs = model(features, bool_masked_pos=bool_masked_pos, return_all_tokens=True)
            
            sum_score = np.zeros(b)
            for k in range(num_codebooks):
                if args.distance == "acc":
                    sum_score += calculate_accuracy((b, h, w), outputs, input_ids, num_codebooks, blk_id=k, mask=bool_masked_pos)
                elif args.distance == "ce":
                    sum_score -= calculate_cross_entropy_loss((b, h, w), outputs, input_ids, num_codebooks, blk_id=k, mask=bool_masked_pos)
            scores_iter[i][j] += (sum_score / num_codebooks)[0]
            
            if i == 0:
                anom_types.extend(inputs["anom_type"])
                gt_labels.extend(inputs["labels"])
        
    scores = np.mean(scores_iter, axis=0)
    
    anom_scores = -1 * scores
    gt_labels = np.array(gt_labels)
    
    str_mask = np.array([True if anomtype in ["str", "good"] else False for anomtype in anom_types])
    log_mask = np.array([True if anomtype in ["log", "good"] else False for anomtype in anom_types])

    labels_log = gt_labels[log_mask]
    labels_str = gt_labels[str_mask]

    scores_log = anom_scores[log_mask]
    scores_str = anom_scores[str_mask]
    
    all_auc = roc_auc_score(gt_labels, anom_scores)
    log_auc = roc_auc_score(labels_log, scores_log)
    str_auc = roc_auc_score(labels_str, scores_str)
    
    log.info(f"Category: {args.category}")
    log.info(f"Overall AUC: {all_auc}")
    log.info(f"Log AUC: {log_auc}")
    log.info(f"Str AUC: {str_auc}")
    
    

if __name__ == "__main__":
    args = get_args()
    model_args = get_args_from_config(args.model_config)
    evaluate(args, model_args)