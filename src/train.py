
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

import random
from pathlib import Path

from dataset import build_loco_dataset, build_ad_dataset, build_visa_dataset, build_brats_dataset, build_liver_dataset, build_pathlogy_dataset
from train_engine import train_one_epoch, validate, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

import utils
from tokenizer import build_tokenizer
from model import get_model_with_args

import wandb
import logging as log
log.basicConfig(level=log.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='Train a MIM model')
    
    # Dataset setting
    parser.add_argument('--data_root', default='data/mvtec_loco', type=str, help='Dataset root')
    parser.add_argument('--category', default='bottle', type=str, help='Category')
    parser.add_argument('--is_mask', action='store_true', help='Use mask')
    
    # Training setting
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--device', default='cuda', type=str, help='Device')
    parser.add_argument('--seed', default=42, type=int, help='Seed')
    
    parser.add_argument('--masking', default="block_random", type=str, help='Masking type')  # random, block_random, object, division
    parser.add_argument('--num_objects', default=1, type=int, help='Number of objects to mask')
    parser.add_argument('--mask_dir', default='', type=str, help='Mask directory path')
    parser.add_argument('--div_type', default='random', type=str, help='Division type')  # block, slice
    parser.add_argument('--block_num', default=1, type=int, help='Number of blocks to mask')
    parser.add_argument('--random_slice_type', action='store_true', help='Random slice type')
    parser.add_argument('--slice_num', default=1, type=int, help='Number of slices to mask')
    parser.add_argument('--slice_type', default=None, type=str, nargs="+")  # horizontal, vertical , multi
    
    parser.add_argument('--input_type', default="img", type=str, help='Input type')  # img, feature, code
    parser.add_argument('--target_type', default="code", type=str, help='type of prediction target for MIM')  # code, hist
    parser.add_argument('--loss', default="ce", type=str, help='Loss type')  # ce, kl, ot, l1
    parser.add_argument('--inherit_codebook', action='store_true', help='Inherit codebook')
    
    # Tokenizer setting
    parser.add_argument('--tokenizer', default='hvq', type=str, help='Tokenizer type')
    parser.add_argument('--tokenizer_model_name', default='', type=str, help='Tokenizer model name')
    parser.add_argument('--tokenizer_weight', default='', type=str, help='Tokenizer weight path')
    parser.add_argument('--codebook_size', default=512, type=int, help='Codebook size')
    parser.add_argument('--codebook_dim', default=64, type=int, help='Codebook dimension')
    parser.add_argument('--tokenizer_input_size', default=224, type=int, help='Tokenizer input size')
    parser.add_argument('--num_codebooks', default=1, type=int, help='Number of codebooks for hierarchical VQ')
    parser.add_argument('--codebook_indices', default=None, type=int, nargs="+", help='Codebook indices')
    
    # Model setting
    parser.add_argument('--input_res', default=224, type=int, help='Input resolution for MIM ViT')
    parser.add_argument('--in_channel', default=3, type=int, help='Input channel')
    parser.add_argument('--window_size', default=24, type=int, help='Feature size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size')
    
    parser.add_argument('--model', default='mim', type=str, help='Model type')
    parser.add_argument('--rel_pos_bias', action='store_true', help='Use relative position bias')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', help='Disable relative position bias', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true', help='Use absolute position embedding')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, help='Initial value for layer scale')
    parser.add_argument('--embed_dim', default=384, type=int, help='Embedding dimension')
    parser.add_argument('--depth', default=12, type=int, help='Depth')
    parser.add_argument('--num_heads', default=6, type=int, help='Number of heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='MLP ratio')
    parser.add_argument('--qkv_bias', action='store_true', help='Use qkv bias')
    parser.add_argument('--qk_scale', default=None, type=float, help='QK scale')
    parser.add_argument('--drop_rate', default=0.0, type=float, help='Drop rate')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='Attention drop rate')
    
    parser.add_argument('--num_mask_patches', default=75, type=int, help='Number of masked patches')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--drop_path', default=0.1, type=float, help='Drop path rate')
    
    # CLS setting
    parser.add_argument('--early_layers', default=9, type=int, help='Number of early layers')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--shared_lm_head', action='store_true', help='Use shared lm head')
    
    # Optimzier setting
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer type')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='Optimizer epsilon')
    parser.add_argument('--opt_betas', default=(0.9, 0.999), type=float, nargs='+', help='Optimizer betas')
    parser.add_argument('--clip_grad', default=None, type=float, help='Gradient clipping')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=None)
    
    parser.add_argument('--base_lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Warmup epochs')  
    parser.add_argument('--warmup_steps', default=-1, type=int, help='Warmup steps')
    
    # Log setting
    parser.add_argument('--log_dir', default='logs', type=str, help='Log directory')
    parser.add_argument('--output_dir', default='output', type=str, help='Output directory')
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval')
    parser.add_argument('--save_interval', default=10, type=int, help='Save interval')
    parser.add_argument('--val_interval', default=10, type=int, help='Validation interval')
    parser.add_argument('--visualize', action='store_true', help='Visualize')
    parser.add_argument('--project_name', default="LADMIM", type=str, help='Project name')
    parser.add_argument('--exp_name', default="", type=str)
    
    # Resume setting
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, help='Start epoch')
    return parser.parse_args()

def main(args):
    
    config = vars(args)
    wandb.init(
        project=args.project_name, 
        name=f"{args.category}_{args.exp_name}"+datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"), 
        config=config
    )
    
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Tokenizer
    tokenizer = build_tokenizer(args)
    tokenizer.to(device)
    args.num_codebooks = len(args.codebook_indices)
    
    # Model
    codebook = None
    if args.inherit_codebook:
        codebook = tokenizer.codebook
    model = get_model_with_args(args, codebook)
    model.to(device)
    
    # Dataset
    if "ad" in args.data_root:
        dataset_train = build_ad_dataset(args, split='train')
        dataset_val = build_ad_dataset(args, split='val')
        dataset_test = build_ad_dataset(args, split='test')
    elif "loco" in args.data_root:
        dataset_train = build_loco_dataset(args, split='train')
        dataset_val = build_loco_dataset(args, split='val')
        dataset_test = build_loco_dataset(args, split='test')
    else:
        raise ValueError("Unknown dataset: {}".format(args.data_root))
    
    # Dataloader
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
        drop_last=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {args.model}\n{model}")
    log.info(f"Model parameters: {n_params}")
    log.info(f"Tokenizer: {args.tokenizer}\n{tokenizer}")
    log.info(f"LR: {args.base_lr}, Warmup LR: {args.warmup_lr}, Min LR: {args.min_lr}, Warmup epochs: {args.warmup_epochs}")
    log.info(f"Optimizer: {args.optimizer}, Weight decay: {args.weight_decay}, Momentum: {args.momentum}")
    log.info(f"Number of training epochs: {args.epochs}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, betas=args.opt_betas, eps=args.opt_eps, weight_decay=args.weight_decay
    )
    loss_scaler = NativeScaler()
    
    niter_per_epoch = len(dataset_train) // args.batch_size
    lr_schedule_values = utils.cosine_scheduler(
        args.base_lr, args.min_lr, args.epochs, niter_per_epoch, args.warmup_epochs, args.warmup_lr, args.warmup_steps
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, niter_per_epoch
    )
    
    log.info(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_stats = train_one_epoch(
            model, tokenizer, dataloader_train, optimizer, epoch, loss_scaler, args.clip_grad,
            lr_scheduler=None, start_steps=epoch*niter_per_epoch, lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values, args=args
        )
        
        if args.output_dir and epoch % args.save_interval == 0:
            
            validate(
                model, tokenizer, dataloader_val, global_step=epoch*niter_per_epoch, args=args
            )
            evaluate(
                model, tokenizer, dataset_test, global_step=epoch*niter_per_epoch, args=args
            )
            utils.save_model(
                model, optimizer, loss_scaler, epoch, args.output_dir
            )
        
    
    utils.save_model(
        model, optimizer, loss_scaler, epoch, args.output_dir
    )
            

if __name__ == '__main__':
    args = get_args()
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    main(args)
    
    
    
    
    
    
    
    
    
    