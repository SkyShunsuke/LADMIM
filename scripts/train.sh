export CUDA_VISIBLE_DEVICES=0
python3 ./src/train.py\
    --data_root ./data/mvtec_loco \
    --category bottle \
    --batch_size 32 \
    --window_size 24 \
    --epochs 1000 \
    --num_workers 4 \
    --device cuda \
    --seed 4 \
    --masking block_random \
    --block_num 4 \
    --input_type feature \
    --target_type hist \
    --loss l1 \
    --inherit_codebook \
    --tokenizer hvq \
    --tokenizer_model_name hvq \
    --tokenizer_weight ./weights/hvq/loco_checkpoint.pth.tar \
    --codebook_size 512 \
    --codebook_dim 64 \
    --tokenizer_input_size 224 \
    --codebook_indices 0 1 2 3\
    --model mim \
    --abs_pos_emb \
    --layer_scale_init_value 0.1 \
    --num_mask_patches 130 \
    --min_mask_patches_per_block 46 \
    --input_res 224 \
    --in_channel 384 \
    --embed_dim 512 \
    --depth 4 \
    --num_heads 8 \
    --mlp_ratio 4 \
    --qkv_bias \
    --drop_rate 0.0 \
    --attn_drop_rate 0.1 \
    --drop_path 0.0 \
    --early_layers 9 \
    --head_layers 2 \
    --shared_lm_head \
    --optimizer adamw \
    --opt_eps 1e-8 \
    --clip_grad 1.0 \
    --weight_decay 1e-6 \
    --weight_decay_end 0.0 \
    --base_lr 1e-4 \
    --warmup_lr 1e-6 \
    --min_lr 1e-6 \
    --warmup_epochs 40 \
    --log_dir ./logs \
    --output_dir ./results/bottle \
    --log_interval 10 \
    --save_interval 1000 