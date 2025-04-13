
import math
import torch
import torch.nn as nn
from functools import partial
from timm.layers import trunc_normal_ as __call_tranc_normal

from transformer import PatchEmbed, Block, RelativePositionBias

def trunc_normal_(tensor, mean=0., std=1.):
    return __call_tranc_normal(tensor, mean=mean, std=std, a=-std, b=std)

class VisionTransformerMIM(nn.Module):
    def __init__(
        self, 
        input_type: str = "img",
        input_res: int = 24,
        patch_size: int = 1,
        in_chans: int = 384,
        codebook_size = 512,
        codebook_dim = 64,
        codebook = None,
        codebook_num = 3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        init_values=None,
        attn_head_dim=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        init_std=0.02,
        num_patches=None
    ):
        super().__init__()
        self.input_type = input_type
        self.feature_size = input_res
        self.patch_embed = PatchEmbed(
            input_type=input_type, input_res=input_res, patch_size=patch_size, \
                in_chans=in_chans, embed_dim=embed_dim, codebook_size=codebook_size, codebook_dim=codebook_dim, \
                    codebook=codebook, num_patches=num_patches
        )
        num_patches = self.patch_embed.num_patches
        self.num_heads = num_heads
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))  # patches + cls token + hist token
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if use_shared_rel_pos_bias:
            self.res_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim
            ) for i in range(depth)  
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.codebook_num = codebook_num
        # TODO: 
        # self.lm_head = nn.Linear(embed_dim, 10*10*3)
        self.lm_head = nn.Linear(embed_dim, codebook_size * codebook_num) 
        # self.lm_head = nn.Linear(embed_dim, in_chans)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.hist_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2 * layer_id))
        
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def forward_features(self, x, bool_masked_pos):
        """Forward features extraction without classification head.

        Args:
            x (Tensor): Input tensor. 
            bool_masked_pos (_type_): (B, N) tensor of bool values indicating whether the position is masked or not.
            
        Returns:
            Tensor: Features of the input tensor which contains class token.  (B, N + 1, C)
        """
        x = self.patch_embed(x)  # (B, N, C)
        B, N, C = x.size()
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        mask_token = self.mask_token.expand(B, N, -1)  # (B, N, C)
        hist_token = self.hist_token.expand(B, 1, -1)  # (B, 1, C)

        m = bool_masked_pos.unsqueeze(-1).type_as(x)  # (B, N, 1), float32
        x = x * (1 - m) + mask_token * m
        
        x = torch.cat((cls_tokens, hist_token, x), dim=1)  # (B, 2 + N, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        
        return self.norm(x)  # (B, 2 + N, C)
        
    def forward(self, x, bool_masked_pos=None, return_hist_token=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(x.size(0), self.patch_embed.num_patches, dtype=torch.bool, device=x.device)  # (B, N), non masked
        x = self.forward_features(x, bool_masked_pos) # (B, 2 + N, C)
        cls_token = x[:, 0]  # (B, C)
        hist_token = x[:, 1]  # (B, C)
        x = x[:, 2:]  # (B, N, C), remove cls token & hist token
        
        if return_patch_tokens:
            return x
        if return_hist_token:
            return self.lm_head(hist_token)  # (B, #V * K)
        else:
            # return only masked tokens
            return self.lm_head(x[bool_masked_pos])  # (B, M, #V * K)
    
    def foward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(x.size(0), self.patch_embed.num_patches, dtype=torch.bool, device=x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        B, N, C = x.size()
        
        hist_token = self.hist_token.expand(B, -1, -1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask_token = self.mask_token.expand(B, N, -1)
        
        mask = bool_masked_pos.unsqueeze(-1).type_as(x)
        x = x * (1 - mask) + mask_token * mask
        
        x = torch.cat((cls_tokens, hist_token, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 2:])  # remove cls token & hist token
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 2:]
        else:
            raise ValueError("layer_id must be int or list of int")
    
    def get_last_attention(self, x):
        B, C, H, W = x.size()
        
        x = self.patch_embed(x)
        B, N, C = x.size()
        
        hist_token = self.hist_token.expand(B, -1, -1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, hist_token, x), dim=1)
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self.interpolate_pos_encoding(x, H, W)
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)
            
class VisionTransformerMIMCLS(VisionTransformerMIM):
    def __init__(self, input_type="img", input_res=224, patch_size=16, in_chans=384, codebook_size=512, codebook_dim=32, codebook=None, embed_dim=768, depth=12, codebook_num=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None, use_abs_pos_emb=True,
                 use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, early_layers=6, head_layers=2, shared_lm_head=True
                ):
        super().__init__(
            input_type=input_type, input_res=input_res, patch_size=patch_size, in_chans=in_chans, codebook_size=codebook_size, embed_dim=embed_dim, 
            codebook_num=codebook_num, codebook_dim=codebook_dim, codebook=codebook,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, attn_head_dim=attn_head_dim, use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias, init_std=init_std
        )
        self.early_layers = early_layers
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, early_layers + head_layers))]
        self.cls_pt_layers = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim
            ) for i in range(early_layers, early_layers + head_layers)
        ])
        self.fix_init_cls_pt_weight()
        
        self.shared_lm_head = shared_lm_head
        if not shared_lm_head:
            self.cls_pt_norm = norm_layer(embed_dim)
            self.cls_pt_lm_head = nn.Linear(embed_dim, codebook_size * self.codebook_num)
            
            self.cls_pt_norm.apply(self._init_weights)
            self.cls_pt_lm_head.apply(self._init_weights)
        
    def fix_init_cls_pt_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2 * layer_id))
            
        for layer_id, layer in enumerate(self.cls_pt_layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
            
    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        B, N, C = x.size()
        
        hist_token = self.hist_token.expand(B, -1, -1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask_token = self.mask_token.expand(B, N, -1)
        
        mask = bool_masked_pos.unsqueeze(-1).type_as(x)
        x = x * (1 - mask) + mask_token * mask
        
        x = torch.cat((cls_tokens, hist_token, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i + 1 == self.early_layers:
                early_states = x[:, 2:]
        
        x_cls_pt = torch.cat([x[:, [0]], early_states], dim=1)  # cls token + early states
        for blk in self.cls_pt_layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)
        
        return self.norm(x), self.norm(x_cls_pt) if self.shared_lm_head else self.cls_pt_norm(x_cls_pt)
    
    def forward(self, x, bool_masked_pos=None, return_hist_token=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(x.size(0), self.patch_embed.num_patches, dtype=torch.bool, device=x.device)
        
        x, x_cls_pt = self.forward_features(x, bool_masked_pos)
        hist_token = x[:, 1]
        x = x[:, 2:]
        x_cls_pt = x_cls_pt[:, 1:]
        if return_patch_tokens:
            return [x, x_cls_pt]
        if return_hist_token:
            return [self.lm_head(hist_token) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt)]
        else:
            return [self.lm_head(x[bool_masked_pos]), self.lm_head(x_cls_pt[bool_masked_pos]) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt[bool_masked_pos])]

def get_model_with_default():
    return VisionTransformerMIM(
        input_res=224, patch_size=16, in_chans=3, codebook_size=20, embed_dim=768, depth=12, num_heads=12, codebook_num=1,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm, init_values=1e-4, attn_head_dim=None, use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False, init_std=0.02, early_layers=6, head_layers=2, shared_lm_head=True
    )

def get_model_with_args(args, codebook=None):
    num_patches = None
    if args.tokenizer == "hvq":
        num_patches = args.window_size ** 2
    return VisionTransformerMIM(
        input_type=args.input_type, input_res=args.window_size, patch_size=args.patch_size, in_chans=args.in_channel,
        codebook_size=args.codebook_size, codebook_dim=args.codebook_dim, codebook=codebook, 
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, codebook_num=args.num_codebooks, mlp_ratio=args.mlp_ratio, qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path, norm_layer=nn.LayerNorm, init_values=1e-4,
        attn_head_dim=None, use_abs_pos_emb=args.abs_pos_emb, use_rel_pos_bias=args.rel_pos_bias, use_shared_rel_pos_bias=False,
        init_std=0.02, num_patches=num_patches,
    )

        
if __name__ == "__main__":
    model = get_model_with_default()
    inputs = torch.randn(4, 384, 24, 24)
    mask = (torch.rand(1, 576) > 0.5)
    outputs = model(inputs, mask, return_all_tokens=True)
    print(outputs[0].shape)
    

