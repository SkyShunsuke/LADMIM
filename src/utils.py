
import os
from collections import deque, defaultdict
import time

import math
import numpy as np

import torch
from torch import inf

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()
        
    def __call__(self, loss, optimzier, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        # dynamically scale the loss value, to prevent underflow/overflow
        self._scaler.scale(loss).backward(create_graph=create_graph)  
        if update_grad:
            if clip_grad is not None:
                self._scaler.unscale_(optimzier)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimzier)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimzier)
            self._scaler.update()
        else:
            norm = None
            
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float=2.0, layer_names=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detech().abs().max().to(device) for p in parameters)
    else:
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print("Top norm values: ", value_top)
                print("Top norm name: ", [layer_names[i][7:] for i in name_top.tolist()])
            
    return total_norm

def cosine_scheduler(base_value, final_value, total_epochs, niter_per_ep, \
    warmup_epochs=0, start_warup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warup_value, base_value, warmup_iters)
    
    iters = np.arange(total_epochs * niter_per_ep - warmup_iters)
    cosine_schedule = np.array([
        final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i/len(iters))) for i in iters
    ])
    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    
    assert len(schedule) == total_epochs * niter_per_ep
    return schedule


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()
    
    @property
    def global_avg(self):
        return self.totla / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )
        
def save_model(model, optimizer, scaler, epoch, output_dir):
    if not output_dir:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch}.pt"))