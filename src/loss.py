import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

def kl_divergence(p: Tensor, q: Tensor, reduction: str = "mean", eps=1e-8) -> Tensor:
    """Calculate KL divergence between two distributions

    Args:
        p (Tensor): target distribution
        q (Tensor): predicted distribution
        reduction (str, optional): Defaults to "mean".

    Returns:
        Tensor: KL divergence
    """
    assert p.shape == q.shape
    assert p.min() >= 0 and q.min() >= 0, f"p: {p.min()}, q: {q.min()}"
    p = p + eps
    q = q + eps
    kl = p * (p.log() - q.log())
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-8, temp: float = 1.0) -> torch.Tensor:
    """Calculate a numerically stable softmax of the input tensor

    Args:
        x (torch.Tensor): input tensor
        dim (int, optional): Dimension over which to perform softmax. Defaults to -1.
        eps (float, optional): Small number to prevent division by zero. Defaults to 1e-8.
        temp (float, optional): Temperature parameter to adjust softness. Defaults to 1.0.

    Returns:
        torch.Tensor: softmax tensor
    """
    # Subtract the maximum value along the specified dimension for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp((x - x_max) / temp)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + eps)


def convert_to_onehot(labels: Tensor, num_classes: int) -> Tensor:
    """Convert label to one-hot tensor

    Args:
        labels (Tensor): label tensor
        num_classes (int): number of classes

    Returns:
        Tensor: one-hot tensor
    """
    assert labels.min() >= 0 and labels.max() < num_classes
    return torch.eye(num_classes)[labels]

def cross_entropy_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-8, mask: Tensor = None
) -> Tensor:
    """Calculate cross entropy loss

    Args:
        pred (Tensor): predicted distribution
        target (Tensor): target distribution
        reduction (str, optional): Defaults to "mean".

    Returns:
        Tensor: cross entropy loss
    """
    if len(target.shape) == 1:
        target = convert_to_onehot(target, pred.shape[-1])
    if pred.min() < 0 or pred.max() > 1:
        pred = softmax(pred)
    assert pred.shape == target.shape
    assert pred.min() >= 0 and target.min() >= 0
    pred = pred + eps
    target = target + eps
    loss = -torch.sum(target * pred.log(), dim=-1)
    
    if mask is not None:
        loss = loss * mask
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "max":
        return loss.max()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
def soft_cross_entropy_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-8, temp: float = 1.0, mask=None
) -> Tensor:
    """Calculate soft cross entropy loss

    Args:
        pred (Tensor): predicted distribution
        target (Tensor): target distribution.
        reduction (str, optional): Defaults to "mean".
        temp (float, optional): temperature. Defaults to 1.0.

    Returns:
        Tensor: soft cross entropy loss
    """    
    assert pred.shape == target.shape
    if pred.min() < 0 or pred.max() > 1:
        print("pred", pred.min(), pred.max())
        import pdb; pdb.set_trace()
    assert pred.min() >= 0 and target.min() >= 0
    
    pred = softmax(pred, temp=temp) if pred[0].sum() != 1 else pred
    target = softmax(target, temp=temp) if target[0].sum() != 1 else target
    loss = -torch.sum(target * pred.log(), dim=-1)
    
    if mask is not None:
        loss = loss * mask
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def convert_target_to_hist(target: Tensor, num_classes: int) -> Tensor:
    """Convert target to histogram

    Args:
        target (Tensor): target tensor
        num_classes (int): number of classes

    Returns:
        Tensor: histogram tensor
    """
    hist = torch.zeros(num_classes, dtype=target.dtype, device=target.device)
    hist.scatter_add_(0, target, torch.ones_like(target))
    return hist

def hist_kl_divergence(
    pred: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-8, mask: Tensor = None, temp: float = 0.01
) -> Tensor:
    """Calculate histogram KL divergence
    Args:
        pred (Tensor): logit tensor
        target (Tensor): target codes
        reduction (str, optional): Defaults to "mean".
        eps (float, optional): Defaults to 1e-8.
        mask (Tensor, optional): Defaults to None.
        temp (float, optional): Defaults to 0.1.
    Note:
        pred -> (M, K)
        mask -> (B, N)
        target -> (M, )
    """
    mask_counts = torch.count_nonzero(mask, dim=1)  # (B, )
    pred_splits = torch.split(pred, mask_counts.tolist(), dim=0)
    target_splits = torch.split(target, mask_counts.tolist(), dim=0)
    
    loss = 0
    for pred_split, target_split in zip(pred_splits, target_splits):
        # pred_split -> (n, K), target_split -> (n, )
        pred_code = softmax(pred_split, dim=-1, eps=eps, temp=temp) # (n, K)
        pred_code = torch.sum(pred_code, dim=0)  # (K, )
        pred_code = pred_code / pred_code.sum()
        target_hist = convert_target_to_hist(target_split, pred_split.shape[-1])  # (K, )
        # print(f"hist: {hist}")
        target_hist = target_hist / target_hist.sum()  # (K, )
        
        kl_loss = kl_divergence(target_hist, pred_code, reduction=reduction, eps=eps)
        loss += kl_loss
    return loss.mean()

def gumbel_softmax_sample(logits, temperature):
    """Sample from the Gumbel-Softmax distribution"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = y_hard - y.detach() + y
    return y

def calculate_l1_distance(tar_dist, pred_dist, weights=None):
    if weights is not None:
        return torch.sum(weights * torch.abs(tar_dist - pred_dist))  # Class weighted L1 loss
    else:
        return torch.sum(torch.abs(tar_dist - pred_dist))

def l1_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-8, mask: Tensor = None
):
    # pred: (B, K), target: (M, )
    pred_soft = F.softmax(pred, dim=-1)  # (B, K)
    # target (M, ) => target_hist (B, K)
    # Target: (M, ), Mask: (B, N)
    mask_counts = torch.count_nonzero(mask, dim=1)  # (B, )
    target_splits = torch.split(target, mask_counts.tolist(), dim=0)
    target_list = []
    for target_sp in target_splits:
        target_hist = convert_target_to_hist(target_sp, pred.shape[-1])  # (K, )
        target_hist = target_hist / target_hist.sum()  # (K, )
        target_list.append(target_hist)
    target_hist = torch.stack(target_list)  # (B, K)
    l1_loss = calculate_l1_distance(target_hist, pred_soft)
    return l1_loss
