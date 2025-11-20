# evaluation.py
import math
import torch
import torch.nn.functional as F

def psnr(true: torch.Tensor, pred: torch.Tensor, max_val: float = 1.0) -> float:
    """
    true, pred : mêmes shapes (par ex. (T, N) ou (B, T, N))
    """
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val**2 / mse).item()


def _gaussian(window_size, sigma):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel=1):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(true: torch.Tensor, pred: torch.Tensor, window_size=11, val_range=1.0) -> float:
    """
    SSIM 2D (pour cartes (space,time)).
    true, pred : (H, W) ou (B,1,H,W)
    """
    if true.dim() == 2:
        true = true.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)

    channel = true.size(1)
    window = _create_window(window_size, channel).to(true.device)

    mu1 = F.conv2d(true, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(true * true, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(true * pred, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    R² global.
    """
    ssr = torch.sum((target - pred) ** 2)
    mean_target = torch.mean(target)
    sst = torch.sum((target - mean_target) ** 2) + eps
    r2 = 1 - ssr / sst
    return r2.item()
