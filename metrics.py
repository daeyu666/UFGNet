# metrics.py
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def calc_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.clamp(pred.detach().float(), 0.0, 1.0)
    target = torch.clamp(target.detach().float(), 0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    return math.sqrt(max(mse, 1e-12))


def calc_psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    rmse = calc_rmse(pred, target)
    if rmse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(max_value / rmse)


def calc_sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean spectral angle in degrees."""
    pred = pred.detach().float()
    target = target.detach().float()
    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + eps)
    target_norm = torch.sqrt(torch.sum(target * target, dim=1) + eps)
    cos = dot / (pred_norm * target_norm + eps)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos) * 180.0 / math.pi
    return torch.mean(angle).item()


def calc_cc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.detach().float()
    target = target.detach().float()
    b, c, _, _ = pred.shape
    pred = pred.view(b, c, -1)
    target = target.view(b, c, -1)
    pred_centered = pred - pred.mean(dim=2, keepdim=True)
    target_centered = target - target.mean(dim=2, keepdim=True)
    numerator = torch.sum(pred_centered * target_centered, dim=2)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2, dim=2)
        * torch.sum(target_centered ** 2, dim=2)
        + eps
    )
    return torch.mean(numerator / (denominator + eps)).item()


def calc_ergas(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_ratio: int,
    eps: float = 1e-8,
) -> float:
    pred = pred.detach().float()
    target = target.detach().float()
    rmse_per_band = torch.sqrt(
        torch.mean((pred - target) ** 2, dim=(0, 2, 3)) + eps
    )
    mean_target = torch.mean(target, dim=(0, 2, 3))
    ergas = 100.0 / scale_ratio * torch.sqrt(
        torch.mean((rmse_per_band / (mean_target + eps)) ** 2)
    )
    return ergas.item()


def _gaussian_window(window_size: int, sigma: float, dtype, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def calc_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> float:
    """Standard local-window SSIM averaged over batch, bands and pixels."""
    pred = torch.clamp(pred.detach().float(), 0.0, data_range)
    target = torch.clamp(target.detach().float(), 0.0, data_range)
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("SSIM expects matching BxCxHxW tensors")

    _, channels, h, w = pred.shape
    max_window = min(window_size, h, w)
    if max_window % 2 == 0:
        max_window -= 1
    max_window = max(max_window, 1)
    if max_window == 1:
        # Degenerate spatial case; retain the SSIM algebra without filtering.
        mu_x, mu_y = pred, target
        sigma_x = torch.zeros_like(pred)
        sigma_y = torch.zeros_like(target)
        sigma_xy = torch.zeros_like(pred)
    else:
        window2d = _gaussian_window(max_window, sigma, pred.dtype, pred.device)
        kernel = window2d.view(1, 1, max_window, max_window).repeat(channels, 1, 1, 1)
        pad = max_window // 2
        pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
        target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")
        mu_x = F.conv2d(pred_pad, kernel, groups=channels)
        mu_y = F.conv2d(target_pad, kernel, groups=channels)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(pred_pad * pred_pad, kernel, groups=channels) - mu_x2
        sigma_y = F.conv2d(target_pad * target_pad, kernel, groups=channels) - mu_y2
        sigma_xy = F.conv2d(pred_pad * target_pad, kernel, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (
        (mu_x * mu_x + mu_y * mu_y + c1)
        * (sigma_x + sigma_y + c2)
    )
    return (numerator / (denominator + eps)).mean().item()


# Backward-compatible name used by any older scripts.
def calc_ssim_simple(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    del eps
    return calc_ssim(pred, target)


def calc_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_ratio: int,
) -> Dict[str, float]:
    return {
        "PSNR": calc_psnr(pred, target),
        "SSIM": calc_ssim(pred, target),
        "ERGAS": calc_ergas(pred, target, scale_ratio),
        "SAM": calc_sam(pred, target),
        "CC": calc_cc(pred, target),
        "RMSE": calc_rmse(pred, target),
    }


class MetricAverager:
    def __init__(self):
        self.data = {}

    def update(self, metric_dict: Dict[str, float]):
        for key, value in metric_dict.items():
            self.data.setdefault(key, []).append(float(value))

    def average(self) -> Dict[str, float]:
        return {key: float(np.mean(values)) for key, values in self.data.items()}

    def reset(self):
        self.data = {}
