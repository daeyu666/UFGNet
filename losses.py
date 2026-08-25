# losses.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ufgnet import project_hsi_to_msi


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss for BxCxHxW tensors, returned in radians."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        dot = torch.sum(pred * target, dim=1)
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1) + self.eps)
        cos = dot / (pred_norm * target_norm + self.eps)
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(cos).mean()


def frequency_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    eta: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Eq. (26): amplitude-relative error + phase-consistency error.

    mode='spatial2d': FFT over H,W independently for each channel.
    mode='spectral1d': FFT over the hyperspectral channel axis at each pixel.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"frequency_distance shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if mode == "spatial2d":
        pred_f = torch.fft.fft2(pred, dim=(-2, -1))
        target_f = torch.fft.fft2(target, dim=(-2, -1))
    elif mode == "spectral1d":
        pred_f = torch.fft.fft(pred, dim=1)
        target_f = torch.fft.fft(target, dim=1)
    else:
        raise ValueError("mode must be 'spatial2d' or 'spectral1d'")

    pred_amp = torch.abs(pred_f)
    target_amp = torch.abs(target_f)
    pred_phase = torch.angle(pred_f)
    target_phase = torch.angle(target_f)

    amplitude_term = torch.abs(pred_amp - target_amp) / (target_amp + eps)
    phase_term = 1.0 - torch.cos(pred_phase - target_phase)
    return (amplitude_term + eta * phase_term).mean()


class UFGNetLoss(nn.Module):
    """Physics-driven UFGNet objective from Eqs. (22)-(26).

    L_total = lambda_rec * L_rec + lambda_sam * L_sam + lambda_freq * L_freq
    L_rec   = ||D_spa(Z)-X||_1 + ||Z P^T-Y||_1
    L_sam   = SAM(D_spa(Z), X)
    L_freq  = D_freq(Y_hat,Y) + gamma D_freq(X_hat,X)
    """

    def __init__(
        self,
        degradation_operator: nn.Module,
        srf: torch.Tensor,
        lambda_rec: float = 1.0,
        lambda_sam: float = 1e-2,
        lambda_freq: float = 1e-2,
        gamma: float = 0.5,
        eta: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.degradation_operator = degradation_operator
        self.register_buffer("srf", torch.as_tensor(srf, dtype=torch.float32).clone())
        self.lambda_rec = float(lambda_rec)
        self.lambda_sam = float(lambda_sam)
        self.lambda_freq = float(lambda_freq)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.eps = float(eps)
        self.sam = SAMLoss(eps=eps)

    def forward(
        self,
        pred_hr_hsi: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_hat = self.degradation_operator.degrade(pred_hr_hsi)
        y_hat = project_hsi_to_msi(pred_hr_hsi, self.srf)

        l_rec_hsi = F.l1_loss(x_hat, lr_hsi)
        l_rec_msi = F.l1_loss(y_hat, hr_msi)
        l_rec = l_rec_hsi + l_rec_msi

        l_sam = self.sam(x_hat, lr_hsi)

        l_freq_spatial = frequency_distance(
            y_hat, hr_msi, mode="spatial2d", eta=self.eta, eps=self.eps
        )
        l_freq_spectral = frequency_distance(
            x_hat, lr_hsi, mode="spectral1d", eta=self.eta, eps=self.eps
        )
        l_freq = l_freq_spatial + self.gamma * l_freq_spectral

        total = (
            self.lambda_rec * l_rec
            + self.lambda_sam * l_sam
            + self.lambda_freq * l_freq
        )
        parts = {
            "loss_total": total.detach(),
            "loss_rec": l_rec.detach(),
            "loss_rec_hsi": l_rec_hsi.detach(),
            "loss_rec_msi": l_rec_msi.detach(),
            "loss_sam": l_sam.detach(),
            "loss_freq": l_freq.detach(),
            "loss_freq_spatial": l_freq_spatial.detach(),
            "loss_freq_spectral": l_freq_spectral.detach(),
        }
        return total, parts
