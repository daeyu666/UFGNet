"""UFGNet reproduction from the paper equations.

Implements:
- QIEM: nonlinear MSI mapping + truncated-SVD subspace closed-form estimate + learnable gate.
- FGM: 1-D spectral FFT amplitude gating and 2-D spatial phase reconstruction.
- CCRM: observation residual back-projection, FASA/SpeDOB, and modulated deformable SpaDOB.

The spatial degradation operator is injected at forward time so the network can reuse
exactly the same Gaussian+Bicubic operator that generated the LR-HSI observations.

Source-faithfulness note:
Fig. 4 schematically draws a phase-domain cosine/sine convolution block, while
Algorithm 1 and Eq. (13) explicitly define phase-only inverse FFT followed by a
3x3 C_phase mapping. The executable baseline follows the explicit algorithm/equation
rather than silently inventing an undocumented fusion of the two descriptions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as tv_deform_conv2d
except Exception:  # torchvision may be unavailable or built without custom ops
    tv_deform_conv2d = None


def project_hsi_to_msi(hsi: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
    """Apply spectral response P in R^(Cm x Ch) to BxChxHxW HSI."""
    if hsi.ndim != 4:
        raise ValueError(f"Expected BxCxHxW HSI, got {tuple(hsi.shape)}")
    if srf.ndim != 2 or srf.shape[1] != hsi.shape[1]:
        raise ValueError(
            f"SRF must have shape (Cm, {hsi.shape[1]}), got {tuple(srf.shape)}"
        )
    return torch.einsum("mc,bchw->bmhw", srf.to(hsi), hsi)


class QuickInitialEstimationModule(nn.Module):
    """QIEM, corresponding to Eqs. (2)-(6)."""

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        rank: int = 5,
        regularization: float = 1e-4,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if regularization <= 0:
            raise ValueError("regularization must be > 0")
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.rank = int(rank)
        self.regularization = float(regularization)

        # C_up in Eq. (2): Conv 3x3 -> BN -> ReLU.
        self.spatial_mapping = nn.Sequential(
            nn.Conv2d(msi_channels, hsi_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hsi_channels),
            nn.ReLU(inplace=True),
        )
        # E([z1,z2]) in Eq. (6): 1x1 convolution followed by sigmoid in forward.
        self.fusion_estimator = nn.Conv2d(
            2 * hsi_channels, hsi_channels, kernel_size=1
        )

    def _subspace_estimate(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, srf: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = lr_hsi.shape
        _, m, H, W = hr_msi.shape
        if c != self.hsi_channels or m != self.msi_channels:
            raise ValueError("Input channel count does not match QIEM construction")
        if self.rank > min(c, h * w):
            raise ValueError(
                f"rank={self.rank} exceeds min(Ch, hw)={min(c, h*w)}"
            )

        # Mode-3 unfolding X_(3) in R^(Ch x hw).
        x3 = lr_hsi.reshape(b, c, h * w)
        # U_x: leading rank-r left singular vectors.
        U, _, _ = torch.linalg.svd(x3, full_matrices=False)
        Ux = U[:, :, : self.rank]  # B x Ch x r

        P = srf.to(dtype=lr_hsi.dtype, device=lr_hsi.device)  # Cm x Ch
        PU = torch.einsum("mc,bcr->bmr", P, Ux)  # B x Cm x r
        y3 = hr_msi.reshape(b, m, H * W)

        # H* = (Ux^T P^T P Ux + lambda I)^-1 Ux^T P^T Y_(3).
        lhs = torch.matmul(PU.transpose(1, 2), PU)
        eye = torch.eye(
            self.rank, dtype=lhs.dtype, device=lhs.device
        ).unsqueeze(0)
        lhs = lhs + self.regularization * eye
        rhs = torch.matmul(PU.transpose(1, 2), y3)
        coeff = torch.linalg.solve(lhs, rhs)  # B x r x HW

        z2 = torch.matmul(Ux, coeff).reshape(b, c, H, W)
        return z2

    def forward(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, srf: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z1_spatial = self.spatial_mapping(hr_msi)
        z2_subspace = self._subspace_estimate(lr_hsi, hr_msi, srf)
        beta = torch.sigmoid(
            self.fusion_estimator(torch.cat([z1_spatial, z2_subspace], dim=1))
        )
        initial = beta * z1_spatial + (1.0 - beta) * z2_subspace
        return initial, {
            "qiem_spatial": z1_spatial,
            "qiem_subspace": z2_subspace,
            "qiem_beta": beta,
        }


class FrequencyGuidedModule(nn.Module):
    """FGM, corresponding to Algorithm 1 and Eqs. (7)-(14)."""

    def __init__(self, hsi_channels: int, spectral_kernel_size: int = 3) -> None:
        super().__init__()
        if spectral_kernel_size < 1 or spectral_kernel_size % 2 == 0:
            raise ValueError("spectral_kernel_size must be a positive odd integer")
        self.hsi_channels = int(hsi_channels)

        # W_gate in Eq. (9), applied along the spectral-frequency axis.
        self.amplitude_gate = nn.Conv1d(
            1,
            1,
            kernel_size=spectral_kernel_size,
            padding=spectral_kernel_size // 2,
            bias=False,
        )
        # Paper states b in R^Ch, so keep an explicit per-frequency learnable bias.
        self.amplitude_bias = nn.Parameter(torch.zeros(1, 1, hsi_channels))

        # Algorithm 1 / Eq. (13): phase-only inverse FFT then a 3x3 C_phase.
        # It is applied band-by-band, so the same 1->1 mapping is shared over bands.
        self.phase_mapping = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def _spectral_branch(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        spectrum = torch.fft.fft(z, dim=1)
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        # Each spatial pixel is an independent Ch-long spectral-frequency signal.
        amp_seq = amplitude.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        gate_logits = self.amplitude_gate(amp_seq) + self.amplitude_bias
        gate = torch.sigmoid(gate_logits)
        refined_amp = amp_seq * gate
        refined_amp = refined_amp.reshape(b, h, w, c).permute(0, 3, 1, 2)

        refined_spectrum = torch.polar(refined_amp, phase)
        return torch.fft.ifft(refined_spectrum, dim=1).real

    def _spatial_branch(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        spectrum = torch.fft.fft2(z, dim=(-2, -1))
        phase = torch.angle(spectrum)
        unit_amplitude = torch.ones_like(phase)
        phase_only = torch.fft.ifft2(
            torch.polar(unit_amplitude, phase), dim=(-2, -1)
        ).real

        mapped = self.phase_mapping(phase_only.reshape(b * c, 1, h, w))
        return mapped.reshape(b, c, h, w)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._spectral_branch(z), self._spatial_branch(z)


class FrequencyAwareSpectralAttention(nn.Module):
    """FASA used by SpeDOB, corresponding exactly to Eqs. (17)-(19)."""

    def __init__(self, channels: int, tau: float = 1.0) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.channels = int(channels)
        self.tau = float(tau)
        self.scale_recovery = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self, spectral_residual: torch.Tensor, spectral_guidance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = spectral_residual.shape
        q = spectral_residual.reshape(b, c, h * w)

        # Data-driven term QQ^T in Eq. (18).
        correlation = torch.matmul(q, q.transpose(1, 2))

        # Eq. (17): P_ij = exp(-||f_i-f_j||_2^2 / tau), where f is obtained
        # by global spatial pooling of F_spe.  For scalar per-band pooled values,
        # ||f_i-f_j||_2^2 reduces to the squared scalar difference.
        f = spectral_guidance.mean(dim=(-2, -1))
        dist2 = (f.unsqueeze(2) - f.unsqueeze(1)).pow(2)
        physical_prior = torch.exp(-dist2 / self.tau)

        # Eq. (18) places the product (Q_i Q_j^T) * P_ij INSIDE the exponent.
        # Therefore the correct numerically-stable implementation is simply a
        # softmax over correlation * physical_prior, not softmax(log-correlation
        # plus log-prior) and not exp(correlation) * P_ij.
        fused_similarity = correlation * physical_prior
        affinity = torch.softmax(fused_similarity, dim=-1)

        # Eq. (19): Conv_1x1(R_spe odot Sigmoid((M R_spe^T)^T)).
        aggregated = torch.matmul(affinity, q).reshape(b, c, h, w)
        gated = spectral_residual * torch.sigmoid(aggregated)
        z2 = self.scale_recovery(gated)
        return z2, affinity


def _fallback_modulated_deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    padding: int,
) -> torch.Tensor:
    """Pure PyTorch Eq. (21) fallback using bilinear grid_sample."""
    b, cin, h, w = x.shape
    cout, win, kh, kw = weight.shape
    if cin != win or kh != kw:
        raise ValueError("Fallback deformable convolution expects square kernel")
    k_total = kh * kw
    if offset.shape[1] != 2 * k_total or mask.shape[1] != k_total:
        raise ValueError("Offset/mask channel count does not match kernel")

    dtype, device = x.dtype, x.device
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    yy = yy.unsqueeze(0).expand(b, -1, -1)
    xx = xx.unsqueeze(0).expand(b, -1, -1)

    out = x.new_zeros((b, cout, h, w))
    k = 0
    for ky in range(kh):
        for kx in range(kw):
            dy = offset[:, 2 * k, :, :]
            dx = offset[:, 2 * k + 1, :, :]
            sy = yy + (ky - padding) + dy
            sx = xx + (kx - padding) + dx

            if h > 1:
                gy = 2.0 * sy / (h - 1) - 1.0
            else:
                gy = torch.zeros_like(sy)
            if w > 1:
                gx = 2.0 * sx / (w - 1) - 1.0
            else:
                gx = torch.zeros_like(sx)
            grid = torch.stack([gx, gy], dim=-1)
            sampled = F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled * mask[:, k : k + 1]
            out = out + torch.einsum("oi,bihw->bohw", weight[:, :, ky, kx], sampled)
            k += 1

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


class SpatialDifferentialOptimizationBranch(nn.Module):
    """SpaDOB with F_spa-driven modulated deformable convolution."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        k_total = kernel_size * kernel_size

        # C_offset itself is explicitly 3x3 in Eq. (20).  Its OUTPUT dimensionality
        # is 3*K_total: 2*K_total offsets plus K_total modulation masks.
        self.offset_estimator = nn.Conv2d(
            channels, 3 * k_total, kernel_size=3, padding=1
        )
        self.weight = nn.Parameter(
            torch.empty(channels, channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(
        self, spatial_residual: torch.Tensor, spatial_guidance: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        params = self.offset_estimator(spatial_guidance)
        k_total = self.kernel_size * self.kernel_size
        offset = params[:, : 2 * k_total]
        mask = torch.sigmoid(params[:, 2 * k_total :])

        if tv_deform_conv2d is not None:
            z3 = tv_deform_conv2d(
                spatial_residual,
                offset,
                self.weight,
                self.bias,
                stride=(1, 1),
                padding=(self.padding, self.padding),
                dilation=(1, 1),
                mask=mask,
            )
        else:
            z3 = _fallback_modulated_deform_conv2d(
                spatial_residual,
                offset,
                mask,
                self.weight,
                self.bias,
                self.padding,
            )
        return z3, {"offset": offset, "mask": mask}


class CrossComplementaryRefinementModule(nn.Module):
    """CCRM with dual-domain residual back-projection."""

    def __init__(
        self,
        hsi_channels: int,
        fasa_tau: float = 1.0,
        deform_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.spe_dob = FrequencyAwareSpectralAttention(hsi_channels, tau=fasa_tau)
        self.spa_dob = SpatialDifferentialOptimizationBranch(
            hsi_channels, kernel_size=deform_kernel_size
        )

    def forward(
        self,
        z1: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        spectral_guidance: torch.Tensor,
        spatial_guidance: torch.Tensor,
        srf: torch.Tensor,
        degradation_operator: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Eq. (15): R_spe = U_spa(X - D_spa(Z1)); paper explicitly gives
        # bicubic as the simple U_spa example. D_spa itself is the exact same
        # injected Gaussian+Bicubic observation operator used to generate X.
        degraded_z1 = degradation_operator.degrade(z1)
        lr_residual = lr_hsi - degraded_z1
        r_spe = F.interpolate(
            lr_residual,
            size=z1.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        # Eq. (16): R_spa = (Y - Z1 x_3 P) x_3 P^dagger.
        P = srf.to(z1)
        projected = project_hsi_to_msi(z1, P)
        msi_residual = hr_msi - projected
        P_pinv = torch.linalg.pinv(P)  # Ch x Cm
        r_spa = torch.einsum("cm,bmhw->bchw", P_pinv, msi_residual)

        z2, affinity = self.spe_dob(r_spe, spectral_guidance)
        z3, spatial_aux = self.spa_dob(r_spa, spatial_guidance)
        z = z1 + z2 + z3

        aux = {
            "spectral_residual": r_spe,
            "spatial_residual": r_spa,
            "spectral_refinement": z2,
            "spatial_refinement": z3,
            "fasa_affinity": affinity,
            **spatial_aux,
        }
        return z, aux


class UFGNet(nn.Module):
    """Complete UFGNet: QIEM -> FGM -> CCRM."""

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        srf: torch.Tensor,
        rank: int = 5,
        qiem_regularization: float = 1e-4,
        fasa_tau: float = 1.0,
        spectral_gate_kernel: int = 3,
        deform_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        srf = torch.as_tensor(srf, dtype=torch.float32)
        if srf.shape != (msi_channels, hsi_channels):
            raise ValueError(
                f"Expected SRF {(msi_channels, hsi_channels)}, got {tuple(srf.shape)}"
            )
        self.register_buffer("srf", srf.clone())

        self.qiem = QuickInitialEstimationModule(
            hsi_channels,
            msi_channels,
            rank=rank,
            regularization=qiem_regularization,
        )
        self.fgm = FrequencyGuidedModule(
            hsi_channels, spectral_kernel_size=spectral_gate_kernel
        )
        self.ccrm = CrossComplementaryRefinementModule(
            hsi_channels,
            fasa_tau=fasa_tau,
            deform_kernel_size=deform_kernel_size,
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        degradation_operator: nn.Module,
        return_aux: bool = False,
    ):
        z1, qiem_aux = self.qiem(lr_hsi, hr_msi, self.srf)
        f_spe, f_spa = self.fgm(z1)
        z, ccrm_aux = self.ccrm(
            z1,
            lr_hsi,
            hr_msi,
            f_spe,
            f_spa,
            self.srf,
            degradation_operator,
        )
        if not return_aux:
            return z
        return z, {
            "initial_estimate": z1,
            "spectral_guidance": f_spe,
            "spatial_guidance": f_spa,
            **qiem_aux,
            **ccrm_aux,
        }
