"""Diagnose a UFGNet training plateau on a saved checkpoint.

This script does not change the network or training protocol. It measures four
common causes of a low full-scene PSNR plateau under the current reproduction:
1) how hard the configured observation pair is (bicubic LR-HSI baseline),
2) whether the paper's r=5 still captures >=99% spectral energy on this scene,
3) whether the chosen SRF leaves many HSI bands weakly/unobserved spatially,
4) whether QIEM BatchNorm running statistics create a train/eval gap.

Example:
python diagnose_ufgnet_plateau.py --dataset PaviaU --resume checkpoints/ufgnet/PaviaU_ufgnet_epoch70.pth --device cuda
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import parse_args, print_config
from metrics import calc_metrics
from train_ufgnet import build_model_and_loss, resolve_device, seed_everything
from ufgnet_data import build_ufgnet_loaders


def _fmt(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:.6f}" for k, v in metrics.items())


def _per_band_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = torch.mean((pred.float() - gt.float()) ** 2, dim=(0, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(eps))


def _rank_energy(lr_hsi: torch.Tensor) -> torch.Tensor:
    x = lr_hsi.squeeze(0).reshape(lr_hsi.shape[1], -1)
    s = torch.linalg.svdvals(x.float())
    energy = s.square()
    return torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)


def _affinity_health(a: torch.Tensor) -> tuple[float, float]:
    a = a.detach().float().clamp_min(1e-12)
    row_max = a.max(dim=-1).values.mean().item()
    entropy = -(a * torch.log(a)).sum(dim=-1) / math.log(max(a.shape[-1], 2))
    return float(row_max), float(entropy.mean().item())


def _bn_batch_stat_prediction(model, lr_hsi, hr_msi, degradation):
    """Use current full-scene batch statistics for BN, then restore buffers."""
    bn_state = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            bn_state.append(
                (
                    module,
                    module.training,
                    module.running_mean.detach().clone() if module.running_mean is not None else None,
                    module.running_var.detach().clone() if module.running_var is not None else None,
                    module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else None,
                )
            )
            module.training = True

    with torch.no_grad():
        pred = model(lr_hsi, hr_msi, degradation)

    for module, was_training, mean, var, nbt in bn_state:
        module.training = was_training
        if mean is not None:
            module.running_mean.copy_(mean)
        if var is not None:
            module.running_var.copy_(var)
        if nbt is not None:
            module.num_batches_tracked.copy_(nbt)
    return pred


def main() -> None:
    cfg = parse_args()
    if not cfg.resume:
        raise ValueError("Pass the checkpoint to diagnose with --resume PATH")
    print_config(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    _, test_loader, info = build_ufgnet_loaders(cfg)
    model, criterion, degradation = build_model_and_loss(cfg, info, device)
    ckpt = torch.load(cfg.resume, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = next(iter(test_loader))
    lr_hsi = batch["lr_hsi"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    gt = batch["gt"].to(device)
    srf = model.srf.to(device)

    print("\n[scene]")
    print(f"GT={tuple(gt.shape)} LR-HSI={tuple(lr_hsi.shape)} HR-MSI={tuple(hr_msi.shape)}")
    print(f"checkpoint_epoch={ckpt.get('epoch', 'unknown')}")

    # 1) Difficulty baseline: spatial interpolation of the actual LR-HSI.
    bicubic = F.interpolate(lr_hsi, size=gt.shape[-2:], mode="bicubic", align_corners=False)
    print("\n[baseline: LR-HSI bicubic upsample]")
    print(_fmt(calc_metrics(bicubic, gt, scale_ratio=cfg.scale_ratio)))

    # 2) Check whether r=5 remains justified on the user's 103-band/full-scene data.
    cumulative = _rank_energy(lr_hsi)
    print("\n[spectral subspace energy]")
    for r in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
        if r <= cumulative.numel():
            print(f"rank={r:02d} cumulative_energy={float(cumulative[r-1]):.8f}")
    r99 = int(torch.nonzero(cumulative >= 0.99, as_tuple=False)[0].item() + 1) if torch.any(cumulative >= 0.99) else int(cumulative.numel())
    print(f"rank_needed_for_99pct={r99}")

    # 3) SRF coverage. Column sums show how strongly each HSI band participates
    # in any HR-MSI observation under the configured SRF.
    coverage = srf.abs().sum(dim=0)
    cov_rel = coverage / coverage.max().clamp_min(1e-12)
    weak_1pct = cov_rel < 1e-2
    weak_01pct = cov_rel < 1e-3
    print("\n[SRF spectral coverage]")
    print(f"MSI_channels={srf.shape[0]} HSI_channels={srf.shape[1]}")
    print(f"bands_below_1pct_of_peak={int(weak_1pct.sum().item())}/{srf.shape[1]}")
    print(f"bands_below_0.1pct_of_peak={int(weak_01pct.sum().item())}/{srf.shape[1]}")
    if info.get("hsi_wavelengths") is not None:
        wavelengths = torch.as_tensor(info["hsi_wavelengths"], device=device)
        if torch.any(weak_1pct):
            weak_wl = wavelengths[weak_1pct]
            print(f"weak_coverage_wavelength_range={float(weak_wl.min()):.1f}-{float(weak_wl.max()):.1f} nm")

    with torch.no_grad():
        initial, qaux = model.qiem(lr_hsi, hr_msi, srf)
        pred, aux = model(lr_hsi, hr_msi, degradation, return_aux=True)
        loss, parts = criterion(pred, lr_hsi, hr_msi)

    print("\n[trained QIEM components]")
    print("spatial: ", _fmt(calc_metrics(qaux["qiem_spatial"], gt, cfg.scale_ratio)))
    print("subspace:", _fmt(calc_metrics(qaux["qiem_subspace"], gt, cfg.scale_ratio)))
    print("fused Z1:", _fmt(calc_metrics(initial, gt, cfg.scale_ratio)))

    print("\n[full UFGNet eval-mode]")
    full_metrics = calc_metrics(pred, gt, cfg.scale_ratio)
    print(_fmt(full_metrics))
    print(f"pred_min={pred.min().item():.6f} pred_max={pred.max().item():.6f} oor={(((pred<0)|(pred>1)).float().mean().item()):.6f}")
    print(
        f"loss_total={loss.item():.8f} rec={parts['loss_rec'].item():.8f} "
        f"sam={parts['loss_sam'].item():.8f} freq={parts['loss_freq'].item():.8f}"
    )
    rowmax, entropy = _affinity_health(aux["fasa_affinity"])
    print(f"FASA_rowmax_mean={rowmax:.8f} FASA_entropy_mean={entropy:.8f}")

    # Per-band performance split by SRF coverage.
    band_psnr = _per_band_psnr(pred, gt)
    print("\n[per-band PSNR vs SRF coverage]")
    if torch.any(~weak_1pct):
        print(f"well_covered_mean_PSNR={band_psnr[~weak_1pct].mean().item():.6f}")
    if torch.any(weak_1pct):
        print(f"weak_coverage_mean_PSNR={band_psnr[weak_1pct].mean().item():.6f}")
    worst = torch.argsort(band_psnr)[:10]
    wavelengths = info.get("hsi_wavelengths")
    for idx in worst.tolist():
        wl = float(wavelengths[idx]) if wavelengths is not None else float("nan")
        print(
            f"band={idx:03d} wl={wl:.1f}nm PSNR={band_psnr[idx].item():.4f} "
            f"coverage_rel={cov_rel[idx].item():.6g}"
        )

    # 4) Detect a BN running-statistics mismatch between patch training and
    # full-scene eval. A large PSNR jump here indicates BN is a real bottleneck.
    model.eval()
    pred_batch_bn = _bn_batch_stat_prediction(model, lr_hsi, hr_msi, degradation)
    bn_metrics = calc_metrics(pred_batch_bn, gt, cfg.scale_ratio)
    print("\n[BN diagnostic: full-scene batch stats instead of running stats]")
    print(_fmt(bn_metrics))
    print(f"delta_PSNR_vs_eval={bn_metrics['PSNR'] - full_metrics['PSNR']:+.6f} dB")
    print(f"delta_SAM_vs_eval={bn_metrics['SAM'] - full_metrics['SAM']:+.6f} deg")


if __name__ == "__main__":
    main()
