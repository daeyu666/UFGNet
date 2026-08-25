"""One-batch real-data sanity check for the UFGNet reproduction.

Run this before a long training job. It uses the exact configured scene-level
operator/SRF, samples one aligned overlapping training patch, executes
QIEM -> FGM -> CCRM, evaluates the unsupervised loss, and performs one backward
pass while reporting intermediate diagnostics.
"""

from __future__ import annotations

import math

import torch

from config import parse_args, print_config
from metrics import calc_metrics
from train_ufgnet import build_model_and_loss, resolve_device, seed_everything
from ufgnet_data import build_ufgnet_loaders


def _stats(x: torch.Tensor) -> str:
    x = x.detach().float()
    return (
        f"shape={tuple(x.shape)} min={x.min().item():.6g} "
        f"mean={x.mean().item():.6g} max={x.max().item():.6g} "
        f"std={x.std(unbiased=False).item():.6g}"
    )


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach().float() ** 2).item())
    return math.sqrt(total)


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _out_of_range_fraction(x: torch.Tensor) -> float:
    return float(((x.detach() < 0.0) | (x.detach() > 1.0)).float().mean().item())


def main() -> None:
    cfg = parse_args()
    print_config(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, _, info = build_ufgnet_loaders(cfg)
    model, criterion, degradation = build_model_and_loss(cfg, info, device)
    model.train()

    batch = next(iter(train_loader))
    lr_hsi = batch["lr_hsi"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    gt = batch["gt"].to(device)

    print("\n[input]")
    print("LR-HSI:", _stats(lr_hsi))
    print("HR-MSI:", _stats(hr_msi))
    print("GT monitor only:", _stats(gt))

    pred, aux = model(lr_hsi, hr_msi, degradation, return_aux=True)
    loss, parts = criterion(pred, lr_hsi, hr_msi)

    if not torch.isfinite(pred).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite prediction or loss before backward")

    loss.backward()

    print("\n[model]")
    print(f"trainable params total={_param_count(model):,}")
    print(f"QIEM params={_param_count(model.qiem):,}")
    print(f"FGM params={_param_count(model.fgm):,}")
    print(f"CCRM params={_param_count(model.ccrm):,}")
    print("prediction:", _stats(pred))
    print(f"prediction out-of-[0,1] fraction={_out_of_range_fraction(pred):.8f}")

    print("\n[QIEM]")
    print("spatial:", _stats(aux["qiem_spatial"]))
    print("subspace:", _stats(aux["qiem_subspace"]))
    print("beta:", _stats(aux["qiem_beta"]))

    print("\n[FGM]")
    print("F_spe:", _stats(aux["spectral_guidance"]))
    print("F_spa:", _stats(aux["spatial_guidance"]))

    print("\n[CCRM]")
    print("R_spe:", _stats(aux["spectral_residual"]))
    print("R_spa:", _stats(aux["spatial_residual"]))
    print("Z2:", _stats(aux["spectral_refinement"]))
    print("Z3:", _stats(aux["spatial_refinement"]))
    print("offset:", _stats(aux["offset"]))
    print("mask:", _stats(aux["mask"]))
    affinity = aux["fasa_affinity"].detach()
    row_sums = affinity.sum(dim=-1)
    a = affinity.float().clamp_min(1e-12)
    row_max = a.max(dim=-1).values
    entropy = -(a * torch.log(a)).sum(dim=-1) / math.log(max(a.shape[-1], 2))
    print("FASA affinity:", _stats(affinity))
    print(
        f"FASA row-sum min={row_sums.min().item():.8f} "
        f"max={row_sums.max().item():.8f}"
    )
    print(
        f"FASA row-max mean={row_max.mean().item():.8f} "
        f"max={row_max.max().item():.8f}"
    )
    print(
        f"FASA normalized entropy mean={entropy.mean().item():.8f} "
        f"min={entropy.min().item():.8f}"
    )
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise RuntimeError("FASA affinity rows do not sum to 1")

    print("\n[loss]")
    print(f"total={loss.item():.8f}")
    for key, value in parts.items():
        print(f"{key}={float(value.item()):.8f}")
    print(
        "weighted contributions: "
        f"rec={cfg.lambda_rec * parts['loss_rec'].item():.8f} "
        f"sam={cfg.lambda_sam * parts['loss_sam'].item():.8f} "
        f"freq={cfg.lambda_freq * parts['loss_freq'].item():.8f}"
    )

    print("\n[gradient norms]")
    print(f"QIEM={_grad_norm(model.qiem):.8g}")
    print(f"FGM={_grad_norm(model.fgm):.8g}")
    print(f"CCRM={_grad_norm(model.ccrm):.8g}")

    metrics = calc_metrics(pred, gt, scale_ratio=cfg.scale_ratio)
    print("\n[GT metrics: monitoring only; raw prediction, no metric clipping]")
    for key, value in metrics.items():
        print(f"{key}={value:.8f}")

    for name, module in (("QIEM", model.qiem), ("FGM", model.fgm), ("CCRM", model.ccrm)):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        if not any(g is not None and torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"{name} has no finite trainable gradient")

    print("\nSanity check PASSED.")


if __name__ == "__main__":
    main()
