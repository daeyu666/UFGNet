"""Short fixed-batch optimization smoke test for UFGNet.

This is deliberately not a benchmark training script. It repeatedly optimizes
one aligned real training patch sampled from the pre-simulated scene pair so we
can verify that the exact UFGNet forward/loss chain is numerically trainable.

Use --epochs as the number of smoke optimization steps, e.g. --epochs 20.
"""

from __future__ import annotations

import math

import torch

from config import parse_args, print_config
from metrics import calc_metrics
from train_ufgnet import build_model_and_loss, resolve_device, seed_everything
from ufgnet_data import build_ufgnet_loaders


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach().float() ** 2).item())
    return math.sqrt(total)


def _out_of_range_fraction(x: torch.Tensor) -> float:
    return float(((x.detach() < 0.0) | (x.detach() > 1.0)).float().mean().item())


def _affinity_health(affinity: torch.Tensor) -> tuple[float, float]:
    a = affinity.detach().float().clamp_min(1e-12)
    row_max = a.max(dim=-1).values.mean().item()
    entropy = -(a * torch.log(a)).sum(dim=-1)
    entropy = entropy / math.log(max(a.shape[-1], 2))
    return float(row_max), float(entropy.mean().item())


def main() -> None:
    cfg = parse_args()
    print_config(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, _, info = build_ufgnet_loaders(cfg)
    model, criterion, degradation = build_model_and_loss(cfg, info, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    batch = next(iter(train_loader))
    lr_hsi = batch["lr_hsi"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    gt = batch["gt"].to(device)

    steps = max(int(cfg.epochs), 1)
    report_steps = {0, 1, 2, 5, 10, steps}
    if steps > 10:
        report_steps.add(steps // 2)

    initial_loss = None
    final_loss = None

    for step in range(0, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(lr_hsi, hr_msi, degradation, return_aux=True)
        loss, parts = criterion(pred, lr_hsi, hr_msi)

        if not torch.isfinite(pred).all() or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite value at smoke step {step}")

        if step == 0:
            initial_loss = float(loss.item())

        if step in report_steps:
            affinity_max, affinity_entropy = _affinity_health(aux["fasa_affinity"])
            metrics = calc_metrics(pred, gt, scale_ratio=cfg.scale_ratio)
            print(
                f"step={step:03d} "
                f"loss={loss.item():.8f} "
                f"rec={parts['loss_rec'].item():.8f} "
                f"sam={parts['loss_sam'].item():.8f} "
                f"freq={parts['loss_freq'].item():.8f} "
                f"pred_min={pred.min().item():.6f} "
                f"pred_max={pred.max().item():.6f} "
                f"oor={_out_of_range_fraction(pred):.6f} "
                f"fasa_rowmax={affinity_max:.6f} "
                f"fasa_entropy={affinity_entropy:.6f} "
                f"PSNR={metrics['PSNR']:.4f} "
                f"SAMdeg={metrics['SAM']:.4f}"
            )

        if step == steps:
            final_loss = float(loss.item())
            break

        loss.backward()
        if step in report_steps:
            print(
                f"          grad QIEM={_grad_norm(model.qiem):.6g} "
                f"FGM={_grad_norm(model.fgm):.6g} "
                f"CCRM={_grad_norm(model.ccrm):.6g}"
            )
        optimizer.step()

    assert initial_loss is not None and final_loss is not None
    print(
        f"\nSmoke summary: initial_loss={initial_loss:.8f} "
        f"final_loss={final_loss:.8f} "
        f"ratio={final_loss / max(initial_loss, 1e-12):.6f}"
    )
    if final_loss >= initial_loss:
        raise RuntimeError(
            "Fixed-batch smoke loss did not decrease. Do not start long training yet."
        )
    print("Fixed-batch optimization smoke test PASSED.")


if __name__ == "__main__":
    main()
