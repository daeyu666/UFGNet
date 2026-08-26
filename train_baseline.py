"""Train the plain CNN baseline on the same observation pair used by UFGNet.

This is a supervised diagnostic baseline: GT HR-HSI patches are used directly in
training. It is therefore NOT an unsupervised apples-to-apples competitor to
UFGNet, but it is useful for checking whether the current PaviaU + IKONOS4 data
protocol can support a high-quality reconstruction with a simple CNN.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from baseline import baseline
from config import get_checkpoint_path, parse_args, print_config
from metrics import MetricAverager, calc_metrics
from ufgnet_data import build_ufgnet_loaders


def parse_baseline_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--baseline_loss",
        type=str,
        default="l1",
        choices=["l1", "mse", "l1_mse"],
        help="Supervised HR-HSI reconstruction loss.",
    )
    parser.add_argument("--baseline_channels", type=int, default=64)
    parser.add_argument("--baseline_num_blocks", type=int, default=8)
    known, remaining = parser.parse_known_args(argv)
    return known, parse_args(remaining)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def supervised_loss(pred: torch.Tensor, gt: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(pred, gt)
    if mode == "mse":
        return F.mse_loss(pred, gt)
    if mode == "l1_mse":
        return F.l1_loss(pred, gt) + F.mse_loss(pred, gt)
    raise ValueError(mode)


def train_one_epoch(model, loader, optimizer, device, loss_mode: str) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(lr_hsi, hr_msi)
        loss = supervised_loss(pred, gt, loss_mode)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite baseline training loss")
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        count += 1

    if count == 0:
        raise RuntimeError("Training loader is empty")
    return total / count


def _out_of_range_fraction(x: torch.Tensor) -> float:
    return float(((x < 0.0) | (x > 1.0)).float().mean().item())


@torch.no_grad()
def evaluate(model, loader, device, scale_ratio: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    meter = MetricAverager()
    diagnostics = {"pred_min": float("inf"), "pred_max": float("-inf"), "oor": 0.0}
    n = 0
    for batch in loader:
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        pred, _ = model(lr_hsi, hr_msi)

        meter.update(calc_metrics(pred, gt, scale_ratio=scale_ratio))
        diagnostics["pred_min"] = min(diagnostics["pred_min"], float(pred.min().item()))
        diagnostics["pred_max"] = max(diagnostics["pred_max"], float(pred.max().item()))
        diagnostics["oor"] += _out_of_range_fraction(pred)
        n += 1

    diagnostics["oor"] /= max(n, 1)
    return meter.average(), diagnostics


def save_checkpoint(path, model, optimizer, epoch, cfg, baseline_args, metrics=None):
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {key: value for key, value in cfg.__dict__.items() if key != "datasets"},
        "baseline": vars(baseline_args),
        "metrics": metrics or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@torch.no_grad()
def save_full_scene_prediction(model, loader, device, path: str) -> None:
    model.eval()
    batch = next(iter(loader))
    lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
    hr_msi = batch["hr_msi"].to(device, non_blocking=True)
    pred, _ = model(lr_hsi, hr_msi)
    array = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def main(argv=None) -> None:
    baseline_args, cfg = parse_baseline_args(argv)
    print_config(cfg)
    print(
        "Baseline config: "
        f"loss={baseline_args.baseline_loss}, "
        f"channels={baseline_args.baseline_channels}, "
        f"blocks={baseline_args.baseline_num_blocks}"
    )

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, test_loader, info = build_ufgnet_loaders(cfg)
    print(
        f"Observation protocol: HSI={info['n_bands']} bands, "
        f"MSI={info['n_select_bands']} bands {info.get('srf_band_names')}, "
        f"spectral={info.get('spectral_protocol')}, "
        f"full_scene={info['full_scene_shape']}"
    )

    model = baseline(
        scale_ratio=cfg.scale_ratio,
        n_select_bands=info["n_select_bands"],
        n_bands=info["n_bands"],
        dataset=cfg.dataset,
        channels=baseline_args.baseline_channels,
        num_blocks=baseline_args.baseline_num_blocks,
    ).to(device)
    print(f"Baseline trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history: List[Dict[str, float]] = []
    best_psnr = float("-inf")
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    best_diag: Dict[str, float] = {}
    final_metrics: Dict[str, float] = {}
    final_diag: Dict[str, float] = {}

    history_path = os.path.join(
        cfg.output_root, "metrics", f"{cfg.dataset}_baseline_history.json"
    )
    total_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        loss = train_one_epoch(
            model, train_loader, optimizer, device, baseline_args.baseline_loss
        )
        record: Dict[str, float] = {"epoch": int(epoch), "loss": float(loss)}
        line = f"Epoch {epoch:04d}/{cfg.epochs} loss={loss:.6f}"

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            final_metrics, final_diag = evaluate(
                model, test_loader, device, cfg.scale_ratio
            )
            record.update(final_metrics)
            record.update({f"diag_{k}": v for k, v in final_diag.items()})
            metric_text = " ".join(
                f"{key}={value:.6f}" for key, value in final_metrics.items()
            )
            line += (
                " | " + metric_text
                + f" pred=[{final_diag['pred_min']:.4f},{final_diag['pred_max']:.4f}]"
                + f" oor={final_diag['oor']:.4f}"
            )

            psnr = float(final_metrics.get("PSNR", float("-inf")))
            if psnr > best_psnr:
                best_psnr = psnr
                best_epoch = epoch
                best_metrics = dict(final_metrics)
                best_diag = dict(final_diag)
                best_path = get_checkpoint_path(
                    cfg, stage="baseline", name=f"{cfg.dataset}_baseline_best.pth"
                )
                save_checkpoint(
                    best_path, model, optimizer, epoch, cfg, baseline_args, best_metrics
                )

        record["seconds"] = float(time.time() - epoch_start)
        history.append(record)
        _write_json(history_path, history)
        line += f" | {record['seconds']:.1f}s"
        print(line)

        if epoch % cfg.save_interval == 0:
            path = get_checkpoint_path(
                cfg, stage="baseline", name=f"{cfg.dataset}_baseline_epoch{epoch}.pth"
            )
            save_checkpoint(
                path, model, optimizer, epoch, cfg, baseline_args, final_metrics
            )

    final_metrics, final_diag = evaluate(model, test_loader, device, cfg.scale_ratio)
    final_path = get_checkpoint_path(
        cfg, stage="baseline", name=f"{cfg.dataset}_baseline_final.pth"
    )
    save_checkpoint(
        final_path, model, optimizer, cfg.epochs, cfg, baseline_args, final_metrics
    )

    metrics_path = os.path.join(
        cfg.output_root, "metrics", f"{cfg.dataset}_baseline_metrics.json"
    )
    _write_json(
        metrics_path,
        {
            "final_metrics": final_metrics,
            "final_diagnostics": final_diag,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_diagnostics": best_diag,
            "baseline": vars(baseline_args),
            "spectral_protocol": info.get("spectral_protocol"),
            "hsi_channels": info["n_bands"],
            "msi_channels": info["n_select_bands"],
            "srf_band_names": info.get("srf_band_names"),
            "sampling_protocol": info["sampling_protocol"],
        },
    )

    prediction_path = os.path.join(
        cfg.output_root,
        "predictions",
        cfg.dataset,
        f"{cfg.dataset}_baseline_final.npy",
    )
    save_full_scene_prediction(model, test_loader, device, prediction_path)

    print(f"Training finished in {time.time() - total_start:.1f}s")
    print(f"Best epoch={best_epoch} best_PSNR={best_psnr:.6f}")
    if best_metrics:
        print("Best metrics: " + " ".join(f"{k}={v:.6f}" for k, v in best_metrics.items()))
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")
    print(f"History: {history_path}")
    print(f"Prediction: {prediction_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
