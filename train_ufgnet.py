"""Train/evaluate UFGNet under the agreed 4x Gaussian+Bicubic + SRF protocol."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import get_checkpoint_path, parse_args, print_config
from losses import UFGNetLoss
from metrics import MetricAverager, calc_metrics
from models import UFGNet
from ufgnet_data import build_ufgnet_loaders


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


def build_model_and_loss(cfg, info, device):
    srf_weights = info.get("srf_weights")
    if srf_weights is None:
        raise RuntimeError(
            "UFGNet reproduction requires msi_mode='srf' so QIEM/CCRM/loss share P."
        )
    srf = torch.as_tensor(srf_weights, dtype=torch.float32)
    model = UFGNet(
        hsi_channels=info["n_bands"],
        msi_channels=info["n_select_bands"],
        srf=srf,
        rank=cfg.ufg_rank,
        qiem_regularization=cfg.ufg_qiem_regularization,
        fasa_tau=cfg.ufg_fasa_tau,
        spectral_gate_kernel=cfg.ufg_spectral_gate_kernel,
        deform_kernel_size=cfg.ufg_deform_kernel_size,
    ).to(device)

    degradation = info["degradation_operator"].to(device)
    criterion = UFGNetLoss(
        degradation_operator=degradation,
        srf=srf,
        lambda_rec=cfg.lambda_rec,
        lambda_sam=cfg.lambda_sam,
        lambda_freq=cfg.lambda_freq,
        gamma=cfg.freq_gamma,
        eta=cfg.freq_eta,
    ).to(device)
    return model, criterion, degradation


def train_one_epoch(model, criterion, degradation, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {}
    count = 0

    for batch in loader:
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(lr_hsi, hr_msi, degradation)
        loss, parts = criterion(pred, lr_hsi, hr_msi)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()

        count += 1
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())

    if count == 0:
        raise RuntimeError("Training loader is empty")
    return {key: value / count for key, value in sums.items()}


def _out_of_range_fraction(x: torch.Tensor) -> float:
    return float(((x < 0.0) | (x > 1.0)).float().mean().item())


@torch.no_grad()
def evaluate(
    model,
    degradation,
    loader,
    device,
    scale_ratio: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Full-scene monitoring evaluation; GT never affects optimization."""
    model.eval()
    meter = MetricAverager()
    diagnostics = {"pred_min": float("inf"), "pred_max": float("-inf"), "oor": 0.0}
    n = 0
    for batch in loader:
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        pred = model(lr_hsi, hr_msi, degradation)
        meter.update(calc_metrics(pred, gt, scale_ratio=scale_ratio))
        diagnostics["pred_min"] = min(diagnostics["pred_min"], float(pred.min().item()))
        diagnostics["pred_max"] = max(diagnostics["pred_max"], float(pred.max().item()))
        diagnostics["oor"] += _out_of_range_fraction(pred)
        n += 1
    diagnostics["oor"] /= max(n, 1)
    return meter.average(), diagnostics


@torch.no_grad()
def save_full_scene_prediction(model, degradation, loader, device, path: str) -> None:
    model.eval()
    batch = next(iter(loader))
    lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
    hr_msi = batch["hr_msi"].to(device, non_blocking=True)
    pred = model(lr_hsi, hr_msi, degradation)
    array = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def save_checkpoint(path, model, optimizer, epoch, cfg, metrics=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            key: value
            for key, value in cfg.__dict__.items()
            if key != "datasets"
        },
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    cfg = parse_args()
    print_config(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    if cfg.scale_ratio != 4:
        print(f"[warning] configured scale_ratio={cfg.scale_ratio}; agreed baseline is 4x")
    if cfg.degradation_mode != "gaussian_bicubic":
        print(
            f"[warning] degradation_mode={cfg.degradation_mode}; "
            "agreed first reproduction baseline is gaussian_bicubic"
        )

    train_loader, test_loader, info = build_ufgnet_loaders(cfg)
    print(
        f"Training patches={info['train_samples']} | "
        f"full scene={info['full_scene_shape']} | "
        f"protocol={info['sampling_protocol']}"
    )
    print(
        f"Retained HR-MSI channels={info['n_select_bands']} | "
        f"bands={info.get('srf_band_names')}"
    )

    model, criterion, degradation = build_model_and_loss(cfg, info, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    start_epoch = 1
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {cfg.resume}, epoch={start_epoch}")

    final_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    history_path = os.path.join(
        cfg.output_root, "metrics", f"{cfg.dataset}_ufgnet_history.json"
    )
    total_start = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        losses = train_one_epoch(
            model, criterion, degradation, train_loader, optimizer, device
        )
        record: Dict[str, float] = {"epoch": int(epoch), **losses}
        line = (
            f"Epoch {epoch:04d}/{cfg.epochs} "
            f"loss={losses['loss_total']:.6f} "
            f"rec={losses['loss_rec']:.6f} "
            f"sam={losses['loss_sam']:.6f} "
            f"freq={losses['loss_freq']:.6f}"
        )

        # Full-reference metrics are monitoring only; they never select a model,
        # stop training, or enter the loss.
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            final_metrics, diag = evaluate(
                model, degradation, test_loader, device, cfg.scale_ratio
            )
            record.update(final_metrics)
            record.update({f"diag_{k}": v for k, v in diag.items()})
            metric_text = " ".join(
                f"{key}={value:.6f}" for key, value in final_metrics.items()
            )
            line += (
                " | " + metric_text
                + f" pred=[{diag['pred_min']:.4f},{diag['pred_max']:.4f}]"
                + f" oor={diag['oor']:.4f}"
            )

        record["seconds"] = float(time.time() - epoch_start)
        history.append(record)
        _write_json(history_path, history)

        line += f" | {record['seconds']:.1f}s"
        print(line)

        if epoch % cfg.save_interval == 0:
            name = cfg.save_name or f"{cfg.dataset}_ufgnet_epoch{epoch}.pth"
            path = get_checkpoint_path(cfg, stage="ufgnet", name=name)
            save_checkpoint(path, model, optimizer, epoch, cfg, final_metrics)

    # Always perform one final full-scene evaluation after the fixed epoch budget.
    final_metrics, final_diag = evaluate(
        model, degradation, test_loader, device, cfg.scale_ratio
    )

    final_name = cfg.save_name or f"{cfg.dataset}_ufgnet_final.pth"
    final_path = get_checkpoint_path(cfg, stage="ufgnet", name=final_name)
    save_checkpoint(final_path, model, optimizer, cfg.epochs, cfg, final_metrics)

    metrics_path = os.path.join(
        cfg.output_root, "metrics", f"{cfg.dataset}_ufgnet_metrics.json"
    )
    _write_json(
        metrics_path,
        {
            "metrics": final_metrics,
            "diagnostics": final_diag,
            "sampling_protocol": info["sampling_protocol"],
            "full_scene_shape": info["full_scene_shape"],
            "msi_channels": info["n_select_bands"],
            "srf_band_names": info.get("srf_band_names"),
            "srf_coverage_policy": cfg.srf_coverage_policy,
            "srf_min_coverage_ratio": cfg.srf_min_coverage_ratio,
            "srf_coverage_diagnostics": info.get("srf_coverage_diagnostics", []),
        },
    )

    prediction_path = os.path.join(
        cfg.output_root,
        "predictions",
        cfg.dataset,
        f"{cfg.dataset}_ufgnet_final.npy",
    )
    save_full_scene_prediction(model, degradation, test_loader, device, prediction_path)

    print(f"Training finished in {time.time() - total_start:.1f}s")
    print(f"Checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")
    print(f"History: {history_path}")
    print(f"Prediction: {prediction_path}")


if __name__ == "__main__":
    main()
