"""Evaluate a trained supervised CNN baseline under a different degradation.

Typical diagnostic:
    train checkpoint: gaussian_bicubic
    evaluation pair: physical

The model is NOT updated in this script. The target LR-HSI/HR-MSI/GT pair is
rebuilt from the current command-line data protocol, so only the observation
operator changes when the remaining dataset/SRF settings are held fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import torch

from baseline import baseline
from config import parse_args, print_config
from metrics import MetricAverager, calc_metrics
from ufgnet_data import build_ufgnet_loaders


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a train_baseline.py checkpoint.",
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default="",
        help="Optional basename for the output JSON.",
    )
    known, remaining = parser.parse_known_args(argv)
    return known, parse_args(remaining)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


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


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv=None) -> None:
    eval_args, cfg = parse_eval_args(argv)
    print_config(cfg)
    device = resolve_device(cfg.device)

    if not os.path.exists(eval_args.checkpoint):
        raise FileNotFoundError(eval_args.checkpoint)

    # Build the target observation pair from the requested degradation.
    _, test_loader, info = build_ufgnet_loaders(cfg)

    checkpoint = torch.load(eval_args.checkpoint, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    baseline_cfg = checkpoint.get("baseline", {})

    train_dataset = checkpoint_cfg.get("dataset", "unknown")
    train_degradation = checkpoint_cfg.get("degradation_mode", "unknown")
    train_spectral = checkpoint_cfg.get("pavia_spectral_protocol", "unknown")
    train_srf = checkpoint_cfg.get("srf_band_set", "unknown")

    print(
        "Checkpoint protocol: "
        f"dataset={train_dataset}, degradation={train_degradation}, "
        f"spectral={train_spectral}, srf={train_srf}, "
        f"epoch={checkpoint.get('epoch', 'unknown')}"
    )
    print(
        "Target protocol: "
        f"dataset={cfg.dataset}, degradation={cfg.degradation_mode}, "
        f"spectral={info.get('spectral_protocol')}, "
        f"HSI={info['n_bands']}, MSI={info['n_select_bands']} "
        f"{info.get('srf_band_names')}"
    )

    if train_dataset != "unknown" and train_dataset != cfg.dataset:
        raise ValueError(
            f"Checkpoint dataset={train_dataset} but target dataset={cfg.dataset}."
        )

    channels = int(baseline_cfg.get("baseline_channels", 64))
    num_blocks = int(baseline_cfg.get("baseline_num_blocks", 8))
    model = baseline(
        scale_ratio=cfg.scale_ratio,
        n_select_bands=info["n_select_bands"],
        n_bands=info["n_bands"],
        dataset=cfg.dataset,
        channels=channels,
        num_blocks=num_blocks,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    metrics, diagnostics = evaluate(model, test_loader, device, cfg.scale_ratio)
    print("Cross-degradation metrics: " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
    print(
        f"Prediction range=[{diagnostics['pred_min']:.6f}, {diagnostics['pred_max']:.6f}] "
        f"oor={diagnostics['oor']:.6f}"
    )

    result_name = eval_args.result_name or (
        f"{cfg.dataset}_baseline_{train_degradation}_to_{cfg.degradation_mode}.json"
    )
    if not result_name.endswith(".json"):
        result_name += ".json"
    result_path = os.path.join(cfg.output_root, "metrics", result_name)
    _write_json(
        result_path,
        {
            "checkpoint": eval_args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "train_protocol": {
                "dataset": train_dataset,
                "degradation_mode": train_degradation,
                "pavia_spectral_protocol": train_spectral,
                "srf_band_set": train_srf,
            },
            "target_protocol": {
                "dataset": cfg.dataset,
                "degradation_mode": cfg.degradation_mode,
                "pavia_spectral_protocol": cfg.pavia_spectral_protocol,
                "srf_band_set": cfg.srf_band_set,
                "hsi_channels": info["n_bands"],
                "msi_channels": info["n_select_bands"],
                "srf_band_names": info.get("srf_band_names"),
            },
            "metrics": metrics,
            "diagnostics": diagnostics,
        },
    )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
