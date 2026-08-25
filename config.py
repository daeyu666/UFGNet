# config.py
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    name: str
    file_name: str
    mat_keys: list
    n_select_bands: int = 8


@dataclass
class TrainConfig:
    project_root: str = "."
    data_root: str = "./data/raw"
    cache_root: str = "./data/cache"
    checkpoint_root: str = "./checkpoints"
    log_root: str = "./logs"
    output_root: str = "./outputs"

    stage: str = "train"
    dataset: str = "PaviaU"

    image_size: int = 128
    patch_size: int = 64
    stride: int = 32
    scale_ratio: int = 4
    n_select_bands: int = 8

    # Current UFGNet reproduction baseline.
    degradation_mode: str = "gaussian_bicubic"
    degradation_sigma: float = 2.0
    degradation_kernel_size: int = 5
    mtf_nyquist: float = 0.2
    psf_truncate: float = 3.0

    progressive_steps: int = 12
    progressive_lift: str = "auto"
    boundary_probability: float = 0.2
    boundary_radius: int = 1

    # Real-SRF MSI simulation. ``auto`` resolves to IKONOS 4-band for PaviaU
    # and WV2 all8 for Houston13/Chikusei. Explicit paths/band sets are kept for
    # ablations and backward-compatible protocol checks.
    msi_mode: str = "srf"
    srf_path: str = ""
    wavelength_root: str = "./data/wavelengths"
    wavelength_path: str = ""
    srf_interp: str = "pchip"
    srf_band_set: str = "auto"
    srf_min_coverage_ratio: float = 0.90
    srf_coverage_policy: str = "filter"

    epochs: int = 300
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 10
    device: str = "cuda"

    # UFGNet loss.
    lambda_rec: float = 1.0
    lambda_sam: float = 1e-2
    lambda_freq: float = 1e-2
    freq_gamma: float = 0.5
    freq_eta: float = 0.5

    # UFGNet architecture. The paper reports r=5 for Pavia. QIEM lambda and
    # FASA tau are not numerically disclosed, so they remain explicit options.
    ufg_rank: int = 5
    ufg_qiem_regularization: float = 1e-4
    ufg_fasa_tau: float = 1.0
    ufg_spectral_gate_kernel: int = 3
    ufg_deform_kernel_size: int = 3

    # Legacy/general fields retained for comparison code compatibility.
    lambda_l1: float = 1.0
    lambda_dc: float = 0.1
    lambda_sgrad: float = 0.05
    lambda_sdir: float = 0.2
    lambda_ns_l1: float = 1.0
    lambda_srf_region: float = 0.3
    lambda_mse: float = 1.0

    save_interval: int = 20
    eval_interval: int = 1
    resume: str = ""
    save_name: str = ""

    datasets: dict = field(default_factory=dict)


def get_dataset_configs():
    return {
        "PaviaU": DatasetConfig(
            name="PaviaU",
            file_name="PaviaU.mat",
            mat_keys=["paviaU", "PaviaU", "img", "data"],
            n_select_bands=4,
        ),
        "Houston13": DatasetConfig(
            name="Houston13",
            file_name="Houston13.mat",
            mat_keys=["Houston13", "Houston_HSI", "data", "img"],
            n_select_bands=8,
        ),
        "Chikusei": DatasetConfig(
            name="Chikusei",
            file_name="Chikusei.mat",
            mat_keys=["chikusei", "Chikusei", "img", "data"],
            n_select_bands=8,
        ),
    }


def resolve_config_defaults(cfg: TrainConfig):
    if cfg.progressive_lift == "auto":
        cfg.progressive_lift = (
            "normalized_adjoint"
            if cfg.degradation_mode == "physical"
            else "bilinear"
        )


def validate_config(cfg: TrainConfig):
    if cfg.scale_ratio < 1:
        raise ValueError("scale_ratio must be >= 1")
    if cfg.progressive_steps < 1:
        raise ValueError("progressive_steps must be >= 1")
    if cfg.degradation_kernel_size < 1 or cfg.degradation_kernel_size % 2 == 0:
        raise ValueError("degradation_kernel_size must be a positive odd integer")
    if cfg.degradation_sigma < 0:
        raise ValueError("degradation_sigma must be >= 0")
    if not 0.0 < cfg.mtf_nyquist <= 1.0:
        raise ValueError("mtf_nyquist must lie in (0, 1]")
    if cfg.psf_truncate <= 0:
        raise ValueError("psf_truncate must be > 0")
    if not 0.0 <= cfg.boundary_probability <= 1.0:
        raise ValueError("boundary_probability must lie in [0, 1]")
    if cfg.boundary_radius < 0:
        raise ValueError("boundary_radius must be >= 0")
    if not 0.0 <= cfg.srf_min_coverage_ratio <= 1.0:
        raise ValueError("srf_min_coverage_ratio must lie in [0, 1]")
    if cfg.srf_coverage_policy not in {"off", "filter", "error"}:
        raise ValueError("srf_coverage_policy must be one of: off, filter, error")
    if cfg.ufg_rank < 1:
        raise ValueError("ufg_rank must be >= 1")
    if cfg.ufg_qiem_regularization <= 0:
        raise ValueError("ufg_qiem_regularization must be > 0")
    if cfg.ufg_fasa_tau <= 0:
        raise ValueError("ufg_fasa_tau must be > 0")
    for name in ("ufg_spectral_gate_kernel", "ufg_deform_kernel_size"):
        value = getattr(cfg, name)
        if value < 1 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")

    if cfg.degradation_mode != "physical" and cfg.progressive_lift in (
        "adjoint",
        "normalized_adjoint",
    ):
        raise ValueError(
            "adjoint/normalized_adjoint lift is only defined for physical degradation"
        )


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="UFGNet HSI-MSI Fusion Reproduction")

    parser.add_argument("--stage", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="PaviaU")
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--output_root", type=str, default="./outputs")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--n_select_bands", type=int, default=8)

    parser.add_argument(
        "--degradation_mode",
        type=str,
        default="gaussian_bicubic",
        choices=["bicubic", "gaussian_bicubic", "physical"],
    )
    parser.add_argument("--degradation_sigma", type=float, default=2.0)
    parser.add_argument("--degradation_kernel_size", type=int, default=5)
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)

    parser.add_argument("--progressive_steps", type=int, default=12)
    parser.add_argument(
        "--progressive_lift",
        type=str,
        default="auto",
        choices=["auto", "bilinear", "nearest", "adjoint", "normalized_adjoint"],
    )
    parser.add_argument("--boundary_probability", type=float, default=0.2)
    parser.add_argument("--boundary_radius", type=int, default=1)

    parser.add_argument("--msi_mode", type=str, default="srf", choices=["uniform", "srf"])
    parser.add_argument(
        "--srf_path",
        type=str,
        default="",
        help="Optional explicit SRF CSV path. Empty uses the selected profile default.",
    )
    parser.add_argument("--wavelength_root", type=str, default="./data/wavelengths")
    parser.add_argument("--wavelength_path", type=str, default="")
    parser.add_argument("--srf_interp", type=str, default="pchip", choices=["pchip", "linear"])
    parser.add_argument(
        "--srf_band_set",
        type=str,
        default="auto",
        choices=["auto", "ikonos4", "wv2_visible5", "wv2_visible6", "wv2_all8"],
        help="auto: PaviaU->IKONOS4, Houston13/Chikusei->WV2 all8.",
    )
    parser.add_argument(
        "--srf_min_coverage_ratio",
        type=float,
        default=0.90,
        help="Minimum fraction of a full physical SRF that must lie inside the HSI spectral support.",
    )
    parser.add_argument(
        "--srf_coverage_policy",
        type=str,
        default="filter",
        choices=["off", "filter", "error"],
        help="How to handle requested SRF bands below the physical overlap threshold.",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_sam", type=float, default=1e-2)
    parser.add_argument("--lambda_freq", type=float, default=1e-2)
    parser.add_argument("--freq_gamma", type=float, default=0.5)
    parser.add_argument("--freq_eta", type=float, default=0.5)

    parser.add_argument("--ufg_rank", type=int, default=5)
    parser.add_argument("--ufg_qiem_regularization", type=float, default=1e-4)
    parser.add_argument("--ufg_fasa_tau", type=float, default=1.0)
    parser.add_argument("--ufg_spectral_gate_kernel", type=int, default=3)
    parser.add_argument("--ufg_deform_kernel_size", type=int, default=3)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_dc", type=float, default=0.1)
    parser.add_argument("--lambda_sgrad", type=float, default=0.05)
    parser.add_argument("--lambda_sdir", type=float, default=0.2)
    parser.add_argument("--lambda_ns_l1", type=float, default=1.0)
    parser.add_argument("--lambda_srf_region", type=float, default=0.3)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args(argv)
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()
    for key, value in vars(args).items():
        setattr(cfg, key, value)

    dataset_cfg = cfg.datasets.get(cfg.dataset)
    if dataset_cfg is None:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    resolve_config_defaults(cfg)
    validate_config(cfg)
    make_dirs(cfg)
    return cfg


def make_dirs(cfg: TrainConfig):
    dirs = [
        cfg.checkpoint_root,
        cfg.log_root,
        cfg.output_root,
        os.path.join(cfg.output_root, "predictions", cfg.dataset),
        os.path.join(cfg.output_root, "metrics"),
        os.path.join(cfg.output_root, "figures"),
    ]
    for path in dirs:
        os.makedirs(path, exist_ok=True)


def get_checkpoint_path(cfg: TrainConfig, stage: str = None, name: str = None):
    stage = stage or cfg.stage
    if not name:
        name = f"{cfg.dataset}_{stage}.pth"
    path = os.path.join(cfg.checkpoint_root, stage, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def print_config(cfg: TrainConfig):
    print("=" * 60)
    print("UFGNet Reproduction Config")
    print("=" * 60)
    for key, value in cfg.__dict__.items():
        if key != "datasets":
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = parse_args()
    print_config(cfg)
