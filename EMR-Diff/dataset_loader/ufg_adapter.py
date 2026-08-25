import os
import sys


EMR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(EMR_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import TrainConfig, get_dataset_configs, resolve_config_defaults, validate_config
from data_loader import build_loaders


def build_ufg_loaders(configs):
    """Build EMR-Diff dataloaders from the exact UFGNet experiment protocol."""
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()

    cfg.dataset = str(configs.data.get("dataset", "PaviaU"))
    cfg.data_root = str(configs.data.get("data_root", os.path.join(REPO_ROOT, "data", "raw")))
    if not os.path.isabs(cfg.data_root):
        cfg.data_root = os.path.abspath(os.path.join(EMR_ROOT, cfg.data_root))

    cfg.image_size = int(configs.data.get("test_size", 128))
    cfg.patch_size = int(configs.data.get("patch_size", 64))
    cfg.stride = int(configs.data.get("stride", 32))
    cfg.scale_ratio = int(configs.diffusion.params.get("sf", 4))
    cfg.n_select_bands = int(configs.data.get("msi_bands", 8))

    # Current comparison protocol: 5x5 Gaussian (sigma=2) + bicubic x4.
    cfg.degradation_mode = "gaussian_bicubic"
    cfg.degradation_sigma = float(configs.data.get("gaussian_sigma", 2.0))
    cfg.degradation_kernel_size = int(configs.data.get("gaussian_kernel_size", 5))

    # All three comparison datasets use 8 uniformly selected HR-MSI bands.
    cfg.msi_mode = "uniform"

    cfg.batch_size = int(configs.train.get("batch", [1, 1])[0])
    cfg.num_workers = int(configs.train.get("num_workers", 0))
    cfg.device = str(configs.train.get("device", "cuda"))

    resolve_config_defaults(cfg)
    validate_config(cfg)

    train_loader, test_loader, info = build_loaders(cfg)
    return train_loader, test_loader, info, cfg
