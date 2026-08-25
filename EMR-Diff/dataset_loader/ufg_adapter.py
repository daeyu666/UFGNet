import os
import sys


EMR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(EMR_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import TrainConfig, get_dataset_configs, resolve_config_defaults, validate_config
from ufgnet_data import build_ufgnet_loaders


def build_ufg_loaders(configs):
    """Build EMR-Diff dataloaders from the shared UFGNet observation protocol.

    This intentionally reuses ``build_ufgnet_loaders`` rather than the legacy
    generic loader so all compared methods see the same single pre-simulated
    LR-HSI/HR-MSI pair, overlapping patches, full-scene evaluation, and the
    same PaviaU 103->93 HMIF spectral preprocessing.
    """
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

    # Current comparison protocol: 5x5 Gaussian (sigma=2) + bicubic x4.
    cfg.degradation_mode = "gaussian_bicubic"
    cfg.degradation_sigma = float(configs.data.get("gaussian_sigma", 2.0))
    cfg.degradation_kernel_size = int(configs.data.get("gaussian_kernel_size", 5))

    # Use exactly the same real-SRF policy as UFGNet. By default this resolves
    # PaviaU -> IKONOS 4-band and Houston13/Chikusei -> WV2 all8, with the same
    # full-SRF coverage protection before normalization.
    cfg.msi_mode = "srf"
    cfg.srf_band_set = str(configs.data.get("srf_band_set", "auto"))
    cfg.srf_min_coverage_ratio = float(
        configs.data.get("srf_min_coverage_ratio", 0.90)
    )
    cfg.srf_coverage_policy = str(configs.data.get("srf_coverage_policy", "filter"))

    # EMR-Diff is normally launched from its own subdirectory, so use absolute
    # repository paths for the shared SRF and wavelength files.
    cfg.wavelength_root = os.path.join(REPO_ROOT, "data", "wavelengths")
    explicit_wavelength = str(configs.data.get("wavelength_path", ""))
    if explicit_wavelength:
        cfg.wavelength_path = (
            explicit_wavelength
            if os.path.isabs(explicit_wavelength)
            else os.path.abspath(os.path.join(EMR_ROOT, explicit_wavelength))
        )
    else:
        # Leave empty so the shared UFGNet loader resolves the formal profile.
        # For PaviaU after 103->93 preprocessing this becomes the benchmark
        # nominal 430-860 nm / 93-band IKONOS mapping.
        cfg.wavelength_path = ""

    explicit_srf = str(configs.data.get("srf_path", ""))
    if explicit_srf:
        cfg.srf_path = (
            explicit_srf
            if os.path.isabs(explicit_srf)
            else os.path.abspath(os.path.join(EMR_ROOT, explicit_srf))
        )
    else:
        resolved = (
            "ikonos4"
            if cfg.srf_band_set == "auto" and cfg.dataset == "PaviaU"
            else "wv2_all8"
            if cfg.srf_band_set == "auto"
            else cfg.srf_band_set
        )
        filename = (
            "ikonos_relative_spectral_response.csv"
            if resolved == "ikonos4"
            else "wv2_relative_spectral_response_data_for_i.atcorr.csv"
        )
        cfg.srf_path = os.path.join(REPO_ROOT, "data", "srf", filename)

    cfg.batch_size = int(configs.train.get("batch", [1, 1])[0])
    cfg.num_workers = int(configs.train.get("num_workers", 0))
    cfg.device = str(configs.train.get("device", "cuda"))

    resolve_config_defaults(cfg)
    validate_config(cfg)

    train_loader, test_loader, info = build_ufgnet_loaders(cfg)
    cfg.n_select_bands = int(info["n_select_bands"])
    return train_loader, test_loader, info, cfg