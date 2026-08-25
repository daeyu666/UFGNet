"""UFGNet-specific single-scene data protocol.

The paper states that LR-HSI / HR-MSI are simulated once from each scene,
overlapping patches are then sampled from that paired observation without a
separate validation split, and reported metrics come from full-image evaluation.

This module keeps the repository's user-fixed 4x Gaussian+Bicubic degradation
and SRF observation model, while making the sampling protocol faithful to that
setup. For the standard 103-band Pavia University cube, the UFGNet fusion
benchmark follows the established 93-band HMIF convention used by HySure and
many later Pavia HSI-MSI fusion papers: the first 10 bands are removed before
simulating the LR-HSI/HR-MSI pair. Requested physical SRF bands are filtered
before normalization when their full-SRF overlap with the HSI spectral support
is below the configured threshold.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_loader import (
    _build_srf,
    build_hsi_degradation,
    build_progressive_degradation,
    crop_to_scale,
    hsi_to_tensor,
    normalize_hsi,
    read_hsi_mat,
)
from srf_utils import hsi_to_msi_numpy


PAVIA_STANDARD_BANDS = 103
PAVIA_FUSION_BANDS = 93
PAVIA_DROP_LEADING_BANDS = PAVIA_STANDARD_BANDS - PAVIA_FUSION_BANDS


def _apply_ufgnet_spectral_protocol(
    img: np.ndarray,
    dataset_name: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the spectral protocol used by the UFGNet Pavia fusion benchmark.

    The standard downloadable PaviaU cube contains 103 valid channels. A large
    body of HSI-MSI fusion work evaluates Pavia on a 93-band cube. HySure's
    executable Pavia benchmark makes this conversion explicitly with
    ``Z = Z(11:end,:)`` after matrix conversion, i.e. it removes the first ten
    channels. We use that exact, reproducible convention here rather than
    inventing a new band selector.

    A preprocessed 93-band Pavia cube is accepted unchanged. Other Pavia band
    counts are rejected so that a silent spectral remapping cannot enter a
    formal experiment.
    """
    img = np.asarray(img)
    original_bands = int(img.shape[2])
    metadata: Dict[str, object] = {
        "spectral_protocol": "native",
        "original_bands": original_bands,
        "retained_bands": original_bands,
        "dropped_band_indices_1based": [],
    }

    if dataset_name != "PaviaU":
        return img, metadata

    if original_bands == PAVIA_STANDARD_BANDS:
        img = np.ascontiguousarray(img[:, :, PAVIA_DROP_LEADING_BANDS:])
        metadata.update(
            {
                "spectral_protocol": "paviau_hmif_103_to_93_drop_first10",
                "retained_bands": PAVIA_FUSION_BANDS,
                "dropped_band_indices_1based": list(
                    range(1, PAVIA_DROP_LEADING_BANDS + 1)
                ),
            }
        )
        return img, metadata

    if original_bands == PAVIA_FUSION_BANDS:
        metadata.update(
            {
                "spectral_protocol": "paviau_hmif_93_preprocessed",
                "retained_bands": PAVIA_FUSION_BANDS,
            }
        )
        return img, metadata

    raise ValueError(
        "Formal PaviaU UFGNet fusion protocol expects either the standard "
        f"{PAVIA_STANDARD_BANDS}-band cube or an already-preprocessed "
        f"{PAVIA_FUSION_BANDS}-band cube, but got {original_bands} bands."
    )


def _axis_positions(length: int, patch: int, stride: int, align: int) -> List[int]:
    if patch > length:
        raise ValueError(f"patch_size={patch} exceeds scene dimension={length}")
    if patch % align != 0 or stride % align != 0 or length % align != 0:
        raise ValueError(
            "For paired HR/LR patch sampling, scene size, patch_size and stride "
            f"must be divisible by scale_ratio={align}."
        )

    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if not positions:
        positions = [0]
    elif positions[-1] != last:
        positions.append(last)
    return positions


def _paired_augment(
    gt: torch.Tensor,
    hr_msi: torch.Tensor,
    lr_hsi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Apply exactly the same geometry to all three paired tensors.
    if random.random() < 0.5:
        gt = torch.flip(gt, dims=(-2,))
        hr_msi = torch.flip(hr_msi, dims=(-2,))
        lr_hsi = torch.flip(lr_hsi, dims=(-2,))
    if random.random() < 0.5:
        gt = torch.flip(gt, dims=(-1,))
        hr_msi = torch.flip(hr_msi, dims=(-1,))
        lr_hsi = torch.flip(lr_hsi, dims=(-1,))
    if random.random() < 0.5:
        k = random.randint(1, 3)
        gt = torch.rot90(gt, k=k, dims=(-2, -1))
        hr_msi = torch.rot90(hr_msi, k=k, dims=(-2, -1))
        lr_hsi = torch.rot90(lr_hsi, k=k, dims=(-2, -1))
    return gt.contiguous(), hr_msi.contiguous(), lr_hsi.contiguous()


class UFGNetPatchDataset(Dataset):
    """Overlapping aligned patches sampled from one pre-simulated scene pair."""

    def __init__(
        self,
        gt_full: torch.Tensor,
        hr_msi_full: torch.Tensor,
        lr_hsi_full: torch.Tensor,
        patch_size: int,
        stride: int,
        scale_ratio: int,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.gt_full = gt_full.contiguous().float()
        self.hr_msi_full = hr_msi_full.contiguous().float()
        self.lr_hsi_full = lr_hsi_full.contiguous().float()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.scale_ratio = int(scale_ratio)
        self.augment = bool(augment)

        _, h, w = self.gt_full.shape
        ys = _axis_positions(h, self.patch_size, self.stride, self.scale_ratio)
        xs = _axis_positions(w, self.patch_size, self.stride, self.scale_ratio)
        self.coords = [(top, left) for top in ys for left in xs]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left = self.coords[index]
        p = self.patch_size
        s = self.scale_ratio

        gt = self.gt_full[:, top : top + p, left : left + p]
        hr_msi = self.hr_msi_full[:, top : top + p, left : left + p]

        lt, ll = top // s, left // s
        lp = p // s
        lr_hsi = self.lr_hsi_full[:, lt : lt + lp, ll : ll + lp]

        if self.augment:
            gt, hr_msi, lr_hsi = _paired_augment(gt, hr_msi, lr_hsi)

        return {
            "lr_hsi": lr_hsi.contiguous().float(),
            "hr_msi": hr_msi.contiguous().float(),
            "gt": gt.contiguous().float(),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[0], dtype=torch.long),
        }


class UFGNetFullSceneDataset(Dataset):
    """Single full-scene sample used only for final/monitoring evaluation."""

    def __init__(
        self,
        gt_full: torch.Tensor,
        hr_msi_full: torch.Tensor,
        lr_hsi_full: torch.Tensor,
    ) -> None:
        super().__init__()
        self.gt = gt_full.contiguous().float()
        self.hr_msi = hr_msi_full.contiguous().float()
        self.lr_hsi = lr_hsi_full.contiguous().float()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return {
            "lr_hsi": self.lr_hsi,
            "hr_msi": self.hr_msi,
            "gt": self.gt,
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(self.gt.shape[0], dtype=torch.long),
        }


def build_ufgnet_datasets(cfg):
    dataset_cfg = cfg.datasets[cfg.dataset]
    file_path = f"{cfg.data_root}/{dataset_cfg.file_name}"

    img = read_hsi_mat(file_path, dataset_cfg.mat_keys)
    img, spectral_metadata = _apply_ufgnet_spectral_protocol(img, cfg.dataset)
    img = normalize_hsi(img)
    img = crop_to_scale(img, cfg.scale_ratio)
    n_bands = img.shape[2]

    if spectral_metadata["spectral_protocol"] != "native":
        print(
            "PaviaU spectral protocol: "
            f"{spectral_metadata['original_bands']} -> "
            f"{spectral_metadata['retained_bands']} bands; "
            f"mode={spectral_metadata['spectral_protocol']}"
        )
    print(f"Loaded {cfg.dataset}: shape={img.shape}, bands={n_bands}")
    (
        srf_weights,
        srf_band_names,
        hsi_wavelengths,
        n_select_bands,
        coverage_diagnostics,
    ) = _build_srf(cfg, n_bands)
    if srf_weights is None:
        raise RuntimeError("UFGNet reproduction requires msi_mode='srf'.")

    degradation_operator = build_hsi_degradation(cfg)
    progressive_degradation = build_progressive_degradation(
        cfg, degradation_operator=degradation_operator
    )

    # Simulate the observation pair ONCE from the full scene, then patch it.
    gt_full = hsi_to_tensor(img)
    with torch.no_grad():
        lr_hsi_full = degradation_operator.degrade(gt_full.unsqueeze(0)).squeeze(0)
    hr_msi_full = hsi_to_tensor(hsi_to_msi_numpy(img, srf_weights))

    train_set = UFGNetPatchDataset(
        gt_full=gt_full,
        hr_msi_full=hr_msi_full,
        lr_hsi_full=lr_hsi_full,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        scale_ratio=cfg.scale_ratio,
        augment=True,
    )
    test_set = UFGNetFullSceneDataset(
        gt_full=gt_full,
        hr_msi_full=hr_msi_full,
        lr_hsi_full=lr_hsi_full,
    )

    info = {
        "dataset": cfg.dataset,
        "n_bands": n_bands,
        "n_select_bands": n_select_bands,
        "scale_ratio": cfg.scale_ratio,
        "train_samples": len(train_set),
        "test_samples": 1,
        "full_scene_shape": tuple(gt_full.shape),
        "degradation_mode": degradation_operator.mode,
        "progressive_steps": progressive_degradation.total_steps,
        "progressive_lift": progressive_degradation.default_lift_mode,
        "degradation_operator": degradation_operator,
        "progressive_degradation": progressive_degradation,
        "msi_mode": getattr(cfg, "msi_mode", "uniform"),
        "srf_weights": srf_weights,
        "srf_band_names": srf_band_names,
        "srf_coverage_diagnostics": coverage_diagnostics,
        "hsi_wavelengths": hsi_wavelengths,
        "sampling_protocol": "single_scene_predegraded_overlapping_patches",
        "spectral_protocol": spectral_metadata["spectral_protocol"],
        "original_n_bands": spectral_metadata["original_bands"],
        "dropped_band_indices_1based": spectral_metadata[
            "dropped_band_indices_1based"
        ],
    }
    print(
        "UFGNet sampling protocol: full scene degraded once -> "
        f"{len(train_set)} overlapping train patches; full-scene evaluation."
    )
    return train_set, test_set, info


def build_ufgnet_loaders(cfg):
    train_set, test_set, info = build_ufgnet_datasets(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader, info