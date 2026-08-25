import os

import numpy as np
import pandas as pd
import pytest

from config import TrainConfig, get_dataset_configs
from data_loader import IKONOS_4_BANDS, _resolve_srf_spec
from srf_utils import build_srf_weights, compute_srf_coverage_ratios


def _write_srf(path):
    pd.DataFrame(
        {
            "WL(nm)": [0.0, 1.0, 2.0, 3.0, 4.0],
            "inside": [0.0, 1.0, 1.0, 1.0, 0.0],
            "outside": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    ).to_csv(path, index=False)


def test_full_srf_overlap_is_measured_before_normalization(tmp_path):
    srf_path = tmp_path / "toy_srf.csv"
    _write_srf(srf_path)
    hsi_wavelengths = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    diagnostics = compute_srf_coverage_ratios(
        str(srf_path), hsi_wavelengths, ["inside", "outside"]
    )
    ratios = {item["band"]: item["coverage_ratio"] for item in diagnostics}

    assert ratios["inside"] == pytest.approx(2.75 / 3.0, rel=1e-6)
    assert ratios["outside"] == pytest.approx(0.25, rel=1e-6)


def test_filter_prevents_low_overlap_tail_from_being_renormalized(tmp_path):
    srf_path = tmp_path / "toy_srf.csv"
    _write_srf(srf_path)
    hsi_wavelengths = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    weights, names, diagnostics = build_srf_weights(
        str(srf_path),
        hsi_wavelengths,
        selected_bands=["inside", "outside"],
        interp_kind="linear",
        normalize=True,
        min_coverage_ratio=0.90,
        coverage_policy="filter",
        return_diagnostics=True,
    )

    assert names == ["inside"]
    assert weights.shape == (1, 3)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
    status = {item["band"]: item["status"] for item in diagnostics}
    assert status["inside"] == "kept"
    assert status["outside"] == "dropped_low_coverage"


def test_error_policy_refuses_low_overlap_band(tmp_path):
    srf_path = tmp_path / "toy_srf.csv"
    _write_srf(srf_path)
    hsi_wavelengths = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="refusing to renormalize"):
        build_srf_weights(
            str(srf_path),
            hsi_wavelengths,
            selected_bands=["inside", "outside"],
            interp_kind="linear",
            min_coverage_ratio=0.90,
            coverage_policy="error",
        )


def test_pavia_ikonos4_nominal_grid_keeps_all_four_physical_bands():
    wavelengths = np.loadtxt(
        "./data/wavelengths/PaviaU_nominal_430_860.txt"
    ).astype(np.float32)
    weights, names, diagnostics = build_srf_weights(
        "./data/srf/ikonos_relative_spectral_response.csv",
        wavelengths,
        selected_bands=IKONOS_4_BANDS,
        interp_kind="pchip",
        normalize=True,
        min_coverage_ratio=0.90,
        coverage_policy="error",
        return_diagnostics=True,
    )

    assert names == IKONOS_4_BANDS
    assert weights.shape == (4, 103)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    ratios = {item["band"]: item["coverage_ratio"] for item in diagnostics}
    assert ratios["IKONOS Blue"] > 0.96
    assert ratios["IKONOS Green"] > 0.98
    assert ratios["IKONOS Red"] > 0.99
    assert ratios["IKONOS NIR"] > 0.92


def test_auto_profile_resolves_pavia_to_ikonos_and_houston_to_wv2():
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()
    cfg.srf_band_set = "auto"
    cfg.srf_path = ""
    cfg.wavelength_path = ""

    cfg.dataset = "PaviaU"
    srf_path, bands, wavelengths, wavelength_path, profile = _resolve_srf_spec(
        cfg, 103
    )
    assert profile == "ikonos4"
    assert bands == IKONOS_4_BANDS
    assert srf_path.endswith("ikonos_relative_spectral_response.csv")
    assert wavelength_path.endswith("PaviaU_nominal_430_860.txt")
    assert wavelengths[0] == pytest.approx(430.0)
    assert wavelengths[-1] == pytest.approx(860.0)

    cfg.dataset = "Houston13"
    _, bands, _, wavelength_path, profile = _resolve_srf_spec(cfg, 144)
    assert profile == "wv2_all8"
    assert len(bands) == 8
    assert wavelength_path.endswith(os.path.join("data", "wavelengths", "Houston13.txt"))
