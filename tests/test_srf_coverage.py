import numpy as np
import pandas as pd
import pytest

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
