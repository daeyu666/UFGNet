"""Audit physical WV2 SRF overlap for all configured HSI wavelength grids.

This script does not require the HSI .mat files. It requests all eight WV2
bands, measures each full physical SRF's energy overlap with the HSI spectral
support, filters bands below the configured threshold before normalization, and
prints the retained MSI channels.
"""

from __future__ import annotations

import os

import numpy as np

from srf_utils import WV2_ALL_8_BANDS, build_srf_weights, print_srf_summary


SRF_PATH = "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"
WAVELENGTH_ROOT = "./data/wavelengths"
DATASETS = ("PaviaU", "Houston13", "Chikusei")
MIN_COVERAGE = 0.90


def main() -> None:
    print(
        "Physical SRF audit: request WV2 all8, retain only bands with "
        f">={100.0 * MIN_COVERAGE:.1f}% full-SRF overlap."
    )
    for dataset in DATASETS:
        path = os.path.join(WAVELENGTH_ROOT, f"{dataset}.txt")
        wavelengths = np.loadtxt(path).astype(np.float32).reshape(-1)
        weights, names, diagnostics = build_srf_weights(
            srf_path=SRF_PATH,
            hsi_wavelengths=wavelengths,
            selected_bands=WV2_ALL_8_BANDS,
            interp_kind="pchip",
            normalize=True,
            min_coverage_ratio=MIN_COVERAGE,
            coverage_policy="filter",
            return_diagnostics=True,
        )
        print(f"\n[{dataset}] retained={len(names)}/8 -> {', '.join(names)}")
        print_srf_summary(
            weights,
            names,
            wavelengths,
            coverage_diagnostics=diagnostics,
            min_coverage_ratio=MIN_COVERAGE,
            coverage_policy="filter",
        )


if __name__ == "__main__":
    main()
