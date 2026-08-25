"""Audit the dataset-aware physical SRF profiles used by the experiments.

PaviaU uses the IKONOS four-band multispectral response on the nominal public
benchmark support 430-860 nm. Houston13 and Chikusei use all eight WV2 bands.
Every requested band is checked against its *full* SRF energy before any
truncation or normalization, so a weak tail can never be promoted to a complete
MSI observation.
"""

from __future__ import annotations

import os

import numpy as np

from data_loader import IKONOS_4_BANDS
from srf_utils import WV2_ALL_8_BANDS, build_srf_weights, print_srf_summary


WV2_SRF_PATH = "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"
IKONOS_SRF_PATH = "./data/srf/ikonos_relative_spectral_response.csv"
WAVELENGTH_ROOT = "./data/wavelengths"
MIN_COVERAGE = 0.90

PROFILES = {
    "PaviaU": {
        "profile": "ikonos4",
        "srf_path": IKONOS_SRF_PATH,
        "wavelength_path": os.path.join(
            WAVELENGTH_ROOT, "PaviaU_nominal_430_860.txt"
        ),
        "bands": IKONOS_4_BANDS,
    },
    "Houston13": {
        "profile": "wv2_all8",
        "srf_path": WV2_SRF_PATH,
        "wavelength_path": os.path.join(WAVELENGTH_ROOT, "Houston13.txt"),
        "bands": WV2_ALL_8_BANDS,
    },
    "Chikusei": {
        "profile": "wv2_all8",
        "srf_path": WV2_SRF_PATH,
        "wavelength_path": os.path.join(WAVELENGTH_ROOT, "Chikusei.txt"),
        "bands": WV2_ALL_8_BANDS,
    },
}


def main() -> None:
    print(
        "Physical SRF audit: dataset-aware real sensor profiles; retain only "
        f"bands with >={100.0 * MIN_COVERAGE:.1f}% full-SRF overlap."
    )
    for dataset, spec in PROFILES.items():
        wavelengths = np.loadtxt(spec["wavelength_path"]).astype(np.float32).reshape(-1)
        weights, names, diagnostics = build_srf_weights(
            srf_path=spec["srf_path"],
            hsi_wavelengths=wavelengths,
            selected_bands=spec["bands"],
            interp_kind="pchip",
            normalize=True,
            min_coverage_ratio=MIN_COVERAGE,
            coverage_policy="filter",
            return_diagnostics=True,
        )
        print(
            f"\n[{dataset}] profile={spec['profile']} retained={len(names)}/"
            f"{len(spec['bands'])} -> {', '.join(names)}"
        )
        print_srf_summary(
            weights,
            names,
            wavelengths,
            coverage_diagnostics=diagnostics,
            min_coverage_ratio=MIN_COVERAGE,
            coverage_policy="filter",
        )

    # Keep the old 430-838 Pavia wavelength file visible as a diagnostic only.
    # Its IKONOS NIR overlap is below 90%, which is why it is no longer the
    # default spectral grid for the Pavia IKONOS benchmark simulation.
    legacy_path = os.path.join(WAVELENGTH_ROOT, "PaviaU.txt")
    if os.path.exists(legacy_path):
        legacy_wavelengths = np.loadtxt(legacy_path).astype(np.float32).reshape(-1)
        _, legacy_names, legacy_diag = build_srf_weights(
            srf_path=IKONOS_SRF_PATH,
            hsi_wavelengths=legacy_wavelengths,
            selected_bands=IKONOS_4_BANDS,
            interp_kind="pchip",
            normalize=True,
            min_coverage_ratio=MIN_COVERAGE,
            coverage_policy="filter",
            return_diagnostics=True,
        )
        nir = next(item for item in legacy_diag if item["band"] == "IKONOS NIR")
        print(
            "\n[PaviaU legacy-grid diagnostic] "
            f"{legacy_wavelengths.min():.2f}-{legacy_wavelengths.max():.2f} nm: "
            f"IKONOS NIR full-SRF overlap={100.0 * nir['coverage_ratio']:.2f}%, "
            f"retained={legacy_names}"
        )


if __name__ == "__main__":
    main()
