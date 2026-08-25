# srf_utils.py
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


WV2_VISIBLE_5_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
]

WV2_VISIBLE_6_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
]

WV2_ALL_8_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
    "WV2 NIR1",
    "WV2 NIR2",
]


def load_hsi_wavelengths(wavelength_path: str, n_bands: int) -> np.ndarray:
    """Read HSI center wavelengths and return nanometers."""
    if not os.path.exists(wavelength_path):
        raise FileNotFoundError(f"Cannot find wavelength file: {wavelength_path}")

    ext = os.path.splitext(wavelength_path)[1].lower()
    if ext == ".npy":
        wavelengths = np.load(wavelength_path).astype(np.float32)
    elif ext in [".txt", ".dat"]:
        wavelengths = np.loadtxt(wavelength_path).astype(np.float32)
    elif ext == ".csv":
        df = pd.read_csv(wavelength_path)
        lower_cols = [c.lower() for c in df.columns]
        selected_col = None
        for key in ["wavelength", "wave", "wl", "lambda", "center"]:
            for col, lower_col in zip(df.columns, lower_cols):
                if key in lower_col:
                    selected_col = col
                    break
            if selected_col is not None:
                break
        if selected_col is None:
            selected_col = df.columns[0]
        wavelengths = df[selected_col].values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported wavelength file type: {ext}")

    wavelengths = np.asarray(wavelengths).reshape(-1).astype(np.float32)
    if wavelengths.size != n_bands:
        raise ValueError(
            f"Wavelength number mismatch: got {wavelengths.size}, "
            f"but HSI has {n_bands} bands."
        )
    if np.nanmax(wavelengths) < 10:
        wavelengths = wavelengths * 1000.0
    if wavelengths.size > 1 and np.any(np.diff(wavelengths) <= 0):
        raise ValueError("HSI wavelengths must be strictly increasing")
    return wavelengths.astype(np.float32)


def estimate_band_edges(wavelengths: np.ndarray) -> np.ndarray:
    """Estimate spectral-bin edges from center wavelengths."""
    wavelengths = np.asarray(wavelengths, dtype=np.float64).reshape(-1)
    if wavelengths.size == 0:
        raise ValueError("wavelengths must not be empty")
    if wavelengths.size == 1:
        return np.array(
            [wavelengths[0] - 0.5, wavelengths[0] + 0.5], dtype=np.float64
        )
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing")

    edges = np.empty(wavelengths.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
    edges[0] = wavelengths[0] - 0.5 * (wavelengths[1] - wavelengths[0])
    edges[-1] = wavelengths[-1] + 0.5 * (wavelengths[-1] - wavelengths[-2])
    return edges


def estimate_band_widths(wavelengths: np.ndarray) -> np.ndarray:
    """Estimate integration width for every HSI center wavelength."""
    edges = estimate_band_edges(wavelengths)
    widths = np.maximum(edges[1:] - edges[:-1], 1e-6)
    return widths.astype(np.float32)


def interp_srf_to_hsi_wavelengths(
    srf_wavelengths: np.ndarray,
    response_values: np.ndarray,
    hsi_wavelengths: np.ndarray,
    interp_kind: str = "pchip",
) -> np.ndarray:
    """Resample an SRF curve at all HSI center wavelengths."""
    srf_wavelengths = np.asarray(srf_wavelengths).astype(np.float32)
    response_values = np.asarray(response_values).astype(np.float32)
    hsi_wavelengths = np.asarray(hsi_wavelengths).astype(np.float32)

    order = np.argsort(srf_wavelengths)
    srf_wavelengths = srf_wavelengths[order]
    response_values = response_values[order]

    if interp_kind == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator

            curve = PchipInterpolator(
                srf_wavelengths,
                response_values,
                extrapolate=False,
            )
            sampled_response = curve(hsi_wavelengths)
        except Exception:
            sampled_response = np.interp(
                hsi_wavelengths,
                srf_wavelengths,
                response_values,
                left=0.0,
                right=0.0,
            )
    elif interp_kind == "linear":
        sampled_response = np.interp(
            hsi_wavelengths,
            srf_wavelengths,
            response_values,
            left=0.0,
            right=0.0,
        )
    else:
        raise ValueError(f"Unsupported interp_kind: {interp_kind}")

    sampled_response = np.nan_to_num(
        sampled_response,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    sampled_response = np.maximum(sampled_response, 0.0)
    return sampled_response.astype(np.float32)


def _integrate_piecewise_linear(
    wavelengths: np.ndarray,
    response: np.ndarray,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> float:
    """Integrate a non-negative sampled SRF, inserting exact range boundaries."""
    x = np.asarray(wavelengths, dtype=np.float64).reshape(-1)
    y = np.asarray(response, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("SRF wavelength/response arrays must have equal size >= 2")

    order = np.argsort(x)
    x = x[order]
    y = np.maximum(np.nan_to_num(y[order], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if np.any(np.diff(x) <= 0):
        raise ValueError("SRF wavelengths must be strictly increasing")

    lo = float(x[0] if lower is None else max(lower, x[0]))
    hi = float(x[-1] if upper is None else min(upper, x[-1]))
    if hi <= lo:
        return 0.0

    inside = (x > lo) & (x < hi)
    xi = np.concatenate(([lo], x[inside], [hi]))
    yi = np.interp(xi, x, y)
    return float(np.trapz(yi, xi))


def compute_srf_coverage_ratios(
    srf_path: str,
    hsi_wavelengths: np.ndarray,
    selected_bands=None,
    eps: float = 1e-12,
) -> List[Dict[str, float]]:
    """Measure how much of each *full physical SRF energy* the HSI can support.

    The HSI support is derived from spectral-bin edges rather than only the first
    and last center wavelengths. This check is performed on the original SRF
    curve before any truncation or per-band normalization, preventing a tiny SRF
    tail inside the HSI range from being amplified into a synthetic full-strength
    MSI channel.
    """
    if not os.path.exists(srf_path):
        raise FileNotFoundError(f"Cannot find SRF file: {srf_path}")

    df = pd.read_csv(srf_path)
    if "WL(nm)" not in df.columns:
        raise ValueError("SRF file must contain column: WL(nm)")
    if selected_bands is None:
        selected_bands = WV2_VISIBLE_5_BANDS

    srf_wavelengths = df["WL(nm)"].values.astype(np.float64)
    hsi_wavelengths = np.asarray(hsi_wavelengths, dtype=np.float64).reshape(-1)
    hsi_edges = estimate_band_edges(hsi_wavelengths)
    hsi_lo, hsi_hi = float(hsi_edges[0]), float(hsi_edges[-1])

    diagnostics: List[Dict[str, float]] = []
    for band in selected_bands:
        if band not in df.columns:
            raise ValueError(f"SRF file does not contain band column: {band}")
        response = df[band].values.astype(np.float64)
        full_energy = _integrate_piecewise_linear(srf_wavelengths, response)
        overlap_energy = _integrate_piecewise_linear(
            srf_wavelengths,
            response,
            lower=hsi_lo,
            upper=hsi_hi,
        )
        ratio = overlap_energy / max(full_energy, eps)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        diagnostics.append(
            {
                "band": band,
                "coverage_ratio": ratio,
                "full_energy": float(full_energy),
                "overlap_energy": float(overlap_energy),
                "hsi_support_min_nm": hsi_lo,
                "hsi_support_max_nm": hsi_hi,
            }
        )
    return diagnostics


def build_srf_weights(
    srf_path: str,
    hsi_wavelengths: np.ndarray,
    selected_bands=None,
    interp_kind: str = "pchip",
    normalize: bool = True,
    eps: float = 1e-12,
    min_coverage_ratio: float = 0.0,
    coverage_policy: str = "off",
    return_diagnostics: bool = False,
):
    """Build discrete SRF weights with optional physical coverage protection.

    coverage_policy:
        off: legacy behavior; no physical-overlap rejection.
        filter: drop bands whose full-SRF overlap ratio is below the threshold.
        error: reject the configuration if any requested band is below threshold.

    Coverage is evaluated before truncation/normalization. Under ``filter`` or
    ``error``, a low-overlap SRF tail is never normalized into a full MSI band.
    """
    if not 0.0 <= min_coverage_ratio <= 1.0:
        raise ValueError("min_coverage_ratio must lie in [0, 1]")
    if coverage_policy not in {"off", "filter", "error"}:
        raise ValueError("coverage_policy must be one of: off, filter, error")
    if not os.path.exists(srf_path):
        raise FileNotFoundError(f"Cannot find SRF file: {srf_path}")

    df = pd.read_csv(srf_path)
    if "WL(nm)" not in df.columns:
        raise ValueError("SRF file must contain column: WL(nm)")
    if selected_bands is None:
        selected_bands = WV2_VISIBLE_5_BANDS

    srf_wavelengths = df["WL(nm)"].values.astype(np.float32)
    hsi_wavelengths = np.asarray(hsi_wavelengths).astype(np.float32).reshape(-1)
    hsi_widths = estimate_band_widths(hsi_wavelengths)
    coverage = compute_srf_coverage_ratios(
        srf_path=srf_path,
        hsi_wavelengths=hsi_wavelengths,
        selected_bands=selected_bands,
        eps=eps,
    )
    coverage_by_band = {item["band"]: dict(item) for item in coverage}

    all_weights = []
    band_names = []
    diagnostics = []

    for band in selected_bands:
        item = coverage_by_band[band]
        ratio = float(item["coverage_ratio"])
        below_threshold = ratio + eps < min_coverage_ratio

        if coverage_policy == "error" and below_threshold:
            raise ValueError(
                f"SRF band {band} coverage={ratio:.4f} is below "
                f"min_coverage_ratio={min_coverage_ratio:.4f}; refusing to "
                "renormalize a truncated physical SRF."
            )
        if coverage_policy == "filter" and below_threshold:
            item["status"] = "dropped_low_coverage"
            diagnostics.append(item)
            continue

        response_values = df[band].values.astype(np.float32)
        sampled_response = interp_srf_to_hsi_wavelengths(
            srf_wavelengths=srf_wavelengths,
            response_values=response_values,
            hsi_wavelengths=hsi_wavelengths,
            interp_kind=interp_kind,
        )
        raw_weight = sampled_response * hsi_widths
        weight_sum = float(np.sum(raw_weight))
        if weight_sum < eps:
            raise ValueError(
                f"SRF band {band} has no valid overlap with HSI wavelengths. "
                f"HSI center range: {hsi_wavelengths.min():.2f}-"
                f"{hsi_wavelengths.max():.2f} nm."
            )

        if normalize:
            weight = raw_weight / (weight_sum + eps)
        else:
            weight = raw_weight
        all_weights.append(weight.astype(np.float32))
        band_names.append(band)
        item["status"] = (
            "kept_policy_off"
            if coverage_policy == "off" and below_threshold
            else "kept"
        )
        item["discrete_weight_sum_before_normalization"] = weight_sum
        diagnostics.append(item)

    if not all_weights:
        ratios = ", ".join(
            f"{item['band']}={item['coverage_ratio']:.3f}" for item in diagnostics
        )
        raise ValueError(
            "No SRF bands remain after physical coverage filtering. "
            f"threshold={min_coverage_ratio:.3f}; {ratios}"
        )

    weights = np.stack(all_weights, axis=0).astype(np.float32)
    if return_diagnostics:
        return weights, band_names, diagnostics
    return weights, band_names


def hsi_to_msi_numpy(
    hsi: np.ndarray,
    srf_weights: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """Apply MxC SRF weights to an HxWxC HSI."""
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be HxWxC, but got shape: {hsi.shape}")
    if srf_weights.ndim != 2:
        raise ValueError(f"srf_weights must be MxC, but got shape: {srf_weights.shape}")

    _, _, c = hsi.shape
    _, c2 = srf_weights.shape
    if c != c2:
        raise ValueError(
            f"Band mismatch: HSI has {c} bands, but SRF weights expect {c2} bands."
        )

    msi = np.tensordot(hsi, srf_weights.T, axes=([2], [0]))
    msi = np.asarray(msi, dtype=np.float32)
    if clip:
        msi = np.clip(msi, 0.0, 1.0)
    return msi


def print_srf_summary(
    srf_weights: np.ndarray,
    band_names,
    hsi_wavelengths: np.ndarray,
    coverage_diagnostics=None,
    min_coverage_ratio: Optional[float] = None,
    coverage_policy: Optional[str] = None,
):
    """Print physical SRF overlap decisions and retained discrete weights."""
    print("=" * 88)
    print("SRF physical coverage + discrete-weight summary")
    print("=" * 88)
    edges = estimate_band_edges(hsi_wavelengths)
    print(
        f"HSI center range: {float(np.min(hsi_wavelengths)):.2f} - "
        f"{float(np.max(hsi_wavelengths)):.2f} nm | "
        f"spectral support: {edges[0]:.2f} - {edges[-1]:.2f} nm"
    )
    if min_coverage_ratio is not None:
        print(
            f"coverage policy={coverage_policy or 'off'} | "
            f"minimum full-SRF overlap={100.0 * min_coverage_ratio:.1f}%"
        )

    if coverage_diagnostics:
        print("Physical full-SRF overlap before normalization:")
        for item in coverage_diagnostics:
            status = str(item.get("status", "unknown")).upper()
            print(
                f"  {item['band']}: overlap={100.0 * item['coverage_ratio']:.2f}% "
                f"-> {status}"
            )

    print("Retained MSI bands after coverage check:")
    for i, band in enumerate(band_names):
        weight = srf_weights[i]
        peak_idx = int(np.argmax(weight))
        peak_wl = float(hsi_wavelengths[peak_idx])
        nonzero = weight > weight.max() * 0.01
        if np.any(nonzero):
            wl_min = float(hsi_wavelengths[nonzero].min())
            wl_max = float(hsi_wavelengths[nonzero].max())
        else:
            wl_min = peak_wl
            wl_max = peak_wl
        print(
            f"  {band}: peak={peak_wl:.2f} nm, "
            f"main_range={wl_min:.2f}-{wl_max:.2f} nm, "
            f"weight_sum={float(weight.sum()):.6f}, "
            f"max_weight={float(weight.max()):.6f}"
        )
    print("=" * 88)
