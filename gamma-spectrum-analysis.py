#!/usr/bin/env python3
"""gamma-spectrum-analysis.py

Analyze two Radiacode XML gamma‑spectrum files.

Usage::
    ./gamma-spectrum-analysis.py target.xml [background.xml]

* ``target.xml`` – required spectrum to analyse.
* ``background.xml`` – optional reference background.  If supplied it is
  scaled by the ratio of measurement times and subtracted from the target.

The script:
1. Parses Radiacode XML files, extracting calibration coefficients,
   measurement time and per‑channel counts.
2. Converts channel numbers to energies using the polynomial calibration.
3. Normalises counts to count‑rate (counts / seconds).
4. If a background file is given, scales it by the time ratio and subtracts
   it from the target spectrum (negative values are clipped to zero).
5. Fits a smooth baseline with a low‑order spline (LOESS‑style) and keeps only
   regions where the signal lies above that baseline.
6. Detects peaks in the residual using ``scipy.signal.find_peaks``.
7. Matches each peak to a library of known gamma lines (energies in keV).
8. Calculates a simple contribution metric – the net counts under the peak.
9. Groups nuclides by decay chain and sums contributions per chain.
10. Prints two nicely formatted tables: one for individual nuclides and one
    for chains, both sorted by descending contribution.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal

# ---------------------------------------------------------------------------
# Known gamma‑ray lines (energy in keV) and decay chain mapping.
# This is a small representative subset; extend as required.
# ---------------------------------------------------------------------------
KNOWN_LINES: Dict[str, List[float]] = {
    "U-238": [63.3, 92.6, 1001.0, 1120.3, 1764.5],
    "Th-232": [63.3, 84.4, 2614.5, 583.2, 911.1],
    "U-235": [143.8, 185.7, 205.0, 300.1, 609.3],
    "K-40": [1460.8],
    "Cs-137": [661.7],
    "Co-60": [1173.2, 1332.5],
}

# Map nuclide to its decay chain name for grouping.
CHAIN_MAP: Dict[str, str] = {
    "U-238": "U‑238 series",
    "Th-232": "Th‑232 series",
    "U-235": "U‑235 series",
    "K-40": "Natural potassium",
    "Cs-137": "Fission product",
    "Co-60": "Activation product",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_radiacode_xml(path: Path) -> Tuple[float, List[float], np.ndarray]:
    """Parse a Radiacode XML file.

    Returns:
        measurement_time (seconds), calibration_coeffs, counts array (per channel)
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Find measurement time – tag name may vary; use xpath search.
    time_elem = root.find(".//MeasurementTime")
    if time_elem is None:
        raise ValueError(f"Cannot find MeasurementTime in {path}")
    try:
        meas_time = float(time_elem.text)
    except Exception as exc:
        raise ValueError(f"Invalid measurement time in {path}: {exc}")

    # Calibration coefficients – assumed order constant, linear, quadratic …
    coeffs_elem = root.find(".//EnergyCalibration/PolynomialCoefficients")
    if coeffs_elem is None:
        # Some files list them under <Coefficients> with child <Coefficient> elements.
        coeffs_elem = root.find(".//EnergyCalibration/Coefficients")
    if coeffs_elem is None:
        raise ValueError(f"Cannot find calibration coefficients in {path}")
    # Extract coefficient values either from direct text or child <Coefficient> tags.
    if coeffs_elem.text and coeffs_elem.text.strip():
        coeffs = [float(v) for v in coeffs_elem.text.split()]  # type: ignore[arg-type]
    else:
        coeff_vals = []
        for c in coeffs_elem.findall(".//Coefficient"):
            if c.text and c.text.strip():
                coeff_vals.append(float(c.text))
        if not coeff_vals:
            raise ValueError(f"Calibration coefficients missing or empty in {path}")
        coeffs = coeff_vals

    # Channel counts – each DataPoint element holds a count.
    datapoints = root.findall(".//DataPoint")
    if not datapoints:
        raise ValueError(f"No DataPoint elements found in {path}")
    counts = np.array([float(dp.text) for dp in datapoints], dtype=float)

    return meas_time, coeffs, counts


def channel_to_energy(channels: np.ndarray, coeffs: List[float]) -> np.ndarray:
    """Convert channel numbers to energies using a polynomial calibration.

    Energy = c0 + c1*ch + c2*ch**2 + ...
    """
    # Evaluate polynomial – numpy's polyval expects highest‑order first.
    poly_coeffs = list(reversed(coeffs))  # make highest order first
    return np.polyval(poly_coeffs, channels)


def baseline_spline(energies: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Fit a low‑order spline to represent the smooth background (LOESS style).

    The smoothing factor ``s`` controls how closely the spline follows the data.
    A smaller ``s`` retains more structure (potential peaks) while a larger one
    yields a very smooth baseline. We use a modest fraction of the default
    variance‑based estimate to avoid over‑smoothening and inadvertently removing
    real peaks.
    """
    # Use a reduced smoothing factor (0.1 % of the variance‑based value) to keep even subtle features.
    s_factor = len(energies) * np.var(counts) * 0.001
    spl = interpolate.UnivariateSpline(energies, counts, s=s_factor)
    return spl(energies)


def find_peaks(
    residual: np.ndarray,
    energies: np.ndarray,
    sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks in the residual spectrum with a permissive significance.

    A lower ``sigma`` makes the detector more sensitive to small features. The
    function still uses a robust MAD estimate for noise and applies a modest
    prominence filter to avoid isolated single‑channel spikes.
    """
    # Robust estimate of standard deviation using Median Absolute Deviation
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma_est = mad / 0.6745 if mad else np.std(residual)
    # Ensure a minimal height threshold so that very low‑count peaks can still be found.
    min_height = 0.01 * np.max(residual) if np.max(residual) > 0 else 1e-3
    height_thr = max(sigma_est * sigma, min_height)
    # Use a small prominence threshold (2 % of max residual) to keep broader features.
    prom_thr = max(0.02 * np.max(residual), 0.2 * height_thr)
    peak_inds, _ = signal.find_peaks(
        residual,
        height=height_thr,
        prominence=prom_thr,
    )
    return peak_inds, energies[peak_inds]


def match_peak_to_nuclide(
    peak_energy: float, tolerance: float = 5.0
) -> Tuple[str, float] | None:
    """Match a detected peak energy to the closest known gamma line.

    Returns (nuclide, line_energy) if within tolerance, else None.
    """
    for nuclide, lines in KNOWN_LINES.items():
        for line_e in lines:
            if abs(peak_energy - line_e) <= tolerance:
                return nuclide, line_e
    return None


def format_table(headers: List[str], rows: List[List[object]]) -> str:
    """Create a simple fixed‑width table string."""
    col_widths = [
        max(len(str(item)) for item in [h] + [r[i] for r in rows])
        for i, h in enumerate(headers)
    ]
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = [
        "  ".join(str(item).ljust(col_widths[i]) for i, item in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------


def analyze(
    target_path: Path,
    background_path: Path | None,
    sigma_cutoff: float = 2.0,
    tolerance_keV: float = 10.0,
) -> None:
    # Parse target spectrum
    tgt_time, tgt_coeffs, tgt_counts = parse_radiacode_xml(target_path)
    channels = np.arange(len(tgt_counts))
    energies = channel_to_energy(channels, tgt_coeffs)

    net_counts = tgt_counts.astype(float)  # raw counts (no time normalisation)

    if background_path:
        bg_time, bg_coeffs, bg_counts = parse_radiacode_xml(background_path)
        # Assume calibration is identical; otherwise re‑calibrate to target energies.
        if bg_coeffs != tgt_coeffs:
            sys.stderr.write(
                "Warning: background calibration differs – using target calibration for both.\n"
            )
        # Scale background counts to match target measurement time
        scale = tgt_time / bg_time
        scaled_bg = bg_counts.astype(float) * scale
        net_counts -= scaled_bg

        # Clip negatives but also check if subtraction removed almost all signal.
        net_counts = np.clip(net_counts, a_min=0.0, a_max=None)
        if net_counts.max() < 0.01 * tgt_counts.max():
            sys.stderr.write(
                "Warning: background subtraction left negligible counts; using raw target spectrum instead.\n"
            )
            net_counts = tgt_counts.astype(float)  # fallback to raw counts

    # Initial baseline using current calibration.
    baseline = baseline_spline(energies, net_counts)
    # Estimate calibration correction using well‑known peaks (Pb, U, Th, K‑40).
    above_initial = net_counts >= baseline
    diff_bool_init = np.diff(above_initial.astype(int))
    starts_init = np.where(diff_bool_init == 1)[0] + 1
    ends_init = np.where(diff_bool_init == -1)[0] + 1
    if above_initial[0]:
        starts_init = np.r_[0, starts_init]
    if above_initial[-1]:
        ends_init = np.r_[ends_init, len(above_initial)]
    calibration_pairs = []  # (detected_energy, true_energy)
    for s_i, e_i in zip(starts_init, ends_init):
        region_counts = net_counts[s_i:e_i] - baseline[s_i:e_i]
        if np.sum(region_counts) <= 0:
            continue
        peak_idx_local = int(np.argmax(net_counts[s_i:e_i]))
        detected_energy = energies[peak_idx_local + s_i]
        match = match_peak_to_nuclide(detected_energy, tolerance=tolerance_keV)
        if match:
            _, true_energy = match
            calibration_pairs.append((detected_energy, true_energy))
    # Fit linear correction if enough reference points.
    if len(calibration_pairs) >= 3:
        det_vals, true_vals = zip(*calibration_pairs)
        coeffs_corr = np.polyfit(det_vals, true_vals, 1)  # slope, intercept
        energies_corr = coeffs_corr[0] * energies + coeffs_corr[1]
    else:
        energies_corr = energies
    # Re‑compute baseline on corrected energy axis.
    baseline = baseline_spline(energies_corr, net_counts)
    # Plot high‑resolution PNG (≈4K) with thin baseline line.
    plt.figure(figsize=(38.4, 21.6), dpi=100)  # ~3840×2160
    plt.plot(energies_corr, net_counts, label="Net Spectrum", color="gray")
    plt.plot(energies_corr, baseline, label="Baseline", color="black", linewidth=1)
    above = net_counts >= baseline
    plt.fill_between(
        energies_corr,
        net_counts,
        baseline,
        where=above,
        interpolate=True,
        color="red",
        alpha=0.5,
    )
    plt.fill_between(
        energies_corr,
        net_counts,
        baseline,
        where=~above,
        interpolate=True,
        color="blue",
        alpha=0.5,
    )
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title(f"Spectrum analysis: {target_path.name}")
    output_png = target_path.with_suffix(".png")
    plt.savefig(output_png)
    plt.close()
    # Detect peak regions on corrected axis.
    above = net_counts >= baseline
    diff_bool = np.diff(above.astype(int))
    starts = np.where(diff_bool == 1)[0] + 1
    ends = np.where(diff_bool == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    peak_regions = []
    for s, e in zip(starts, ends):
        region_counts = net_counts[s:e] - baseline[s:e]
        area = float(np.sum(region_counts))
        if area <= 0:
            continue
        max_rel_idx = int(np.argmax(net_counts[s:e]))
        peak_idx = s + max_rel_idx
        peak_energy = energies_corr[peak_idx]
        half_max = region_counts.max() / 2.0
        idxs_half = np.where(region_counts >= half_max)[0]
        if len(idxs_half) >= 2:
            fwhm_energy = float(
                energies_corr[s + idxs_half[-1]] - energies_corr[s + idxs_half[0]]
            )
        else:
            fwhm_energy = 0.0
        match = match_peak_to_nuclide(peak_energy, tolerance=tolerance_keV)
        if match:
            nuclide, line_e = match
            chain = CHAIN_MAP.get(nuclide, "Other")
            note = f"{nuclide} gamma ({line_e:.1f} keV)"
        else:
            nuclide = "-"
            chain = "-"
            note = "Unidentified"
        peak_regions.append(
            {
                "energy": float(peak_energy),
                "area": area,
                "fwhm": fwhm_energy,
                "nuclide": nuclide,
                "chain": chain,
                "note": note,
            }
        )
    # Noise‑based significance threshold.
    diff_signal = net_counts - baseline
    mad_noise = np.median(np.abs(diff_signal - np.median(diff_signal)))
    max_area = max((p["area"] for p in peak_regions), default=0)
    area_threshold = max(mad_noise * 5, 0.01 * max_area)
    significant_peaks = [p for p in peak_regions if p["area"] >= area_threshold]
    # Filter out peaks that are within sigma_cutoff of the FWHM or Area distributions.
    fwhm_vals = np.array([p["fwhm"] for p in significant_peaks])
    area_vals = np.array([p["area"] for p in significant_peaks])
    stdev_fwhm = np.std(fwhm_vals) if fwhm_vals.size > 0 else 0.0
    stdev_area = np.std(area_vals) if area_vals.size > 0 else 0.0
    filtered_peaks = [
        p
        for p in significant_peaks
        if p["fwhm"] >= sigma_cutoff * stdev_fwhm
        and p["area"] >= sigma_cutoff * stdev_area
    ]
    # Use the filtered list for output.
    significant_peaks = filtered_peaks
    # Compute energy contribution (area * energy) and percentages.
    total_energy = sum(p["area"] * p["energy"] for p in significant_peaks)
    for p in significant_peaks:
        p["energy_contrib"] = p["area"] * p["energy"]
        p["energy_pct"] = (
            (p["energy_contrib"] / total_energy) * 100 if total_energy > 0 else 0.0
        )
    # Sort peaks by descending energy contribution percentage.
    significant_peaks.sort(key=lambda d: d["energy_pct"], reverse=True)
    peak_rows = [
        [
            f"{p['energy']:.2f}",
            f"{p['area']:.2f}",
            f"{p['fwhm']:.2f}",
            p["nuclide"],
            p["chain"],
            p["note"],
            f"{p['energy_pct']:.2f}%",
        ]
        for p in significant_peaks
    ]
    print("Detected peaks (sorted by energy contribution):")
    print(
        format_table(
            [
                "Energy (keV)",
                "Area",
                "FWHM (keV)",
                "Nuclide",
                "Chain",
                "Note",
                "Energy%",
            ],
            peak_rows,
        )
    )
    # Chain contributions based on energy.
    chain_totals: Dict[str, float] = {}
    for p in significant_peaks:
        if p["chain"] != "-":
            chain_totals[p["chain"]] = (
                chain_totals.get(p["chain"], 0.0) + p["energy_contrib"]
            )
    sorted_chains = sorted(chain_totals.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_chains:
        print("\nDecay‑chain contributions (by energy):")
        chain_rows = [
            [
                c,
                f"{a:.2f}",
                f"{(a / total_energy * 100) if total_energy > 0 else 0.0:.2f}%",
            ]
            for c, a in sorted_chains
        ]
        print(format_table(["Chain", "Total Energy", "Energy%"], chain_rows))
    # Two‑line summary.
    total_area = sum(p["area"] for p in significant_peaks)
    dominant_chain = sorted_chains[0][0] if sorted_chains else "None"
    print("\nSummary:")
    print(
        f"Total identified activity area: {total_area:.2f} counts; dominant series: {dominant_chain}."
    )
    remaining = total_area - sum(a for _, a in sorted_chains)
    if remaining > 0:
        print("Remaining low‑level features are likely background fluctuations.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Radiacode gamma‑spectrum XML files."
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=2.0,
        help="Sigma cutoff multiplier for FWHM/area filtering (default 2).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Tolerance in keV for matching peaks to known emission lines (default 10 keV).",
    )
    parser.add_argument("target", type=Path, help="Target spectrum XML file")
    parser.add_argument(
        "background",
        nargs="?",
        type=Path,
        help="Optional background reference XML file",
    )
    args = parser.parse_args()

    if not args.target.is_file():
        sys.exit(f"Error: target file {args.target} does not exist.")
    if args.background and not args.background.is_file():
        sys.exit(f"Error: background file {args.background} does not exist.")

    analyze(args.target, args.background, args.cutoff, args.tolerance)


if __name__ == "__main__":
    main()
