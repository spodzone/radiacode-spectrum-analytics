# Radiacode Spectrum Analytics v3

A small Python utility for analysing [Radiacode](https://www.radiacode.com/) gamma‑spectrum XML files.

---

## 📦 Installation

```bash
# Clone the repository (if you haven't already)
git clone <repo-url>
cd radiacode-spectrum-analytics

# Create a virtual environment (recommended) and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

> The script only depends on the standard library plus **numpy**, **scipy** and **matplotlib**.

---

## 🚀 Quick start – analysing a single spectrum

```bash
./gamma-spectrum-analysis.py "Spectrum.xml"
```

* The script reads the XML file and outputs:
  * `Spectrum.png` – a high‑resolution plot showing the net spectrum, the fitted baseline and shaded regions above/below it.
  * A terminal table listing detected peaks (energy, area, FWHM, matched nuclide, decay chain, notes, % contribution).
  * A second table summarising contributions by decay chain.
  * A short textual summary.

---

## 🛡️ Optional background reference

If you have a separate background measurement you can supply it as the second positional argument:

```bash
./gamma-spectrum-analysis.py target.xml background.xml
```

The script will:
1. Parse both files.
2. Scale the background counts to match the measurement time of the target.
3. Subtract the scaled background (negative values clipped to zero).
4. Continue with baseline fitting and peak detection on the net spectrum.

If the subtraction leaves almost no signal, a warning is printed and the raw target spectrum is used instead.

---

## 🧩 How it works – algorithm overview

1. **Calibration** – Channel numbers are converted to energies using the polynomial coefficients stored in the XML (`E = c0 + c1·ch + c2·ch² …`).
2. **Baseline estimation** – A low‑order spline (`scipy.interpolate.UnivariateSpline`) is fitted with a *very small* smoothing factor (≈0.1 % of the variance‑based default). This gives a LOESS‑style smooth background that follows the broad shape of the spectrum without tracking individual peaks.
3. **Peak regions** – The net counts are compared to the baseline; contiguous sections where `net ≥ baseline` define candidate peak regions.
4. **Peak refinement** – Within each region the channel with the maximum count defines the peak centre. Full‑width‑half‑maximum (FWHM) is estimated from the half‑max points.
5. **Statistical filtering** –
   * A robust noise estimate (`MAD`) sets a minimum area threshold.
   * Peaks are kept only if their FWHM and area exceed `sigma_cutoff` times the standard deviation of the respective distributions (default σ = 2).
6. **Nuclide matching** – Detected peak energies are matched to a small built‑in library within a user‑configurable tolerance (default ±10 keV). 
7. **Scoring & reporting** – Each retained peak contributes `area × energy`. Percent contributions are calculated, tables sorted by contribution and a summary printed.

---

## ⚙️ Command‑line parameters (noise filtering)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cutoff` | float | `2.0` | Peaks with fewer than this many sigma in either **FWHM** or **area** are filtered. |
| `--tolerance` | float | `10.0` | Energy tolerance (keV) used when matching a detected peak to a known gamma line. A larger window will produce more matches but also increase false positives. |

These two flags let you prune spurious detections that arise from statistical fluctuations or an overly aggressive baseline fit.

---

## 📂 Repository layout
```
radiacode-spectrum-analytics/
├── gamma-spectrum-analysis.py  # main entry point
├── background.xml              # sample background file
├── spectrum.xml                # sample target spectrum
└── README.md                   # this file
```

---

## 🙋‍♀️ Contributing & issues
Feel free to open a pull request or an issue if you spot a bug, have suggestions for additional nuclides, or want to improve the baseline algorithm.

---

*Happy analysing!*
