"""
Step 10: Confidence-threshold sensitivity analysis.

What this script does, in plain English:
We picked 85% as the minimum model confidence required (in both years) for
a Forest-to-other flag to count, back in 08_change_detection.py. That
number wasn't arbitrary, but we also never checked what happens at
nearby thresholds -- maybe 90% would have been meaningfully better, or
maybe 75% would have caught more real deforestation without much cost in
precision. Reporting a single threshold without checking its neighbors is
exactly the kind of thing a paper reviewer would (rightly) ask about.

This script re-runs the confidence filter at every threshold from 70% to
95% (in 5-point steps), and for each one recomputes precision and recall
against Global Forest Watch (reusing the same GFW loss data extracted in
09_validate_gfw.py). It prints a table and saves a plot so the choice of
85% is a documented, justified decision rather than an assumption.
"""

import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
FOREST_INDEX = CLASSES.index("Forest")

# Must match 06_download_sentinel2.py / 09_validate_gfw.py exactly.
CENTER_LON = -63.55
CENTER_LAT = -9.05
WIDTH_PX = 1600
HEIGHT_PX = 1664
N_ROWS = HEIGHT_PX // 64
N_COLS = WIDTH_PX // 64

HANSEN_TILE = "00N_070W"
HANSEN_LOCAL_PATH = os.path.join(DATA_DIR, f"hansen_lossyear_{HANSEN_TILE}.tif")
LOSS_YEAR_MIN = 19  # 2019
LOSS_YEAR_MAX = 23  # 2023
GFW_LOSS_FRACTION_THRESHOLD = 0.10

THRESHOLDS_TO_TEST = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def compute_bbox():
    utm_zone = int((CENTER_LON + 180) / 6) + 1
    utm_epsg = 32700 + utm_zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    center_easting, center_northing = to_utm.transform(CENTER_LON, CENTER_LAT)
    half_w, half_h = WIDTH_PX * 10 / 2, HEIGHT_PX * 10 / 2
    utm_bbox = [center_easting - half_w, center_northing - half_h,
                center_easting + half_w, center_northing + half_h]
    to_lonlat = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_lonlat.transform(utm_bbox[0], utm_bbox[1])
    lon_max, lat_max = to_lonlat.transform(utm_bbox[2], utm_bbox[3])
    return [lon_min, lat_min, lon_max, lat_max]


def load_gfw_flagged_grid():
    bbox = compute_bbox()
    with rasterio.open(HANSEN_LOCAL_PATH) as src:
        window = from_bounds(*bbox, transform=src.transform)
        lossyear = src.read(1, window=window)

    loss_mask = (lossyear >= LOSS_YEAR_MIN) & (lossyear <= LOSS_YEAR_MAX)
    h, w = loss_mask.shape
    gfw_loss_fraction = np.zeros((N_ROWS, N_COLS))
    for row in range(N_ROWS):
        for col in range(N_COLS):
            r0, r1 = int(row * h / N_ROWS), int((row + 1) * h / N_ROWS)
            c0, c1 = int(col * w / N_COLS), int((col + 1) * w / N_COLS)
            block = loss_mask[r0:r1, c0:c1]
            gfw_loss_fraction[row, col] = block.mean() if block.size > 0 else 0.0

    return gfw_loss_fraction >= GFW_LOSS_FRACTION_THRESHOLD


def main():
    grid_2018 = np.load(os.path.join(DATA_DIR, "patch_grid_2018.npy"))
    grid_2024 = np.load(os.path.join(DATA_DIR, "patch_grid_2024.npy"))
    confidence_2018 = np.load(os.path.join(DATA_DIR, "patch_confidence_2018.npy"))
    confidence_2024 = np.load(os.path.join(DATA_DIR, "patch_confidence_2024.npy"))

    was_forest = grid_2018 == FOREST_INDEX
    still_forest = grid_2024 == FOREST_INDEX
    deforested_raw = was_forest & ~still_forest

    print("Loading Global Forest Watch reference data...")
    gfw_flagged = load_gfw_flagged_grid()
    print(f"GFW flags {gfw_flagged.sum()} of {N_ROWS*N_COLS} patches as having tree cover loss (2019-2023)\n")

    print(f"{'Threshold':>10} | {'# Flags':>8} | {'# Confirmed':>11} | {'Precision':>10} | {'Recall':>8}")
    print("-" * 60)

    results = []
    for threshold in THRESHOLDS_TO_TEST:
        confident_both = (confidence_2018 >= threshold) & (confidence_2024 >= threshold)
        flagged = deforested_raw & confident_both
        n_flags = flagged.sum()
        confirmed = (flagged & gfw_flagged).sum()

        precision = confirmed / n_flags if n_flags > 0 else float("nan")
        recall = confirmed / gfw_flagged.sum() if gfw_flagged.sum() > 0 else float("nan")

        results.append((threshold, n_flags, confirmed, precision, recall))
        print(f"{threshold:>9.0%} | {n_flags:>8} | {confirmed:>11} | {precision:>9.1%} | {recall:>7.1%}")

    make_sensitivity_plot(results)


def make_sensitivity_plot(results):
    thresholds = [r[0] * 100 for r in results]
    n_flags = [r[1] for r in results]
    precisions = [r[3] * 100 for r in results]
    recalls = [r[4] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Confidence threshold (%)")
    ax1.set_ylabel("Precision / Recall vs. GFW (%)")
    ax1.plot(thresholds, precisions, marker="o", color="#c0392b", label="Precision")
    ax1.plot(thresholds, recalls, marker="o", color="#3b82c4", label="Recall")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of flagged patches")
    ax2.bar(thresholds, n_flags, width=2, alpha=0.15, color="gray", label="# Flags")
    ax2.legend(loc="upper right")

    plt.title("Confidence Threshold Sensitivity Analysis")
    plt.axvline(x=85, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "confidence_sensitivity.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved sensitivity plot to {out_path}")


if __name__ == "__main__":
    main()
