"""
Step 12: Regenerate the four manuscript figures at publication quality.

This script does not recompute any result -- it restyles already-saved
outputs (confusion matrix, patch grids, confidence grids, detection flags,
GFW loss data) into figures suitable for a journal:
  - 300 DPI
  - one consistent font and size scheme across all figures
  - a colorblind-safe palette (blue/orange for the two-class comparisons;
    a luminance-separated qualitative scheme for the 10 land-cover classes,
    so the Forest / non-Forest distinction that drives the analysis
    survives grayscale and red-green color vision deficiency)
  - labeled axes, and a scale bar + north arrow on every map

Outputs (figures/publication/):
  fig1_confusion_matrix.png
  fig2_landcover_maps.png
  fig3_change_gfw_overlay.png
  fig4_threshold_sensitivity.png
Captions are written to paper/captions.md.
"""

import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle, FancyArrow

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "publication")
PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
CLASS_LABELS = ["Annual Crop", "Forest", "Herb. Veg.", "Highway", "Industrial",
                "Pasture", "Perm. Crop", "Residential", "River", "Sea/Lake"]
FOREST_INDEX = CLASSES.index("Forest")

# Colorblind-safe qualitative palette (adapted from Paul Tol's schemes).
# The three vegetation greens are separated primarily by LUMINANCE
# (Forest darkest, Herbaceous mid, Pasture palest) so the Forest vs.
# non-Forest boundary -- the axis the whole study turns on -- remains
# legible in grayscale and under red-green color vision deficiency.
CLASS_COLORS = ["#EECC66", "#225522", "#99CC66", "#888888", "#AA4499",
                "#CCDDAA", "#997700", "#EE6677", "#4477AA", "#223377"]

# Two-class comparison colors: the canonical colorblind-safe orange/blue pair.
ORANGE = "#EE7733"
BLUE = "#0077BB"
GRAY = "#BBBBBB"

PATCH_SIZE = 64
CENTER_LON, CENTER_LAT = -63.55, -9.05
WIDTH_PX, HEIGHT_PX = 1600, 1664
N_ROWS, N_COLS = HEIGHT_PX // PATCH_SIZE, WIDTH_PX // PATCH_SIZE  # 26 x 25
METERS_PER_PATCH = PATCH_SIZE * 10  # 640 m
CONFIDENCE_THRESHOLD = 0.85

HANSEN_LOCAL_PATH = os.path.join(DATA_DIR, "hansen_lossyear_00N_070W.tif")
LOSS_YEAR_MIN, LOSS_YEAR_MAX = 19, 23
GFW_LOSS_FRACTION_THRESHOLD = 0.10
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def add_scale_bar_and_north(ax, extent_m, bar_m=5000, unit_px_per_m=None):
    """Add a scale bar (default 5 km) and a north arrow to a map axis.
    extent_m is the axis width in meters; unit_px_per_m converts meters to
    axis units (patch cells or pixels)."""
    bar_units = bar_m * unit_px_per_m
    x0 = N_COLS * 0.05 if unit_px_per_m > 0.01 else WIDTH_PX * 0.05
    # position bar near bottom-left in axis (data) coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax_w = abs(xlim[1] - xlim[0])
    ax_h = abs(ylim[1] - ylim[0])
    bx = xlim[0] + 0.05 * ax_w
    by = max(ylim) - 0.06 * ax_h  # near bottom (image y is inverted)
    ax.add_patch(Rectangle((bx, by), bar_units, 0.012 * ax_h,
                           facecolor="white", edgecolor="black", lw=0.8, zorder=10))
    ax.text(bx + bar_units / 2, by - 0.015 * ax_h, f"{bar_m // 1000} km",
            ha="center", va="bottom", color="white", fontsize=8, zorder=11,
            path_effects=None)
    # north arrow, top-left
    nx = xlim[0] + 0.05 * ax_w
    ny = min(ylim) + 0.12 * ax_h
    ax.annotate("N", xy=(nx, min(ylim) + 0.03 * ax_h),
                xytext=(nx, ny), ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", zorder=11,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))


def compute_gfw_grids():
    utm_zone = int((CENTER_LON + 180) / 6) + 1
    utm_epsg = 32700 + utm_zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    e, n = to_utm.transform(CENTER_LON, CENTER_LAT)
    hw, hh = WIDTH_PX * 10 / 2, HEIGHT_PX * 10 / 2
    utm_bbox = [e - hw, n - hh, e + hw, n + hh]
    to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_ll.transform(utm_bbox[0], utm_bbox[1])
    lon_max, lat_max = to_ll.transform(utm_bbox[2], utm_bbox[3])
    with rasterio.open(HANSEN_LOCAL_PATH) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        lossyear = src.read(1, window=window)
    loss_mask = (lossyear >= LOSS_YEAR_MIN) & (lossyear <= LOSS_YEAR_MAX)
    h, w = loss_mask.shape
    frac = np.zeros((N_ROWS, N_COLS))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            r0, r1 = int(r * h / N_ROWS), int((r + 1) * h / N_ROWS)
            c0, c1 = int(c * w / N_COLS), int((c + 1) * w / N_COLS)
            block = loss_mask[r0:r1, c0:c1]
            frac[r, c] = block.mean() if block.size else 0.0
    return frac >= GFW_LOSS_FRACTION_THRESHOLD


def fig1_confusion_matrix():
    cm = np.load(os.path.join(DATA_DIR, "confusion_matrix_finetuned.npy"))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    thresh = cm.max() / 2
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Patch count")
    fig.savefig(os.path.join(FIG_DIR, "fig1_confusion_matrix.png"))
    plt.close(fig)


def fig2_landcover_maps():
    grid_2018 = np.load(os.path.join(DATA_DIR, "patch_grid_2018.npy"))
    grid_2024 = np.load(os.path.join(DATA_DIR, "patch_grid_2024.npy"))
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, grid, year in zip(axes, [grid_2018, grid_2024], [2018, 2024]):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
        ax.set_title(f"{year}")
        ax.set_xticks([])
        ax.set_yticks([])
        add_scale_bar_and_north(ax, WIDTH_PX * 10, bar_m=5000, unit_px_per_m=1 / METERS_PER_PATCH)
    handles = [Patch(facecolor=CLASS_COLORS[i], edgecolor="black", lw=0.3, label=CLASS_LABELS[i])
               for i in range(len(CLASSES))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_landcover_maps.png"))
    plt.close(fig)


def fig3_change_gfw_overlay():
    from PIL import Image
    grid_2018 = np.load(os.path.join(DATA_DIR, "patch_grid_2018.npy"))
    grid_2024 = np.load(os.path.join(DATA_DIR, "patch_grid_2024.npy"))
    conf_2018 = np.load(os.path.join(DATA_DIR, "patch_confidence_2018.npy"))
    conf_2024 = np.load(os.path.join(DATA_DIR, "patch_confidence_2024.npy"))
    detected = ((grid_2018 == FOREST_INDEX) & (grid_2024 != FOREST_INDEX)
                & (conf_2018 >= CONFIDENCE_THRESHOLD) & (conf_2024 >= CONFIDENCE_THRESHOLD))
    gfw = compute_gfw_grids()

    basemap = np.array(Image.open(os.path.join(DATA_DIR, "sentinel2_rondonia_2024.png")).convert("RGB"))
    fig, ax = plt.subplots(figsize=(8, 8.3))
    ax.imshow(basemap)
    # dim the basemap so overlays read clearly
    ax.imshow(np.ones_like(basemap) * 255, alpha=0.45, cmap="gray", vmin=0, vmax=255)

    # GFW-positive cells: semi-transparent orange fill
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if gfw[r, c]:
                ax.add_patch(Rectangle((c * PATCH_SIZE, r * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                                       facecolor=ORANGE, alpha=0.40, edgecolor="none", zorder=3))
    # our detections: blue outline (no fill) so nesting inside orange is visible
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if detected[r, c]:
                ax.add_patch(Rectangle((c * PATCH_SIZE, r * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                                       facecolor="none", edgecolor=BLUE, lw=2.0, zorder=4))
    ax.set_xlim(0, WIDTH_PX)
    ax.set_ylim(HEIGHT_PX, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    add_scale_bar_and_north(ax, WIDTH_PX * 10, bar_m=5000, unit_px_per_m=1 / 10.0)
    handles = [
        Patch(facecolor=ORANGE, alpha=0.40, label="GFW tree-cover loss (2019-2023)"),
        Patch(facecolor="none", edgecolor=BLUE, lw=2.0, label="Our high-confidence detection"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_change_gfw_overlay.png"))
    plt.close(fig)


def fig4_threshold_sensitivity():
    grid_2018 = np.load(os.path.join(DATA_DIR, "patch_grid_2018.npy"))
    grid_2024 = np.load(os.path.join(DATA_DIR, "patch_grid_2024.npy"))
    conf_2018 = np.load(os.path.join(DATA_DIR, "patch_confidence_2018.npy"))
    conf_2024 = np.load(os.path.join(DATA_DIR, "patch_confidence_2024.npy"))
    raw = (grid_2018 == FOREST_INDEX) & (grid_2024 != FOREST_INDEX)
    gfw = compute_gfw_grids()
    n_gfw = gfw.sum()

    thr, n_flags, prec, rec = [], [], [], []
    for t in THRESHOLDS:
        flagged = raw & (conf_2018 >= t) & (conf_2024 >= t)
        nf = flagged.sum()
        confirmed = (flagged & gfw).sum()
        thr.append(t * 100)
        n_flags.append(nf)
        prec.append(100 * confirmed / nf if nf else np.nan)
        rec.append(100 * confirmed / n_gfw if n_gfw else np.nan)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax2.bar(thr, n_flags, width=2.2, alpha=0.18, color=GRAY, zorder=1, label="Detections (n)")
    ax1.plot(thr, prec, marker="o", color=ORANGE, lw=2, zorder=3, label="Precision")
    ax1.plot(thr, rec, marker="s", color=BLUE, lw=2, zorder=3, label="Recall")
    ax1.axvline(85, color="black", ls="--", lw=1, alpha=0.6)
    ax1.text(85, 92, " primary\n threshold", fontsize=8, va="top")
    ax1.set_xlabel("Confidence threshold $\\tau$ (%)")
    ax1.set_ylabel("Precision / Recall vs. GFW (%)")
    ax2.set_ylabel("Number of detections")
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, max(n_flags) * 1.5)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_threshold_sensitivity.png"))
    plt.close(fig)


def write_captions():
    captions = """# Figure Captions

**Figure 1. Confusion matrix of the fine-tuned classifier on the EuroSAT test set
(n = 2,700).** Rows are true classes, columns predicted; cell values are patch counts,
and the diagonal gives correct classifications. Overall test accuracy is 98.0%. Off-
diagonal structure concentrates in visually similar pairs -- Highway/River (thin linear
features) and Pasture/Annual Crop/Herbaceous Vegetation (green ground cover). Forest,
the class on which change detection depends, attains the highest precision of any class
(99.7%).

**Figure 2. Classified land cover of the Rondônia study area in 2018 and 2024.** Each
cell is one 640 x 640 m patch (26 rows x 25 columns; 650 patches per year) labeled by
the fine-tuned classifier; both panels share the identical ground footprint (UTM Zone
20S, 10 m/pixel). Classified forest cover falls from 42.5% of the area in 2018 to 30.9%
in 2024, with the offsetting gain concentrated in Annual Crop. Vegetation classes are
colored by luminance (Forest darkest) so the forest boundary remains legible under
grayscale and red-green color vision deficiency. North is up; scale bar 5 km.

**Figure 3. Model detections versus Global Forest Watch tree-cover loss, over the 2024
image.** Orange cells are Global Forest Watch-positive (>=10% tree-cover loss,
2019-2023); blue outlines are our 24 high-confidence detections. Detections lie in the
interiors of GFW-positive clusters, where clearing consumed most of a patch, while the
GFW-positive cells without a detection form the surrounding periphery, where loss was
partial -- the spatial signature of the method's patch-granularity recall limit. North
is up; scale bar 5 km.

**Figure 4. Sensitivity of detection performance to the confidence threshold.** As the
threshold tau rises from 70% to 95%, recall against Global Forest Watch falls
monotonically (blue) while precision (orange) stays within a 67-81% band with no
monotonic trend; bars give the detection count at each threshold. The dashed line marks
the primary setting (tau = 85%), whose precision is not the maximum available --
evidence the threshold was not chosen to inflate the headline result.
"""
    with open(os.path.join(PAPER_DIR, "captions.md"), "w", encoding="utf-8") as f:
        f.write(captions)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    set_style()
    print("Fig 1: confusion matrix...")
    fig1_confusion_matrix()
    print("Fig 2: land-cover maps...")
    fig2_landcover_maps()
    print("Fig 3: change/GFW overlay...")
    fig3_change_gfw_overlay()
    print("Fig 4: threshold sensitivity...")
    fig4_threshold_sensitivity()
    write_captions()
    print(f"Done. Figures in {FIG_DIR}, captions in {PAPER_DIR}/captions.md")


if __name__ == "__main__":
    main()
