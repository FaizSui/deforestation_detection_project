"""
Step 9: Validate our model's deforestation flags against Global Forest
Watch (GFW) data.

What this script does, in plain English:
1. Downloads (once, then caches) the Hansen Global Forest Change tile
   that covers our study area. This is the actual raw dataset behind
   Global Forest Watch's tree-cover-loss maps -- a global 30m-resolution
   record of which years each pixel of forest was cleared, built from
   Landsat satellite imagery by University of Maryland / Google
   researchers. It's public data, no account needed.
2. Reads just the "lossyear" band (each pixel's value is 0 if no loss was
   ever detected, or 1-23 meaning loss was first detected in 2001-2023)
   for the small window matching our bounding box.
3. For each of our 26x25 patches, checks what fraction of its Hansen
   pixels show loss in 2019-2023 (loss recorded AFTER our 2018 baseline
   image, closest available proxy for "loss during our study period").
   A patch counts as "GFW-confirmed loss" if more than 10% of its area
   shows this.
4. Compares that against our model's 23 high-confidence deforestation
   flags: how many does GFW also confirm? How many did GFW catch that we
   missed? Prints the agreement rate -- this is the core validation
   number for the paper.

Important caveat printed by this script: the Hansen dataset (v1.11) only
covers loss THROUGH 2023, not 2024. Any clearing that happened specifically
in our 2024 image but not by end of 2023 would show up in our detections
but COULDN'T show up in GFW's data yet -- that's a real limitation to
report, not a bug to hide.
"""

import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# This MUST derive the exact same footprint as 06_download_sentinel2.py,
# or the Hansen loss-year window read below will be shifted relative to
# our patches. Rather than hardcode a separately-computed lon/lat box
# (which is how a ~150-300m misalignment bug slipped in previously), we
# rebuild the identical UTM-exact bbox from the same center point and
# convert it back to lon/lat -- the Hansen tile's native CRS -- the same
# way 06_download_sentinel2.py does for its own catalog search.
CENTER_LON = -63.55
CENTER_LAT = -9.05
WIDTH_PX = 1600
HEIGHT_PX = 1664
PATCH_SIZE = 64

_utm_zone = int((CENTER_LON + 180) / 6) + 1
_utm_epsg = 32700 + _utm_zone
_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{_utm_epsg}", always_xy=True)
_center_easting, _center_northing = _to_utm.transform(CENTER_LON, CENTER_LAT)
_half_width_m = WIDTH_PX * 10 / 2
_half_height_m = HEIGHT_PX * 10 / 2
_utm_bbox = [
    _center_easting - _half_width_m,
    _center_northing - _half_height_m,
    _center_easting + _half_width_m,
    _center_northing + _half_height_m,
]
_to_lonlat = Transformer.from_crs(f"EPSG:{_utm_epsg}", "EPSG:4326", always_xy=True)
_lon_min, _lat_min = _to_lonlat.transform(_utm_bbox[0], _utm_bbox[1])
_lon_max, _lat_max = _to_lonlat.transform(_utm_bbox[2], _utm_bbox[3])
BBOX = [_lon_min, _lat_min, _lon_max, _lat_max]
N_ROWS = HEIGHT_PX // PATCH_SIZE  # 26
N_COLS = WIDTH_PX // PATCH_SIZE   # 25

# The 10x10 degree Hansen tile covering our bbox (lon -70..-60, lat -10..0).
# Note: Hansen tiles are named by their literal NW (top-left) corner
# coordinate, extending SOUTH and EAST from there. Our first guess,
# "10S_070W", turned out to be the WRONG neighboring tile -- it actually
# spans lat -10 to -20 (i.e. it's the tile immediately south of ours).
# The correct tile covering lat -10 to 0 is "00N_070W" (verified by
# checking its bounds directly before downloading the full file).
HANSEN_TILE = "00N_070W"
HANSEN_VERSION = "GFC-2023-v1.11"
HANSEN_URL = (
    f"https://storage.googleapis.com/earthenginepartners-hansen/{HANSEN_VERSION}/"
    f"Hansen_{HANSEN_VERSION}_lossyear_{HANSEN_TILE}.tif"
)
HANSEN_LOCAL_PATH = os.path.join(DATA_DIR, f"hansen_lossyear_{HANSEN_TILE}.tif")

# Hansen's lossyear encodes 1=2001 ... 23=2023. Our 2018 baseline image
# means we only care about loss AFTER that: 2019 (19) through 2023 (23).
# We can't check 2024 -- the dataset doesn't go that far yet.
LOSS_YEAR_MIN = 19  # 2019
LOSS_YEAR_MAX = 23  # 2023 (latest available in v1.11)

# A patch counts as "GFW confirms loss here" if at least this fraction of
# its 30m Hansen pixels show loss in the window above.
GFW_LOSS_FRACTION_THRESHOLD = 0.10


def download_hansen_tile():
    if os.path.exists(HANSEN_LOCAL_PATH):
        print(f"Hansen tile already downloaded at {HANSEN_LOCAL_PATH}")
        return

    print(f"Downloading Hansen tile {HANSEN_TILE} (~85MB, one-time download)...")
    response = requests.get(HANSEN_URL, stream=True)
    response.raise_for_status()

    with open(HANSEN_LOCAL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    size_mb = os.path.getsize(HANSEN_LOCAL_PATH) / (1024 * 1024)
    print(f"Saved {HANSEN_LOCAL_PATH} ({size_mb:.0f} MB)")


def read_lossyear_window():
    """Read just the small window of the Hansen tile matching our bbox,
    then resample it onto a grid with one value per our 64x64 patch:
    the fraction of that patch's area showing loss in 2019-2023."""

    with rasterio.open(HANSEN_LOCAL_PATH) as src:
        min_lon, min_lat, max_lon, max_lat = BBOX
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
        lossyear = src.read(1, window=window)
        print(f"Read Hansen window: {lossyear.shape[1]}x{lossyear.shape[0]} pixels "
              f"(Hansen is ~30m/pixel, ours is 10m/pixel)")

    loss_mask = (lossyear >= LOSS_YEAR_MIN) & (lossyear <= LOSS_YEAR_MAX)

    # Resize the Hansen loss mask onto our N_ROWS x N_COLS patch grid by
    # computing, for each of our patches, what fraction of the
    # corresponding Hansen pixels fall inside that patch's footprint and
    # show loss. Since Hansen pixels (30m) are coarser than our patches'
    # source resolution but the whole window covers the same ground
    # area, we can just divide the Hansen array into N_ROWS x N_COLS
    # blocks proportionally.
    h, w = loss_mask.shape
    gfw_loss_fraction = np.zeros((N_ROWS, N_COLS))

    for row in range(N_ROWS):
        for col in range(N_COLS):
            r0 = int(row * h / N_ROWS)
            r1 = int((row + 1) * h / N_ROWS)
            c0 = int(col * w / N_COLS)
            c1 = int((col + 1) * w / N_COLS)
            block = loss_mask[r0:r1, c0:c1]
            gfw_loss_fraction[row, col] = block.mean() if block.size > 0 else 0.0

    return gfw_loss_fraction


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_hansen_tile()

    print(f"\nExtracting GFW tree-cover-loss data for our study area...")
    print(f"Loss window checked: 20{LOSS_YEAR_MIN}-20{LOSS_YEAR_MAX} "
          f"(GFW data doesn't yet cover 2024)")
    gfw_loss_fraction = read_lossyear_window()

    gfw_flagged = gfw_loss_fraction >= GFW_LOSS_FRACTION_THRESHOLD
    print(f"\nGFW flags {gfw_flagged.sum()} of {N_ROWS*N_COLS} patches as having "
          f">={GFW_LOSS_FRACTION_THRESHOLD:.0%} tree cover loss in 2019-2023")

    our_flags = np.load(os.path.join(DATA_DIR, "deforestation_flags.npy"))
    our_flagged = np.zeros((N_ROWS, N_COLS), dtype=bool)
    for row, col in our_flags:
        our_flagged[row, col] = True

    print(f"Our model flags {our_flagged.sum()} patches as high-confidence deforestation")

    both = our_flagged & gfw_flagged
    only_ours = our_flagged & ~gfw_flagged
    only_gfw = gfw_flagged & ~our_flagged

    print(f"\n--- Agreement ---")
    print(f"Flagged by BOTH our model and GFW: {both.sum()}")
    print(f"Flagged by our model ONLY (GFW disagrees or has no data yet): {only_ours.sum()}")
    print(f"Flagged by GFW ONLY (our model missed): {only_gfw.sum()}")

    if our_flagged.sum() > 0:
        agreement_rate = 100 * both.sum() / our_flagged.sum()
        print(f"\nOf our {our_flagged.sum()} detections, {both.sum()} ({agreement_rate:.1f}%) "
              f"are independently confirmed by Global Forest Watch.")

    if gfw_flagged.sum() > 0:
        recall_rate = 100 * both.sum() / gfw_flagged.sum()
        print(f"Of GFW's {gfw_flagged.sum()} flagged patches, we caught {both.sum()} "
              f"({recall_rate:.1f}%).")

    make_comparison_map(our_flagged, gfw_flagged)


def make_comparison_map(our_flagged, gfw_flagged):
    """Blue = only us, yellow = only GFW, red = both agree, so we can see
    WHERE the agreement and disagreement is, not just the count."""

    category = np.zeros((N_ROWS, N_COLS), dtype=int)
    category[our_flagged & ~gfw_flagged] = 1  # ours only
    category[gfw_flagged & ~our_flagged] = 2  # GFW only
    category[our_flagged & gfw_flagged] = 3   # both

    colors = ["#f0f0f0", "#3b82c4", "#e8b923", "#c0392b"]
    cmap = ListedColormap(colors)

    plt.figure(figsize=(9, 9))
    plt.imshow(category, cmap=cmap, vmin=0, vmax=3)
    plt.title("Our Model vs. Global Forest Watch: Detected Forest Loss")
    plt.axis("off")

    from matplotlib.patches import Patch as LegendPatch
    legend_handles = [
        LegendPatch(color=colors[1], label="Flagged by our model only"),
        LegendPatch(color=colors[2], label="Flagged by GFW only"),
        LegendPatch(color=colors[3], label="Flagged by both (agreement)"),
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "gfw_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison map to {out_path}")


if __name__ == "__main__":
    main()
