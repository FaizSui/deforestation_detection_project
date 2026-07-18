"""
Step 11: Decompose recall into two denominators.

Our headline recall (10.5%) divides detections by ALL GFW-positive cells
(172). But our detection rule can only fire on cells the 2018 map
classified Forest -- a GFW-positive cell we labeled (say) Pasture in 2018
was never eligible for detection, no matter how much tree loss occurred
inside it. This script separates the two effects:

  recall_all        = TP / (all GFW-positive cells)
  recall_restricted = TP / (GFW-positive cells classified Forest in 2018)

The gap between them measures how much of the missed loss was
structurally out of reach (2018 label != Forest: cells already majority
non-forest at baseline, or misclassified), versus reachable-but-missed
(partial clearing inside cells that stayed majority forest, plus
confidence-filter rejections). Reviewers will ask exactly this question,
so we answer it with a number.

Reuses the exact bbox/Hansen logic of 09_validate_gfw.py and the exact
detection rule of 08_change_detection.py.
"""

import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
FOREST_INDEX = CLASSES.index("Forest")
CONFIDENCE_THRESHOLD = 0.85

CENTER_LON = -63.55
CENTER_LAT = -9.05
WIDTH_PX = 1600
HEIGHT_PX = 1664
N_ROWS = 26
N_COLS = 25

LOSS_YEAR_MIN = 19
LOSS_YEAR_MAX = 23
GFW_LOSS_FRACTION_THRESHOLD = 0.10
HANSEN_LOCAL_PATH = os.path.join(DATA_DIR, "hansen_lossyear_00N_070W.tif")


def compute_bbox():
    utm_zone = int((CENTER_LON + 180) / 6) + 1
    utm_epsg = 32700 + utm_zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    e, n = to_utm.transform(CENTER_LON, CENTER_LAT)
    hw, hh = WIDTH_PX * 10 / 2, HEIGHT_PX * 10 / 2
    utm_bbox = [e - hw, n - hh, e + hw, n + hh]
    to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_ll.transform(utm_bbox[0], utm_bbox[1])
    lon_max, lat_max = to_ll.transform(utm_bbox[2], utm_bbox[3])
    return [lon_min, lat_min, lon_max, lat_max]


def gfw_positive_grid():
    bbox = compute_bbox()
    with rasterio.open(HANSEN_LOCAL_PATH) as src:
        window = from_bounds(*bbox, transform=src.transform)
        lossyear = src.read(1, window=window)
    loss_mask = (lossyear >= LOSS_YEAR_MIN) & (lossyear <= LOSS_YEAR_MAX)

    h, w = loss_mask.shape
    fraction = np.zeros((N_ROWS, N_COLS))
    for row in range(N_ROWS):
        for col in range(N_COLS):
            r0, r1 = int(row * h / N_ROWS), int((row + 1) * h / N_ROWS)
            c0, c1 = int(col * w / N_COLS), int((col + 1) * w / N_COLS)
            block = loss_mask[r0:r1, c0:c1]
            fraction[row, col] = block.mean() if block.size else 0.0
    return fraction >= GFW_LOSS_FRACTION_THRESHOLD


def main():
    grid_2018 = np.load(os.path.join(DATA_DIR, "patch_grid_2018.npy"))
    grid_2024 = np.load(os.path.join(DATA_DIR, "patch_grid_2024.npy"))
    conf_2018 = np.load(os.path.join(DATA_DIR, "patch_confidence_2018.npy"))
    conf_2024 = np.load(os.path.join(DATA_DIR, "patch_confidence_2024.npy"))

    was_forest = grid_2018 == FOREST_INDEX
    detected = (was_forest
                & (grid_2024 != FOREST_INDEX)
                & (conf_2018 >= CONFIDENCE_THRESHOLD)
                & (conf_2024 >= CONFIDENCE_THRESHOLD))

    gfw = gfw_positive_grid()

    tp = (detected & gfw).sum()
    n_gfw = gfw.sum()
    n_gfw_forest = (gfw & was_forest).sum()
    n_gfw_nonforest = (gfw & ~was_forest).sum()

    print(f"Detections (tau={CONFIDENCE_THRESHOLD}): {detected.sum()}")
    print(f"GFW-positive cells (all): {n_gfw}")
    print(f"  of which classified Forest in 2018 (eligible for our rule): {n_gfw_forest}")
    print(f"  of which classified non-Forest in 2018 (ineligible): {n_gfw_nonforest} "
          f"({100*n_gfw_nonforest/n_gfw:.1f}% of GFW-positive cells)")

    print(f"\nTrue positives (detected AND GFW-positive): {tp}")
    print(f"Recall, all GFW-positive cells:        {tp}/{n_gfw} = {100*tp/n_gfw:.1f}%")
    print(f"Recall, eligible (2018-Forest) cells:  {tp}/{n_gfw_forest} = {100*tp/n_gfw_forest:.1f}%")

    # Where did the ineligible cells' 2018 labels go? Context for Discussion.
    print(f"\n2018 labels of the {n_gfw_nonforest} ineligible GFW-positive cells:")
    labels, counts = np.unique(grid_2018[gfw & ~was_forest], return_counts=True)
    for idx, count in sorted(zip(labels, counts), key=lambda x: -x[1]):
        print(f"  {CLASSES[idx]}: {count}")

    # And among eligible-but-missed: how many failed the confidence gate vs
    # stayed classified Forest in 2024?
    eligible_missed = gfw & was_forest & ~detected
    still_forest_2024 = (grid_2024 == FOREST_INDEX)
    n_still_forest = (eligible_missed & still_forest_2024).sum()
    n_conf_reject = (eligible_missed & ~still_forest_2024).sum()
    print(f"\nEligible GFW-positive cells we missed: {eligible_missed.sum()}")
    print(f"  still classified Forest in 2024 (partial clearing, label unchanged): {n_still_forest}")
    print(f"  label changed but rejected by confidence filter: {n_conf_reject}")


if __name__ == "__main__":
    main()
