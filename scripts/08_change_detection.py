"""
Step 8: Compare the 2018 and 2024 land-cover grids and flag deforestation.

What this script does, in plain English:
1. Loads the two classification grids saved by 07_classify_patches.py.
   Because both Sentinel-2 images used the EXACT same bounding box and
   pixel dimensions, grid cell (row=5, col=10) in 2018 and grid cell
   (row=5, col=10) in 2024 represent the SAME 64x64 patch of ground --
   so we can compare them directly, cell by cell.
2. For every cell, checks: was this Forest in 2018, and NOT Forest in
   2024? If so, we flag it as a candidate deforestation event.
3. Filters those candidates by CONFIDENCE: a spot-check of the raw flags
   found real deforestation mixed in with likely noise from patches that
   sit at a forest/field boundary (mixed content the model has to force
   into one label). We only keep a flag if the model was confident (by
   default >=85%) about its prediction in BOTH years -- confident it was
   Forest in 2018, and confident it was something else in 2024. This
   trades a smaller count for a more trustworthy one. We print BOTH the
   raw and filtered counts so the effect of this filter is visible, not
   hidden.
4. Prints a breakdown of what the (filtered) deforested patches turned
   into (e.g. how many became AnnualCrop vs Pasture vs Highway), since
   which class it became is informative -- cropland conversion tells a
   different story than highway construction.
5. Saves a "change map" figure: the 2024 image with every high-confidence
   flagged patch highlighted in red, so we can see exactly where the
   model thinks deforestation happened.

Important honesty check: even after confidence filtering, this is a
MODEL-DETECTED change map, not verified ground truth -- that's exactly
why the next step compares it against Global Forest Watch data.
"""

import os

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

PATCH_SIZE = 64
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
FOREST_INDEX = CLASSES.index("Forest")

# Minimum softmax confidence required in BOTH years for a Forest-to-other
# flag to be kept. 0.85 is a reasonably strict cutoff -- comfortably above
# "barely more likely than the next class" (which for 10 classes could be
# as low as ~0.11) but not so strict we throw away good detections.
CONFIDENCE_THRESHOLD = 0.85


def main():
    grid_2018 = np.load(os.path.join(DATA_DIR, "patch_grid_2018.npy"))
    grid_2024 = np.load(os.path.join(DATA_DIR, "patch_grid_2024.npy"))
    confidence_2018 = np.load(os.path.join(DATA_DIR, "patch_confidence_2018.npy"))
    confidence_2024 = np.load(os.path.join(DATA_DIR, "patch_confidence_2024.npy"))

    assert grid_2018.shape == grid_2024.shape, "Grids must be the same shape -- did both images use the same bbox/size?"
    n_rows, n_cols = grid_2018.shape
    total_patches = n_rows * n_cols

    was_forest = grid_2018 == FOREST_INDEX
    still_forest = grid_2024 == FOREST_INDEX
    deforested_raw = was_forest & ~still_forest

    confident_both_years = (confidence_2018 >= CONFIDENCE_THRESHOLD) & (confidence_2024 >= CONFIDENCE_THRESHOLD)
    deforested = deforested_raw & confident_both_years

    n_forest_2018 = was_forest.sum()
    n_deforested_raw = deforested_raw.sum()
    n_deforested = deforested.sum()

    print(f"Grid size: {n_rows} rows x {n_cols} cols = {total_patches} patches")
    print(f"Forest patches in 2018: {n_forest_2018} ({100*n_forest_2018/total_patches:.1f}% of area)")
    print(f"Forest patches in 2024: {still_forest.sum()} ({100*still_forest.sum()/total_patches:.1f}% of area)")
    print(f"\nRaw Forest-to-other flags (no confidence filter): {n_deforested_raw}")
    print(f"Confidence threshold: >={CONFIDENCE_THRESHOLD:.0%} in BOTH years")
    print(f"High-confidence flags after filtering: {n_deforested} "
          f"(dropped {n_deforested_raw - n_deforested} low-confidence flags, "
          f"{100*(n_deforested_raw - n_deforested)/max(n_deforested_raw,1):.1f}% of raw flags)")
    if n_forest_2018 > 0:
        print(f"  High-confidence flags = {100*n_deforested/n_forest_2018:.1f}% of 2018's forest area")

    print(f"\nWhat those high-confidence deforested patches became in 2024:")
    changed_to = grid_2024[deforested]
    unique, counts = np.unique(changed_to, return_counts=True)
    for idx, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {CLASSES[idx]}: {count} patches ({100*count/n_deforested:.1f}% of deforested patches)")

    # Sanity check the other direction too: patches that BECAME forest
    # (gain, not loss). Some of this is expected classifier noise, but a
    # large number would be a red flag worth investigating.
    gained_forest_raw = (~was_forest) & still_forest
    gained_forest = gained_forest_raw & confident_both_years
    print(f"\nFor reference, patches that became Forest (regrowth or misclassification): "
          f"{gained_forest_raw.sum()} raw, {gained_forest.sum()} high-confidence")

    # Save the flagged patch coordinates so the next step (GFW validation)
    # can look up their real-world lat/lon.
    flagged_coords = np.argwhere(deforested)
    out_path = os.path.join(DATA_DIR, "deforestation_flags.npy")
    np.save(out_path, flagged_coords)
    print(f"\nSaved {len(flagged_coords)} high-confidence flagged patch coordinates to {out_path}")

    make_change_map(grid_2018, deforested, n_rows, n_cols)


def make_change_map(grid_2018, deforested, n_rows, n_cols):
    """Overlay red squares on the 2024 image everywhere we flagged
    Forest-to-other change, so we can see WHERE it's happening, not just
    the count."""

    image_2024 = Image.open(os.path.join(DATA_DIR, "sentinel2_rondonia_2024.png")).convert("RGB")
    overlay = image_2024.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for row in range(n_rows):
        for col in range(n_cols):
            if deforested[row, col]:
                left = col * PATCH_SIZE
                top = row * PATCH_SIZE
                draw.rectangle(
                    [left, top, left + PATCH_SIZE, top + PATCH_SIZE],
                    fill=(255, 0, 0, 110),
                    outline=(255, 0, 0, 255),
                )

    out_path = os.path.join(FIGURES_DIR, "deforestation_flags.png")
    overlay.save(out_path)
    print(f"Saved change map to {out_path}")


if __name__ == "__main__":
    main()
