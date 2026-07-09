"""
Step 7: Chop each Sentinel-2 image into 64x64 patches and classify every
patch with our fine-tuned model.

What this script does, in plain English:
1. Loads the 2018 and 2024 Rondonia images (each 1600x1664 pixels).
2. Slices each one into a grid of non-overlapping 64x64 pixel patches --
   25 columns x 26 rows = 650 patches per image. This matches the
   resolution our model was trained on (EuroSAT patches are also 64x64
   at 10m/pixel), so each patch here represents the same kind of "chunk
   of land" the model already knows how to classify.
3. Runs every patch through the same preprocessing as training (resize to
   224x224, ImageNet normalization) and gets the model's predicted class
   AND its confidence (the softmax probability of that predicted class --
   basically "how sure was the model," from 0 to 1).
4. Saves the results as a grid of class predictions for each year, plus a
   matching grid of confidence scores, plus a quick visual "land cover
   map" so we can sanity-check the output before trusting it for change
   detection.

Why confidence matters here: a spot-check of the raw Forest-to-other
flags showed real deforestation mixed in with likely noise -- patches
sitting at a forest/field boundary contain a mix of land cover, and the
model is forced to pick one label, sometimes with low confidence. Saving
confidence lets the next step filter out the shaky calls and keep only
the ones the model was genuinely sure about in both years.

Outputs:
  ./data/patch_grid_2018.npy       (26x25 array of class indices)
  ./data/patch_grid_2024.npy       (26x25 array of class indices)
  ./data/patch_confidence_2018.npy (26x25 array of confidence scores, 0-1)
  ./data/patch_confidence_2024.npy (26x25 array of confidence scores, 0-1)
  ./figures/landcover_2018.png
  ./figures/landcover_2024.png
"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
MODEL_PATH = os.path.join(MODELS_DIR, "resnet50_finetuned.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PATCH_SIZE = 64
BATCH_SIZE = 64

# Same class order ImageFolder used during training (alphabetical).
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# One distinct color per class, used consistently across both years' maps.
CLASS_COLORS = {
    "AnnualCrop": "#e8d078",
    "Forest": "#1a5c1a",
    "HerbaceousVegetation": "#8fce6b",
    "Highway": "#7d7d7d",
    "Industrial": "#b03a2e",
    "Pasture": "#c2e08a",
    "PermanentCrop": "#a67c3d",
    "Residential": "#d94f4f",
    "River": "#2e86c1",
    "SeaLake": "#1b4f72",
}

YEARS = [2018, 2024]


def build_model(device):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()


def slice_into_patches(image):
    width, height = image.size
    n_cols = width // PATCH_SIZE
    n_rows = height // PATCH_SIZE

    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            left = col * PATCH_SIZE
            top = row * PATCH_SIZE
            patch = image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE)).convert("RGB")
            patches.append(patch)

    return patches, n_rows, n_cols


def classify_patches(model, patches, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_preds = []
    all_confidences = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch = patches[i:i + BATCH_SIZE]
            tensor_batch = torch.stack([transform(p) for p in batch]).to(device)
            outputs = model(tensor_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    return np.array(all_preds), np.array(all_confidences)


def plot_landcover_map(grid, year):
    cmap = ListedColormap([CLASS_COLORS[c] for c in CLASSES])

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1)
    plt.title(f"Land Cover Classification: {year}")
    plt.axis("off")

    legend_handles = [Patch(color=CLASS_COLORS[c], label=c) for c in CLASSES]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f"landcover_{year}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved land cover map to {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_model(device)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for year in YEARS:
        print(f"\nProcessing {year}...")
        image_path = os.path.join(DATA_DIR, f"sentinel2_rondonia_{year}.png")
        image = Image.open(image_path)
        print(f"  Image size: {image.size}")

        patches, n_rows, n_cols = slice_into_patches(image)
        print(f"  Sliced into {len(patches)} patches ({n_rows} rows x {n_cols} cols)")

        preds, confidences = classify_patches(model, patches, device)
        grid = preds.reshape(n_rows, n_cols)
        confidence_grid = confidences.reshape(n_rows, n_cols)

        # Print class distribution for this year so we can sanity-check
        # (e.g. "does Forest coverage make sense for a Rondonia scene?")
        unique, counts = np.unique(preds, return_counts=True)
        print(f"  Class distribution:")
        for idx, count in zip(unique, counts):
            pct = 100 * count / len(preds)
            print(f"    {CLASSES[idx]}: {count} patches ({pct:.1f}%)")

        print(f"  Mean prediction confidence: {confidences.mean():.3f} "
              f"(min={confidences.min():.3f}, max={confidences.max():.3f})")

        out_path = os.path.join(DATA_DIR, f"patch_grid_{year}.npy")
        np.save(out_path, grid)
        print(f"  Saved classification grid to {out_path}")

        confidence_path = os.path.join(DATA_DIR, f"patch_confidence_{year}.npy")
        np.save(confidence_path, confidence_grid)
        print(f"  Saved confidence grid to {confidence_path}")

        plot_landcover_map(grid, year)


if __name__ == "__main__":
    main()
