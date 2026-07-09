"""
Step 2: Explore the EuroSAT dataset before training anything.

What this script does, in plain English:
1. Counts how many images are in each of the 10 class folders, and prints
   a bar chart of those counts to ./figures/class_counts.png
   (Checking class balance matters: if one class had 10x more images than
   another, the model could get lazy and just guess the common class a lot.)
2. Picks a few random sample images from each class and saves them as a
   grid to ./figures/sample_images.png, so we can visually sanity-check
   that the images look like what their folder names claim.

We're not touching PyTorch yet — this is just "look at the data before you
trust it," which is standard practice before any ML project.
"""

import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "EuroSAT_RGB")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def get_class_counts():
    classes = sorted(os.listdir(DATA_DIR))
    counts = OrderedDict()
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))
    return counts


def plot_class_counts(counts):
    plt.figure(figsize=(10, 5))
    plt.bar(counts.keys(), counts.values(), color="seagreen")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of images")
    plt.title("EuroSAT RGB: images per class")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "class_counts.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved class distribution chart to {out_path}")


def plot_sample_images(counts, samples_per_class=3):
    classes = list(counts.keys())
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(samples_per_class * 2, len(classes) * 2))

    for row, cls in enumerate(classes):
        cls_path = os.path.join(DATA_DIR, cls)
        image_files = os.listdir(cls_path)
        chosen = random.sample(image_files, samples_per_class)

        for col, fname in enumerate(chosen):
            img = Image.open(os.path.join(cls_path, fname))
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, rotation=0, labelpad=40, fontsize=9)

        axes[row, 0].set_title(cls, loc="left", fontsize=9, pad=2)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "sample_images.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved sample image grid to {out_path}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    counts = get_class_counts()
    print("Class counts:")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")
    print(f"Total images: {sum(counts.values())}")

    plot_class_counts(counts)
    plot_sample_images(counts)
