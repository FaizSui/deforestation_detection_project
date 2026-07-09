"""
Step 4: Evaluate the trained model on the held-out test set.

What this script does, in plain English:
1. Rebuilds the exact same train/val/test split used during training
   (same seed, same method), so the test set here is guaranteed to be the
   2,700 images the model has NEVER seen or been tuned against.
2. Loads the saved weights from ./models/resnet50_stage1.pth into a fresh
   ResNet50.
3. Runs every test image through the model and compares predictions to
   true labels.
4. Prints a classification report (precision/recall/F1 per class) and
   saves a confusion matrix figure to ./figures/confusion_matrix.png

This is the real, final number for this stage -- not validation accuracy,
which we already saw during training. Test accuracy is what actually goes
in a paper because the model's weights were never adjusted based on it.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "EuroSAT_RGB")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
SEED = 42  # must match the seed used in 03_train.py so we get the same split


def build_test_set():
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    full_eval = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)

    n = len(full_eval)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    _, _, test_subset = random_split(full_eval, [train_size, val_size, test_size], generator=generator)

    return test_subset, full_eval.classes


def build_model(num_classes, device, model_path):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def evaluate(model_path, tag):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Evaluating model: {model_path}")

    test_set, classes = build_test_set()
    print(f"Test set size: {len(test_set)}")

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(classes), device=device, model_path=model_path)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification report (test set):")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Overall test accuracy: {overall_accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({tag}, Test Accuracy: {overall_accuracy:.4f})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["stage1", "finetuned"], default="stage1")
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, f"resnet50_{args.model}.pth")
    evaluate(model_path, tag=args.model)
