"""
Step 3: Train a ResNet50 classifier on EuroSAT using transfer learning.

What this script does, in plain English:
1. Loads all 27,000 images from disk and splits them 80/10/10 into
   train/validation/test sets.
2. Resizes every image to 224x224 and normalizes pixel values using the
   exact mean/std that ImageNet was trained with (required for the
   pretrained weights to behave correctly).
3. Loads ResNet50 with its pretrained ImageNet weights, freezes every
   layer except the final one, and replaces that final layer so it
   outputs 10 classes instead of 1,000.
4. Trains only that final layer for 10 epochs, printing training loss and
   validation accuracy after each epoch so we can see it actually learning.
5. Saves the trained weights to ./models/resnet50_stage1.pth

This is "stage 1" training (frozen backbone). Once we confirm this works
and look at the results together, the optional next step (per the guide)
is unfreezing the whole model for a lower-learning-rate fine-tune pass.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "EuroSAT_RGB")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42


def build_datasets():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # We load the dataset twice, once with each transform, because
    # random_split just picks *indices* -- it doesn't let different
    # subsets use different transforms on their own. Loading twice with a
    # fixed random seed guarantees both loads split into the same indices,
    # so we can safely combine "train indices + augmenting transform" with
    # "val/test indices + non-augmenting transform".
    full_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_eval = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)

    n = len(full_train)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, _, _ = random_split(full_train, [train_size, val_size, test_size], generator=generator)

    generator = torch.Generator().manual_seed(SEED)
    _, val_subset, test_subset = random_split(full_eval, [train_size, val_size, test_size], generator=generator)

    return train_subset, val_subset, test_subset, full_train.classes


def build_model(num_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(device)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_set, val_set, test_set, classes = build_datasets()
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(classes), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Accuracy: {val_accuracy:.4f} | Time: {elapsed:.1f}s")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "resnet50_stage1.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model weights to {save_path}")


if __name__ == "__main__":
    train()
