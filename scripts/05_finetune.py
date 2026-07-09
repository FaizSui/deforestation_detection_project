"""
Step 5: Fine-tune the whole network (optional stage 2).

What this script does, in plain English:
1. Loads the stage-1 weights we already trained (frozen backbone, final
   layer only) from ./models/resnet50_stage1.pth
2. Unfreezes EVERY layer in the network, so all of ResNet50's weights,
   not just the final layer, can now be adjusted.
3. Continues training for up to 8 more epochs, but with a learning rate
   10x lower (0.0001 vs 0.001) than stage 1. This is intentional: the
   pretrained features are already good, so we want tiny nudges to adapt
   them to satellite imagery, not big changes that could wreck them.
4. Watches for overfitting: if validation accuracy stops improving while
   training loss keeps dropping, that's the signal the model is starting
   to memorize the training set rather than generalize. We print a clear
   warning when that pattern shows up, and save the BEST epoch's weights
   (by validation accuracy), not just whatever the last epoch happened to be.
5. Saves the best weights to ./models/resnet50_finetuned.pth

Same train/val/test split as before (same seed), so nothing here touches
the test set -- that stays untouched until final evaluation.
"""

import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "EuroSAT_RGB")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
STAGE1_PATH = os.path.join(MODELS_DIR, "resnet50_stage1.pth")
FINETUNED_PATH = os.path.join(MODELS_DIR, "resnet50_finetuned.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.0001
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

    full_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_eval = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)

    n = len(full_train)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, _, _ = random_split(full_train, [train_size, val_size, test_size], generator=generator)

    generator = torch.Generator().manual_seed(SEED)
    _, val_subset, _ = random_split(full_eval, [train_size, val_size, test_size], generator=generator)

    return train_subset, val_subset, full_train.classes


def build_model(num_classes, device):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(STAGE1_PATH, map_location=device))
    return model.to(device)


def finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_set, val_set, classes = build_datasets()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(classes), device=device)

    # Unfreeze everything -- this is the key difference from stage 1.
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    best_epoch = -1
    best_state_dict = None
    prev_val_accuracy = None
    prev_loss = None

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

        avg_loss = running_loss / len(train_loader)

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

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.4f} | Time: {elapsed:.1f}s")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

        if prev_val_accuracy is not None and prev_loss is not None:
            if avg_loss < prev_loss and val_accuracy < prev_val_accuracy:
                print(f"  Note: training loss dropped but validation accuracy also dropped "
                      f"this epoch. Possible early sign of overfitting -- watch the next "
                      f"couple epochs.")

        prev_val_accuracy = val_accuracy
        prev_loss = avg_loss

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(best_state_dict, FINETUNED_PATH)
    print(f"\nBest epoch was {best_epoch} with val accuracy {best_val_accuracy:.4f}")
    print(f"Saved best fine-tuned weights to {FINETUNED_PATH}")


if __name__ == "__main__":
    finetune()
