
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from common import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
    DEVICE,
    get_transforms,
    save_class_names,
)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    # Expected: data/raw/cardboard, glass, metal, paper, plastic, trash
    train_base = datasets.ImageFolder(
        DATA_DIR, transform=get_transforms(training=True)
    )
    val_base = datasets.ImageFolder(
        DATA_DIR, transform=get_transforms(training=False)
    )

    if len(train_base.classes) < 2:
        raise ValueError(
            "Add images in at least two class folders inside data/raw/."
        )

    if len(train_base) < 10:
        raise ValueError("Dataset is too small. Add more images before training.")

    if train_base.classes != val_base.classes:
        raise RuntimeError("Dataset class folders changed during setup.")

    class_names = train_base.classes
    save_class_names(class_names)

    # Stratified split: keep class proportions similar in both sets.
    targets = np.array(train_base.targets)
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []

    for class_id in range(len(class_names)):
        indices = np.where(targets == class_id)[0]
        rng.shuffle(indices)

        if len(indices) < 2:
            raise ValueError(
                f"Class '{class_names[class_id]}' needs at least 2 images."
            )

        val_count = max(1, round(len(indices) * 0.2))
        val_count = min(val_count, len(indices) - 1)

        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    train_data = Subset(train_base, train_indices)
    val_data = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ImageNet pretrained weights may download on the first run.
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained MobileNetV2 weights. "
            "Check your internet connection and try again."
        ) from exc

    # Freeze the pretrained feature extractor; train the new classifier.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.last_channel, len(class_names)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total
        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.1%} | "
            f"validation loss: {val_loss:.4f} | "
            f"validation accuracy: {val_accuracy:.1%}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)

    with open(MODEL_DIR / "training_summary.json", "w") as file:
        json.dump(
            {
                "classes": class_names,
                "image_size": IMAGE_SIZE,
                "epochs": EPOCHS,
                "best_validation_accuracy": best_val_accuracy,
                "training_images": len(train_data),
                "validation_images": len(val_data),
            },
            file,
            indent=2,
        )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Best validation accuracy: {best_val_accuracy:.1%}")


if __name__ == "__main__":
    main()
