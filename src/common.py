
from pathlib import Path
import json
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "ecosort_mobilenetv2.pth"
CLASSES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = 160
DEVICE = torch.device("cpu")


def get_transforms(training=False):
    if training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def save_class_names(class_names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASSES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def load_class_names():
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
