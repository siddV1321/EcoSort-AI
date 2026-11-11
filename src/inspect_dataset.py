from collections import Counter
from pathlib import Path
from PIL import Image
import json, sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "dataset_report.json"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR}")
        print("Create data/raw and put the six class folders inside it.")
        sys.exit(1)
    classes = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    counts, unreadable, examples = {}, [], []
    total = 0
    for folder in classes:
        imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        counts[folder.name] = len(imgs)
        total += len(imgs)
        for img in imgs:
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                unreadable.append(str(img.relative_to(ROOT)))
    result = {
        "dataset_path": str(DATA_DIR),
        "total_images": total,
        "class_counts": counts,
        "unreadable_image_count": len(unreadable),
        "unreadable_images": unreadable[:100],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nReport saved: {REPORT}")

if __name__ == "__main__":
    main()
