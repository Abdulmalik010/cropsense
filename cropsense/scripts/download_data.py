"""Prepare image folders for the CropSense classifier.

PlantVillage (tomato, maize, pepper) and iCassava are the intended public
sources. This script creates the class folders CropSense expects and, if
Kaggle credentials exist, tries to download those competitions/datasets.

Without Kaggle, copy images yourself into:

  data/processed/train/<class_id>/*.jpg
  data/processed/val/<class_id>/*.jpg

Class ids are listed in models/labels.json.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import class_ids, load_settings

SOURCE_HINTS = {
    "Tomato___Early_blight": ["Tomato___Early_blight"],
    "Tomato___Late_blight": ["Tomato___Late_blight"],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ],
    "Tomato___healthy": ["Tomato___healthy"],
    "Maize___Northern_Leaf_Blight": [
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___Northern_Leaf_Blight",
    ],
    "Maize___Common_rust": ["Corn_(maize)___Common_rust_", "Corn_(maize)___Common_rust"],
    "Maize___Gray_leaf_spot": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    ],
    "Maize___healthy": ["Corn_(maize)___healthy"],
    "Pepper___Bacterial_spot": ["Pepper,_bell___Bacterial_spot"],
    "Pepper___healthy": ["Pepper,_bell___healthy"],
    "Cassava___mosaic_disease": ["cmd", "CMD", "cassava_mosaic"],
    "Cassava___brown_streak_disease": ["cbsd", "CBSD"],
    "Cassava___bacterial_blight": ["cbb", "CBB"],
    "Cassava___healthy": ["healthy", "Healthy"],
}


def _make_folders(processed: Path, labels: list[str]) -> None:
    for split in ("train", "val"):
        for label in labels:
            (processed / split / label).mkdir(parents=True, exist_ok=True)


def _split_copy(files: list[Path], dest_train: Path, dest_val: Path, val_ratio: float = 0.2) -> None:
    rng = random.Random(42)
    shuffled = files[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 5 else max(0, len(shuffled) // 5)
    val_files = shuffled[:cut]
    train_files = shuffled[cut:] or shuffled
    for path in train_files:
        shutil.copy2(path, dest_train / path.name)
    for path in val_files:
        shutil.copy2(path, dest_val / path.name)


def ingest_existing_tree(raw_dir: Path, processed: Path, labels: list[str]) -> int:
    """Copy images from a PlantVillage-like folder tree into processed splits."""
    copied = 0
    for label in labels:
        matches: list[Path] = []
        for hint in SOURCE_HINTS.get(label, [label]):
            matches.extend(raw_dir.rglob(hint))
        image_files: list[Path] = []
        for match in matches:
            if match.is_dir():
                image_files.extend(
                    [
                        path
                        for path in match.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    ]
                )
        if not image_files:
            continue
        _split_copy(
            image_files,
            processed / "train" / label,
            processed / "val" / label,
        )
        copied += len(image_files)
        print(f"  {label}: {len(image_files)} images")
    return copied


def try_kaggle_download(raw_dir: Path) -> None:
    """Download PlantVillage / cassava datasets when the Kaggle CLI is configured."""
    kaggle = shutil.which("kaggle")
    if not kaggle:
        print("Kaggle CLI not found. Skipping automatic download.")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [kaggle, "datasets", "download", "-d", "emmarex/plantdisease", "-p", str(raw_dir), "--unzip"],
    ]
    for command in commands:
        print("Running:", " ".join(command))
        try:
            import subprocess

            subprocess.run(command, check=False)
        except OSError as exc:
            print(f"Kaggle download failed: {exc}")


def write_readme(processed: Path, labels: list[str]) -> None:
    guide = processed / "README.md"
    listing = "\n".join(f"- `{name}`" for name in labels)
    guide.write_text(
        "# Image folders\n\n"
        "Put JPEG/PNG files in `train/<class_id>/` and `val/<class_id>/`.\n\n"
        "Suggested sources:\n"
        "- PlantVillage (tomato, maize/corn, bell pepper)\n"
        "- iCassava 2019 / cassava leaf disease (cassava classes)\n\n"
        "Class ids:\n"
        f"{listing}\n",
        encoding="utf-8",
    )


def main() -> None:
    settings = load_settings()
    labels = class_ids()
    raw_dir = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    _make_folders(processed, labels)
    write_readme(processed, labels)
    try_kaggle_download(raw_dir)
    copied = ingest_existing_tree(raw_dir, processed, labels)
    print(f"Prepared folders under {processed}")
    if copied == 0:
        print(
            "No images were copied. Place a PlantVillage (or similar) folder tree "
            f"inside {raw_dir} and run this script again, or copy files by hand."
        )
    print(json.dumps({"classes": labels, "copied_images": copied}, indent=2))


if __name__ == "__main__":
    main()
