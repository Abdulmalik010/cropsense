"""Fine-tune MobileNetV3 on CropSense class folders."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_label_catalog, load_settings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main() -> None:
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tqdm import tqdm

    class OrderedImageFolder(datasets.ImageFolder):
        """Keep folder labels in models/labels.json order, not alphabetical."""

        def find_classes(self, directory):
            return class_names, {name: index for index, name in enumerate(class_names)}


    settings = load_settings()
    catalog = load_label_catalog(settings.labels_path)
    class_names = [item["id"] for item in catalog["classes"]]
    model_name = catalog.get("model_name", "mobilenetv3_small_100")
    data_root = ROOT / "data" / "processed"
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists():
        raise SystemExit(
            f"Missing {train_dir}. Run scripts/download_data.py and add images first."
        )

    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_ds = OrderedImageFolder(str(train_dir), transform=train_tf)
    val_ds = OrderedImageFolder(str(val_dir), transform=val_tf) if val_dir.exists() else None
    if train_ds.classes != class_names:
        missing = [name for name in class_names if name not in train_ds.classes]
        extra = [name for name in train_ds.classes if name not in class_names]
        raise SystemExit(
            "ImageFolder class names must match models/labels.json order and ids.\n"
            f"Expected: {class_names}\nFound: {train_ds.classes}\n"
            f"Missing: {missing}\nExtra: {extra}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    model.to(device)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0) if val_ds else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, targets in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
        print(f"train loss: {running / len(train_ds):.4f}")
        if val_loader:
            _evaluate(model, val_loader, device, class_names)

    settings.classifier_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "class_names": class_names,
            "state_dict": model.state_dict(),
        },
        settings.classifier_weights,
    )
    print(f"Wrote {settings.classifier_weights}")


def _evaluate(model, loader, device, class_names: list[str]) -> None:
    import torch
    from sklearn.metrics import classification_report

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(targets.tolist())
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
