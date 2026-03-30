import os
import zipfile
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from model import SimpleCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CyrillicDataset(Dataset):
    IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, archive_path=None, extract_dir="./extracted_data", data_dir=None, transform=None):
        self.transform = transform
        self.extract_dir = Path(extract_dir)

        if archive_path is not None:
            self.data_dir = self._extract_archive(archive_path, self.extract_dir)
        elif data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            raise ValueError("Нужно указать archive_path или data_dir")

        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        self._find_classes_and_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"Не найдено изображений в {self.data_dir}")

    def _extract_archive(self, archive_path, extract_dir):
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Архив не найден: {archive_path}")

        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)

        candidate_dirs = [extract_dir]

        for root, dirs, _ in os.walk(extract_dir):
            for d in dirs:
                candidate_dirs.append(Path(root) / d)

        best_dir = None
        best_score = -1

        for cdir in candidate_dirs:
            if not cdir.is_dir():
                continue

            subdirs = [d for d in cdir.iterdir() if d.is_dir()]
            score = 0

            for subdir in subdirs:
                try:
                    img_count = sum(
                        1 for f in subdir.iterdir()
                        if f.is_file() and f.suffix.lower() in self.IMG_EXTENSIONS
                    )
                    if img_count > 0:
                        score += 1
                except PermissionError:
                    pass

            if score > best_score:
                best_score = score
                best_dir = cdir

        if best_dir is None:
            raise RuntimeError("Не удалось определить корневую папку датасета после распаковки")

        return best_dir

    def _find_classes_and_samples(self):
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda x: x.name)

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]

            for root, _, files in os.walk(class_dir):
                for file_name in files:
                    file_path = Path(root) / file_name
                    if file_path.suffix.lower() in self.IMG_EXTENSIONS:
                        self.samples.append((str(file_path), class_idx))

    def _load_image(self, img_path):
        image = Image.open(img_path)

        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("L")
        else:
            image = image.convert("L")

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        dataset = self.subset.dataset
        real_idx = self.subset.indices[idx]

        img_path, label = dataset.samples[real_idx]
        image = dataset._load_image(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_training_plot(train_losses, val_losses, train_accs, val_accs, output_path="train.png"):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="train_acc")
    plt.plot(epochs, val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    archive_path = "cyrillic.zip"

    image_size = 64
    batch_size = 64
    epochs = 20
    learning_rate = 1e-3
    train_split = 0.8

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(8),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    base_dataset = CyrillicDataset(
        archive_path=archive_path,
        extract_dir="./extracted_data",
        transform=None
    )

    total_size = len(base_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    train_dataset = SubsetWithTransform(train_subset, train_transform)
    val_dataset = SubsetWithTransform(val_subset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    num_classes = len(base_dataset.classes)

    print("Classes:", base_dataset.classes)
    print("Num classes:", num_classes)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": base_dataset.classes,
                    "image_size": image_size
                },
                "best_cyrillic_cnn.pth"
            )
            print(f"Лучшая модель сохранена. val_acc={best_val_acc:.4f}")

    save_training_plot(train_losses, val_losses, train_accs, val_accs, output_path="train.png")

    print(f"\nОбучение завершено.")
    print(f"Лучшая точность на валидации: {best_val_acc:.4f}")
    print("Модель сохранена в: best_cyrillic_cnn.pth")
    print("График сохранен в: train.png")


if __name__ == "__main__":
    main()