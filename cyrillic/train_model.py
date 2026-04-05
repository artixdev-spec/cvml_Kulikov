import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SEED = 42
IMAGE_SIZE = 32
ALLOWED_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}


class CyrillicDataset(Dataset):
    """Dataset that reads images from an archive extracted into Cyrillic/<class_name>/..."""

    def __init__(self, zip_path='cyrillic.zip', extract_path='cyril', transform=None):
        self.zip_path = Path(zip_path)
        self.extract_path = Path(extract_path)
        self.transform = transform
        self.samples = []

        if not self.zip_path.exists():
            raise FileNotFoundError(f'Не найден архив: {self.zip_path}')

        target = self.extract_path / 'Cyrillic'
        if not target.exists():
            self.extract_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                zf.extractall(self.extract_path)

        self.root_dir = self.extract_path / 'Cyrillic'
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f'После распаковки не найдена папка {self.root_dir}. '
                'Проверьте структуру архива.'
            )

        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_SUFFIXES:
                    self.samples.append((file_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError('В датасете не найдено ни одного изображения.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class TransformSubset(Dataset):
    """Subset-like wrapper that applies its own transform."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]

        img = Image.open(img_path).convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def stratified_split_indices(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError('Сумма долей train/val/test должна быть равна 1.')

    random.seed(seed)
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for _, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_idx.extend(indices[:train_end])
        val_idx.extend(indices[train_end:val_end])
        test_idx.extend(indices[val_end:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    return train_idx, val_idx, test_idx


class CyrillicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total



def plot_history(history, output_path='train.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Динамика функции потерь')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train accuracy')
    plt.plot(history['val_acc'], label='Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Динамика точности')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()



def save_labels(classes, output_path='classes.json'):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)



def build_transforms(image_size=IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_transform, eval_transform



def parse_args():
    parser = argparse.ArgumentParser(description='Обучение CNN для классификации кириллических символов.')
    parser.add_argument('--zip_path', type=str, default='cyrillic.zip', help='Путь к архиву с датасетом.')
    parser.add_argument('--extract_path', type=str, default='cyril', help='Папка для распаковки архива.')
    parser.add_argument('--epochs', type=int, default=15, help='Количество эпох.')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=SEED, help='Seed для воспроизводимости.')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='Размер изображения после Resize.')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Куда сохранить лучшую модель.')
    parser.add_argument('--classes_path', type=str, default='classes.json', help='Куда сохранить список классов.')
    parser.add_argument('--plot_path', type=str, default='train.png', help='Куда сохранить график обучения.')
    return parser.parse_args()



def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    train_transform, eval_transform = build_transforms(args.image_size)

    base_dataset = CyrillicDataset(
        zip_path=args.zip_path,
        extract_path=args.extract_path,
        transform=None,
    )
    print(f'Размер датасета: {len(base_dataset)}')
    print(f'Количество классов: {len(base_dataset.classes)}')

    train_idx, val_idx, test_idx = stratified_split_indices(
        base_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )

    train_dataset = TransformSubset(base_dataset, train_idx, transform=train_transform)
    val_dataset = TransformSubset(base_dataset, val_idx, transform=eval_transform)
    test_dataset = TransformSubset(base_dataset, test_idx, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CyrillicCNN(num_classes=len(base_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f'Epoch [{epoch + 1}/{args.epochs}] | '
            f'train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} | val_acc={val_acc:.4f}'
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f'Лучшая модель сохранена в {args.model_path}')

    plot_history(history, args.plot_path)
    save_labels(base_dataset.classes, args.classes_path)

    best_model = CyrillicCNN(num_classes=len(base_dataset.classes)).to(device)
    best_model.load_state_dict(torch.load(args.model_path, map_location=device))
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)

    print(f'Лучшая validation accuracy: {best_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'График обучения сохранён в {args.plot_path}')
    print(f'Список классов сохранён в {args.classes_path}')


if __name__ == '__main__':
    main()
