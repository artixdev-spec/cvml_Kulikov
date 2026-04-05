import argparse
import json
import math
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


ALLOWED_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}
IMAGE_SIZE = 32


class CyrillicDataset(Dataset):
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
        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_SUFFIXES:
                    self.samples.append((file_path, self.class_to_idx[class_name]))

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



def load_classes(classes_path):
    classes_file = Path(classes_path)
    if not classes_file.exists():
        raise FileNotFoundError(
            f'Не найден файл классов: {classes_file}. '
            'Сначала обучите модель в train_model.py.'
        )
    with open(classes_file, 'r', encoding='utf-8') as f:
        return json.load(f)



def build_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])



def show_random_predictions(model, dataset, class_names, device='cpu', n=9):
    model.eval()
    model.to(device)

    n = min(n, len(dataset))
    indices = torch.randperm(len(dataset))[:n]
    grid_size = math.ceil(n ** 0.5)

    plt.figure(figsize=(10, 10))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx.item()]
            output = model(image.unsqueeze(0).to(device))
            pred_label = output.argmax(dim=1).item()

            img = image.squeeze().cpu().numpy()
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(
                f'true: {class_names[true_label]}\npred: {class_names[pred_label]}',
                color='green' if pred_label == true_label else 'red'
            )

    plt.tight_layout()
    plt.show()



def predict_single_image(model, image_path, transform, class_names, device='cpu'):
    img = Image.open(image_path).convert('RGBA')
    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(background, img).convert('L')

    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Предсказание: {class_names[pred]}')
    plt.show()

    print(f'Предсказанный класс: {class_names[pred]}')



def parse_args():
    parser = argparse.ArgumentParser(description='Демонстрация работы модели классификации кириллических символов.')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Путь к файлу с весами модели.')
    parser.add_argument('--classes_path', type=str, default='classes.json', help='Путь к JSON со списком классов.')
    parser.add_argument('--zip_path', type=str, default='cyrillic.zip', help='Путь к архиву с датасетом.')
    parser.add_argument('--extract_path', type=str, default='cyril', help='Папка для распаковки архива.')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='Размер изображения после Resize.')
    parser.add_argument('--n', type=int, default=9, help='Сколько случайных примеров показать.')
    parser.add_argument('--image_path', type=str, default=None, help='Путь к отдельному изображению для предсказания.')
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    class_names = load_classes(args.classes_path)
    transform = build_transform(args.image_size)

    model = CyrillicCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.image_path:
        predict_single_image(model, args.image_path, transform, class_names, device=device)
        return

    dataset = CyrillicDataset(
        zip_path=args.zip_path,
        extract_path=args.extract_path,
        transform=transform,
    )
    show_random_predictions(model, dataset, class_names, device=device, n=args.n)


if __name__ == '__main__':
    main()
