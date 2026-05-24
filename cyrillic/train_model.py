import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
VAL_RATIO = 0.2
IMAGE_SIZE = 64

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda")
base_dir = Path(__file__).resolve().parent


class CyrillicMNISTDataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.files = []

        for label_idx, label_name in enumerate(self.classes):
            for img_path in (self.root_dir / label_name).glob("*.png"):
                self.files.append((img_path, label_idx))

        if indices is not None:
            self.files = [self.files[i] for i in indices]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = Image.open(img_path).split()[-1]

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
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def split_indices(files, val_ratio):
    class_indices = {}

    for i, (_, label) in enumerate(files):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

    train_indices = []
    val_indices = []

    for indices in class_indices.values():
        random.shuffle(indices)
        split = int(len(indices) * (1 - val_ratio))
        train_indices += indices[:split]
        val_indices += indices[split:]

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return train_indices, val_indices


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(8),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def main():
    full_dataset = CyrillicMNISTDataset(base_dir / "cyrillic")
    train_indices, val_indices = split_indices(full_dataset.files, VAL_RATIO)

    train_dataset = CyrillicMNISTDataset(base_dir / "cyrillic", train_indices, train_transform)
    val_dataset = CyrillicMNISTDataset(base_dir / "cyrillic", val_indices, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CyrillicCNN(len(full_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_acc = 0
    model_path = base_dir / "model.pth"
    plot_path = base_dir / "train.png"

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step()

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"train_loss={train_loss} | "
            f"train_acc={train_acc} | "
            f"val_loss={val_loss} | "
            f"val_acc={val_acc}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": full_dataset.classes,
                },
                model_path,
            )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_loss_history, label="train")
    plt.plot(val_loss_history, label="val")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_acc_history, label="train")
    plt.plot(val_acc_history, label="val")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Best val accuracy: {best_val_acc}")


if __name__ == "__main__":
    main()
