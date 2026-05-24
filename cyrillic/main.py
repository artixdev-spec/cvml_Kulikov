import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_model import (
    BATCH_SIZE,
    SEED,
    VAL_RATIO,
    CyrillicCNN,
    CyrillicMNISTDataset,
    base_dir,
    split_indices,
    val_transform,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    full_dataset = CyrillicMNISTDataset(base_dir / "cyrillic")
    _, val_indices = split_indices(full_dataset.files, VAL_RATIO)

    val_dataset = CyrillicMNISTDataset(
        base_dir / "cyrillic",
        val_indices,
        val_transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    checkpoint = torch.load(base_dir / "model.pth", map_location=device)

    model = CyrillicCNN(len(checkpoint["classes"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images, labels = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        predicted = model(images).argmax(dim=1)

    num_images = min(10, len(images))

    plt.figure(figsize=(15, 6))

    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].cpu().squeeze(0), cmap="gray")
        plt.title(
            f"T: {val_dataset.classes[labels[i].item()]}\n"
            f"P: {val_dataset.classes[predicted[i].item()]}",
            fontsize=10,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
