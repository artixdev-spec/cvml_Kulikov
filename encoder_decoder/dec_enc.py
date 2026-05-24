import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import torch.optim as optim
import random
import string
from pathlib import Path


class ImageDataset(Dataset):

    def __init__(self, n=200, size=128, data_variant=1):
        super().__init__()
        self.n = n
        self.size = size
        self.data_variant = data_variant
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        letters = string.ascii_letters + string.digits

        if self.data_variant==1:
            text = "ABC"
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)

        elif self.data_variant==2:
            text = ''.join(random.choice(letters) for _ in range(3))
            x = 30
            y = 30

        elif self.data_variant == 3:
            text_len = random.randint(1, 8)
            text = ''.join(random.choice(letters) for _ in range(text_len))
            x = 30
            y = 30

        elif self.data_variant == 4:
            text_len = random.randint(1, 8)
            text = ''.join(random.choice(letters) for _ in range(text_len))
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor


class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_size=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    for data_variant in range(1, 5):
        print(f"\nTraining data variant {data_variant}")

        ds = ImageDataset(2000, 256, data_variant=data_variant)
        dataloader = DataLoader(ds, batch_size=32, shuffle=True)

        encoder = Encoder()
        decoder = Decoder()

        ep = sum(p.numel() for p in encoder.parameters())
        dp = sum(p.numel() for p in decoder.parameters())

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(ep, dp, device)

        encoder.to(device)
        decoder.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) +
                               list(decoder.parameters()))

        encoder.train()
        decoder.train()
        epochs = 10

        for epoch in range(epochs):
            epoch_loss = 0.0

            for imgs, _ in dataloader:
                imgs = imgs.to(device)

                optimizer.zero_grad()

                latent = encoder(imgs)
                output = decoder(latent)

                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"data_variant={data_variant}, epoch={epoch}, avg_loss={avg_loss:.6f}")

        torch.save(encoder.state_dict(), BASE_DIR / f"encoder_{data_variant}.pth")
        torch.save(decoder.state_dict(), BASE_DIR / f"decoder_{data_variant}.pth")