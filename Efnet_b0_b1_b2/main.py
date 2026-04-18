import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models

# Config
CLASS_NAMES = ["circle", "square", "triangle"]
MODEL_PATH = "results/efficientnet_b0.pth"
TEST_DIR = "dataset_prepared/test"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CLASS = 3


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(model, image, transform, classes, device):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        idx = torch.argmax(probs, dim=1).item()
        return classes[idx], probs[0, idx].item()


def main():
    transform = get_eval_transform(IMAGE_SIZE)
    
    model = build_model("efficientnet_b0", len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    # Collect samples
    samples = []
    for cls in CLASS_NAMES:
        folder = os.path.join(TEST_DIR, cls)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            for fname in random.sample(files, min(SAMPLES_PER_CLASS, len(files))):
                samples.append((os.path.join(folder, fname), cls))

    if not samples:
        print("No images found")
        return

    # Plot
    n = len(samples)
    cols, rows = 3, (n + 2) // 3
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, (path, true_label) in enumerate(samples, 1):
        img = Image.open(path).convert("RGB")
        pred_label, conf = predict(model, img, transform, CLASS_NAMES, DEVICE)
        
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        color = "green" if pred_label == true_label else "red"
        plt.title(f"{true_label} → {pred_label}\n{conf:.2f}", color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()