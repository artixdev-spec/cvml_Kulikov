from PIL import Image
import torch
from torchvision import transforms

from model import SimpleCNN


def load_image_correctly(img_path):
    image = Image.open(img_path)

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("L")
    else:
        image = image.convert("L")

    return image


def predict(img_path, model_path="best_cyrillic_cnn.pth"):
    checkpoint = torch.load(model_path, map_location="cpu")

    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    image = load_image_correctly(img_path)
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(dim=1).item()

    print("Predicted class:", classes[pred_idx])


if __name__ == "__main__":
    predict("test.png")