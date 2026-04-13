import os
import time
from collections import deque

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

MODEL_PATH = "efficientnet_person_detector.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )
    return model


model = build_model().to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 1e-3,
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class Buffer:
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames)).to(DEVICE)
        labels = torch.tensor(list(self.labels), dtype=torch.float32, device=DEVICE)
        return images, labels

    def clear(self):
        self.frames.clear()
        self.labels.clear()


def train_on_buffer(buffer):
    if len(buffer) < 10:
        return None

    model.train()
    images, labels = buffer.get_batch()

    for _ in range(10):  
        optimizer.zero_grad()
        predictions = model(images).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    return loss.item()


def predict(frame):
    model.eval()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor).squeeze()
        prob = torch.sigmoid(logits).item()

    label = "person" if prob > 0.5 else "no_person"
    return label, prob


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

    buffer = Buffer()
    prediction_text = "Press P to predict"

    print("Управление:")
    print("1 - сохранить кадр как person")
    print("2 - сохранить кадр как no_person")
    print("p - предсказать")
    print("s - сохранить модель")
    print("q - выйти")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Не удалось прочитать кадр с камеры")
            break

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            prediction_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display_frame,
            f"Buffer: {len(buffer)}/{buffer.frames.maxlen}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Camera", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("1"):
            tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer.append(tensor, 1.0)
            print(f"Added: person | buffer size = {len(buffer)}")

        elif key == ord("2"):
            tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer.append(tensor, 0.0)
            print(f"Added: no_person | buffer size = {len(buffer)}")

        elif key == ord("p"):
            t0 = time.perf_counter()
            label, confidence = predict(frame)
            dt = time.perf_counter() - t0
            prediction_text = f"{label} | prob={confidence:.4f} | {dt:.3f}s"
            print(f"Prediction: {label} | prob={confidence:.4f} | elapsed={dt:.4f}s")

        elif key == ord("s"):
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved")

        if len(buffer) >= buffer.frames.maxlen:
            loss = train_on_buffer(buffer)
            if loss is not None:
                print(f"Training loss = {loss:.4f}")
                prediction_text = f"Last train loss = {loss:.4f}"
            buffer.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()