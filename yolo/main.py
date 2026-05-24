from pathlib import Path
import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "yolo" / "weights" / "best.pt"

model = YOLO(str(model_path))

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_EXPOSURE, -4)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            label = model.names[cls_id]
            if label == "neither":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()
