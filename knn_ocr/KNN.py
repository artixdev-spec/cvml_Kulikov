
import os
import cv2
import numpy as np


IMG_SIZE = 32
K = 3


def get_base_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def is_image_file(name: str) -> bool:
    name = name.lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def numeric_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def decode_label(label: str) -> str:
    if len(label) == 2 and label.startswith("s"):
        return label[1]
    return label


def preprocess_binary(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Не удалось прочитать изображение")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Нужны белые символы на черном фоне
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = 255 - binary

    binary = cv2.medianBlur(binary, 3)
    return binary


def crop_to_content(binary: np.ndarray) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return binary[y1:y2 + 1, x1:x2 + 1]


def normalize_symbol(binary: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    symbol = crop_to_content(binary)
    h, w = symbol.shape

    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)

    scale = min((size - 4) / w, (size - 4) / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(symbol, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def extract_features(binary: np.ndarray) -> np.ndarray:
    return normalize_symbol(binary, IMG_SIZE).flatten().astype(np.float32) / 255.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


class MyKNN:
    def __init__(self, k: int = 3):
        if k <= 0:
            raise ValueError("k должно быть > 0")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y):
        if len(X) != len(y):
            raise ValueError("Размеры X и y не совпадают")
        self.X_train = X
        self.y_train = np.array(y)

    def predict_one(self, x: np.ndarray) -> str:
        distances = []

        for train_x, label in zip(self.X_train, self.y_train):
            dist = euclidean_distance(x, train_x)
            distances.append((dist, label))

        distances.sort(key=lambda item: item[0])
        neighbors = distances[:self.k]

        votes = {}
        for dist, label in neighbors:
            weight = 1.0 / (dist + 1e-8)
            votes[label] = votes.get(label, 0.0) + weight

        return max(votes.items(), key=lambda item: item[1])[0]


def estimate_angle(binary: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary > 0))
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    w, h = rect[1]

    if w < h:
        angle += 90.0

    return angle


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    border_value = (0, 0, 0) if img.ndim == 3 else 0
    return cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=border_value)


def load_train_data(train_dir: str):
    X = []
    y = []

    for label in sorted(os.listdir(train_dir)):
        label_dir = os.path.join(train_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for filename in sorted(os.listdir(label_dir)):
            if not is_image_file(filename):
                continue

            path = os.path.join(label_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            binary = preprocess_binary(img)

            X.append(extract_features(binary))
            y.append(label)

    if not X:
        raise ValueError("Обучающие изображения не найдены")

    return np.array(X), y


def get_component_boxes(binary: np.ndarray):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 10:
            continue
        boxes.append((int(x), int(y), int(w), int(h), int(area)))

    boxes.sort(key=lambda box: box[0])
    return boxes


def merge_two_part_symbols(boxes):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: box[0])
    merged = []
    used = [False] * len(boxes)

    for i, box in enumerate(boxes):
        if used[i]:
            continue

        x, y, w, h, area = box
        used[i] = True
        changed = True

        while changed:
            changed = False

            for j, other in enumerate(boxes):
                if used[j]:
                    continue

                x2, y2, w2, h2, area2 = other

                overlap_x = min(x + w, x2 + w2) - max(x, x2)
                center_close = abs((x + w / 2) - (x2 + w2 / 2)) <= max(w, w2) * 0.6

                # Склеиваем компоненты одного символа, например точку и палочку у 'i'
                if center_close and overlap_x > -2:
                    nx1 = min(x, x2)
                    ny1 = min(y, y2)
                    nx2 = max(x + w, x2 + w2)
                    ny2 = max(y + h, y2 + h2)

                    x, y, w, h = nx1, ny1, nx2 - nx1, ny2 - ny1
                    area += area2
                    used[j] = True
                    changed = True

        merged.append((x, y, w, h, area))

    merged.sort(key=lambda box: box[0])
    return merged


def segment_symbols(img: np.ndarray):
    binary = preprocess_binary(img)
    angle = estimate_angle(binary)
    rotated = rotate_image(img, angle)

    rotated_binary = preprocess_binary(rotated)
    boxes = get_component_boxes(rotated_binary)
    boxes = merge_two_part_symbols(boxes)

    symbols = []
    clean_boxes = []

    for x, y, w, h, _ in boxes:
        crop = rotated_binary[y:y + h, x:x + w]
        crop = crop_to_content(crop)
        if crop.size == 0:
            continue

        symbols.append(crop)
        clean_boxes.append((x, y, w, h))

    return symbols, clean_boxes


def find_space_positions(boxes):
    if len(boxes) < 2:
        return set()

    gaps = []
    for i in range(len(boxes) - 1):
        x1, _, w1, _ = boxes[i]
        x2, _, _, _ = boxes[i + 1]
        gaps.append(x2 - (x1 + w1))

    values = np.array(gaps, dtype=np.float32)
    if len(values) < 2:
        return set()

    sorted_values = np.sort(values)
    diffs = np.diff(sorted_values)

    if len(diffs) == 0:
        return set()

    best_idx = int(np.argmax(diffs))
    best_jump = float(diffs[best_idx])

    # Нет выраженного разрыва -> пробелы не вставляем
    if best_jump < max(8.0, 0.35 * float(np.median(values))):
        return set()

    threshold = float((sorted_values[best_idx] + sorted_values[best_idx + 1]) / 2.0)
    return {i for i, gap in enumerate(gaps) if gap > threshold}


def reconstruct_text(knn: MyKNN, symbols, boxes) -> str:
    if not symbols:
        return ""

    preds = []
    for symbol in symbols:
        feat = extract_features(symbol)
        label = knn.predict_one(feat)
        preds.append(decode_label(label))

    spaces_after = find_space_positions(boxes)

    result = []
    for i, pred in enumerate(preds):
        result.append(pred)
        if i in spaces_after:
            result.append(" ")

    return "".join(result)


def find_task_dir(base_dir: str) -> str:
    direct = os.path.join(base_dir, "task")
    if os.path.isdir(direct):
        return direct

    for root, dirs, _ in os.walk(base_dir):
        if "task" in dirs:
            return os.path.join(root, "task")

    raise FileNotFoundError("Папка task не найдена")


def main():
    base_dir = get_base_dir()
    task_dir = find_task_dir(base_dir)

    train_dir = os.path.join(task_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError("Папка train не найдена")

    X_train, y_train = load_train_data(train_dir)

    k = min(K, len(X_train))
    if k % 2 == 0 and k > 1:
        k -= 1

    knn = MyKNN(k=k)
    knn.fit(X_train, y_train)

    test_images = []
    for name in os.listdir(task_dir):
        path = os.path.join(task_dir, name)
        if os.path.isfile(path) and is_image_file(name):
            test_images.append(path)

    test_images.sort(key=numeric_key)

    for image_path in test_images:
        img = cv2.imread(image_path)
        symbols, boxes = segment_symbols(img)
        text = reconstruct_text(knn, symbols, boxes)

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        print(f"{image_id} {text}")


main()
