import os, shutil, random, copy, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# CONFIG
ARCHIVE, EXTRACT_DIR, DATA_ROOT, RES_DIR = "shapes.zip", "extracted", "dataset_prepared", "results"
SPLITS, CLASSES = ["train", "val", "test"], ["circle", "square", "triangle"]
IMG_SIZE, BATCH, EPOCHS, LR = 224, 32, 10, 1e-4
MODEL_NAMES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def prepare_data():
    if os.path.exists(DATA_ROOT): return
    os.makedirs(DATA_ROOT, exist_ok=True)
    if not os.path.exists(EXTRACT_DIR):
        import zipfile
        with zipfile.ZipFile(ARCHIVE) as z: z.extractall(EXTRACT_DIR)
    for split in SPLITS:
        for cls in CLASSES:
            dst = os.path.join(DATA_ROOT, split, cls)
            os.makedirs(dst, exist_ok=True)
            src = os.path.join(EXTRACT_DIR, split, cls, "images")
            if os.path.exists(src):
                for f in os.listdir(src):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                        shutil.copy2(os.path.join(src, f), dst)

def get_loaders():
    tr_t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10), transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    ev_t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    loaders = {}
    for s, t in [("train", tr_t), ("val", ev_t), ("test", ev_t)]:
        ds = datasets.ImageFolder(os.path.join(DATA_ROOT, s), transform=t)
        loaders[s] = DataLoader(ds, batch_size=BATCH, shuffle=(s=="train"), num_workers=2, pin_memory=True)
    return loaders, ds.classes

def build_model(name, n_cls):
    # torchvision >= 0.13 поддерживает строковые веса. Для старых версий используйте getattr(models, name)(weights=models.get_weight(name+"_weights.IMAGENET1K_V1"))
    m = models.get_model(name, weights="IMAGENET1K_V1")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    return m.to(DEVICE)

def run_epoch(model, loader, criterion, opt=None):
    model.train(opt is not None)
    loss, acc, tot, lbls, preds = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            l = criterion(out, y)
            if opt: opt.zero_grad(); l.backward(); opt.step()
            loss += l.item() * len(y)
            p = out.argmax(1)
            acc += (p == y).sum().item()
            tot += len(y)
            lbls.extend(y.cpu().numpy()); preds.extend(p.cpu().numpy())
    return loss/tot, acc/tot, np.array(lbls), np.array(preds)

def experiment(name, loaders, classes):
    model = build_model(name, len(classes))
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    best_w, best_acc = None, 0.0

    for ep in range(1, EPOCHS+1):
        tr_l, tr_a, _, _ = run_epoch(model, loaders["train"], crit, opt)
        va_l, va_a, _, _ = run_epoch(model, loaders["val"], crit)
        if va_a > best_acc: best_acc, best_w = va_a, copy.deepcopy(model.state_dict())
        print(f"[{name}] Ep {ep}/{EPOCHS} | tr:{tr_a:.3f} va:{va_a:.3f}")

    model.load_state_dict(best_w)
    te_l, te_a, y_true, y_pred = run_epoch(model, loaders["test"], crit)

    os.makedirs(RES_DIR, exist_ok=True)
    torch.save(best_w, f"{RES_DIR}/{name}.pth")
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(cmap="Blues", colorbar=False)
    plt.title(f"CM - {name}"); plt.tight_layout(); plt.savefig(f"{RES_DIR}/{name}_cm.png", dpi=200); plt.close()

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(f"{RES_DIR}/{name}_report.txt", "w") as f:
        f.write(f"Model: {name}\nBest Val Acc: {best_acc:.4f}\nTest Acc: {te_a:.4f}\n\n{report}")
    return {"name": name, "best_val": best_acc, "test_acc": te_a}

def main():
    set_seed()
    prepare_data()
    loaders, classes = get_loaders()
    results = [experiment(m, loaders, classes) for m in MODEL_NAMES]
    best = max(results, key=lambda x: x["test_acc"])
    with open(f"{RES_DIR}/summary.txt", "w") as f:
        for r in results: f.write(f"{r['name']}: val={r['best_val']:.4f} test={r['test_acc']:.4f}\n")
        f.write(f"\nBest: {best['name']} (test={best['test_acc']:.4f})")
    print(f"\nDone. Best: {best['name']} ({best['test_acc']:.4f})")

if __name__ == "__main__":
    main()