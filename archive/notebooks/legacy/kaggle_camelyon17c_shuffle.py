#!/usr/bin/env python3
"""Camelyon17-C with shuffled hospital-2 patches (fixes all-negative 8k prefix).

Source-only, seed-42 BEST.pt. ~15 min T4. Do not adapt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm

from src.data.corruptions import DEFAULT_CORRUPTIONS, corrupt_tensor
from src.evaluation.classification import binary_metrics, brier, ece

SEED, N, BS = 42, 8000, 64
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TF = transforms.Compose(
    [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)


def cid(v):
    return int(v.item() if hasattr(v, "item") else v)


class HFCamelyon(Dataset):
    def __init__(self, split, centers):
        centers = {int(c) for c in centers}
        keep = [i for i, c in enumerate(split["center"]) if cid(c) in centers]
        self.ds = split.select(keep)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        return TF(img.convert("RGB")), torch.tensor(float(int(r["label"])))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=None)
        d = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.classifier = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(d, 1)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def find_ckpt():
    hits = list(Path("/kaggle/input").rglob("camelyon17_resnet18_source_s42_BEST.pt"))
    if not hits:
        raise FileNotFoundError("Add Input seed-42 BEST.pt")
    return hits[0]


def mets(y, p):
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    m = binary_metrics(y, p)
    m["ece"] = ece(y, p)
    m["brier"] = brier(y, p)
    m["n_pos"] = int(y.sum())
    m["n"] = int(len(y))
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_ckpt()
    print("ckpt", ckpt)
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    test = HFCamelyon(raw["test"], [2])
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(test), size=min(N, len(test)), replace=False)
    sub = Subset(test, idx.tolist())
    loader = DataLoader(sub, batch_size=BS, shuffle=False, num_workers=2)
    xs, ys = [], []
    for x, y in tqdm(loader, desc="load"):
        xs.append(x)
        ys.append(y.view(-1))
    X, Y = torch.cat(xs), torch.cat(ys).numpy()
    print("n", len(Y), "pos", int(Y.sum()), "neg", int((1 - Y).sum()))
    if Y.min() == Y.max():
        raise SystemExit("still one class — abort")

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m = Net().to(device)
    m.load_state_dict(sd)
    m.eval()

    rows = []
    with torch.no_grad():
        def predict(xb):
            return torch.sigmoid(m(xb.to(device))).view(-1).cpu().numpy()

        p = np.concatenate([predict(X[i : i + BS]) for i in range(0, len(X), BS)])
        clean = mets(Y, p)
        print("identity AUROC", round(clean["auroc"], 4), "F1", round(clean["f1"], 4))
        rows.append({"corruption": "identity", "severity": 0, **clean})
        for name in DEFAULT_CORRUPTIONS:
            for sev in (1, 3):
                probs = []
                for i in range(0, len(X), BS):
                    xb = corrupt_tensor(X[i : i + BS], name, sev)
                    probs.append(predict(xb))
                mm = mets(Y, np.concatenate(probs))
                rows.append({"corruption": name, "severity": int(sev), "auroc": mm["auroc"], "f1": mm["f1"], "ece": mm["ece"]})
                print(name, sev, "AUROC", round(mm["auroc"], 4), "F1", round(mm["f1"], 4))

    dest = Path("/kaggle/working/camelyon17_c_shuffled_s42.json")
    dest.write_text(json.dumps({"n": int(len(Y)), "n_pos": int(Y.sum()), "rows": rows}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
