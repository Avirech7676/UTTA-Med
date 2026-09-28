#!/usr/bin/env python3
"""Optional remaining GPU job — seed 42 BEST.pt, no retrain, no retune.

Temperature (hospital 1) · SAR · EATA-C · DLTTA · Camelyon17-C
Protocol: bn_freeze_stats, lr=1e-5, k=1. Target labels unused for T / τ / lr.
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

ROOT = Path("/kaggle/working")
if (ROOT / "src" / "tta" / "sar.py").exists():
    sys.path.insert(0, str(ROOT))
else:
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, Path("/kaggle/working/UTTA-Med")]:
        if (p / "src" / "tta" / "sar.py").exists():
            sys.path.insert(0, str(p))
            os.chdir(p)
            break

from src.calibration.temperature import TemperatureScaler  # noqa: E402
from src.data.corruptions import DEFAULT_CORRUPTIONS, corrupt_tensor  # noqa: E402
from src.evaluation.adaptation import flip_table  # noqa: E402
from src.evaluation.classification import binary_metrics, brier, ece  # noqa: E402
from src.tta.dltta import DLTTA  # noqa: E402
from src.tta.eata import EATAC  # noqa: E402
from src.tta.sar import SAR  # noqa: E402

SEED, LR, E_MARGIN, BS = 42, 1e-5, 0.4, 64
MAX_C = 8000
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TF = transforms.Compose(
    [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)


def cid(v) -> int:
    return int(v.item() if hasattr(v, "item") else v)


class HFCamelyon(Dataset):
    def __init__(self, split, centers):
        centers = {int(c) for c in centers}
        keep = [i for i, c in enumerate(split["center"]) if cid(c) in centers]
        self.ds = split.select(keep) if len(keep) < len(split) else split
        print(" kept", len(self.ds), sorted(centers))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        y = torch.tensor(float(int(r["label"])))
        return TF(img.convert("RGB")), y


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


def find_ckpt() -> Path:
    hits = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            hits.extend(root.rglob("camelyon17_resnet18_source_s42_BEST.pt"))
    if not hits:
        raise FileNotFoundError("Add Input camelyon17_resnet18_source_s42_BEST.pt")
    print("ckpt", hits[0])
    return hits[0]


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m = Net().to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def collect_logits(model, loader, device):
    zs, ys = [], []
    for x, y in tqdm(loader, desc="eval"):
        x = x.to(device)
        zs.append(model(x).view(-1).cpu())
        ys.append(y.view(-1).float().cpu())
    return torch.cat(zs), torch.cat(ys)


def metrics_from_p(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    m = binary_metrics(y, p)
    m["ece"] = ece(y, p)
    m["brier"] = brier(y, p)
    return m


def adapt_stream(adapter, source, loader, device):
    ys, pb, pa = [], [], []
    source.eval()
    for x, y in tqdm(loader, desc=adapter.__class__.__name__):
        x = x.to(device)
        with torch.no_grad():
            p0 = torch.sigmoid(source(x)).view(-1).cpu().numpy()
        p1 = adapter.adapt_batch(x).detach().cpu().numpy()
        ys.append(y.view(-1).numpy())
        pb.append(p0)
        pa.append(p1)
    y = np.concatenate(ys)
    b = np.concatenate(pb)
    a = np.concatenate(pa)
    return {
        "before": metrics_from_p(y, b),
        "after": metrics_from_p(y, a),
        "harm": flip_table(y, b, a),
        "coverage": float(getattr(adapter, "coverage", 1.0)),
        "n_reset": int(getattr(adapter, "n_reset", 0)),
        "n_skipped": int(getattr(adapter, "n_skipped", 0)),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    ckpt = find_ckpt()
    print("loading HuggingFace Camelyon17-WILDS", flush=True)
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    val = HFCamelyon(raw["validation"], [1])
    test = HFCamelyon(raw["test"], [2])
    val_loader = DataLoader(val, batch_size=BS, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=BS, shuffle=False, num_workers=2)

    source = load_model(ckpt, device)
    z_val, y_val = collect_logits(source, val_loader, device)
    sc = TemperatureScaler().fit(z_val, y_val)
    print("fitted T on hospital 1:", sc.temperature, flush=True)

    z_test, y_test_t = collect_logits(source, test_loader, device)
    y_np = y_test_t.numpy()
    p_raw = torch.sigmoid(z_test).numpy()
    p_t = sc.transform_prob(z_test).numpy()
    temp_block = {
        "T": sc.temperature,
        "fit_split": "ood_val_hospital_1",
        "raw": metrics_from_p(y_np, p_raw),
        "scaled": metrics_from_p(y_np, p_t),
        "note": "AUROC invariant to T; ECE/Brier may change. Never fit T on hospital 2.",
    }
    print(
        "temp ECE raw", round(temp_block["raw"]["ece"], 4),
        "scaled", round(temp_block["scaled"]["ece"], 4),
        "AUROC", round(temp_block["raw"]["auroc"], 4),
        flush=True,
    )

    out = {
        "cuda": torch.cuda.is_available(),
        "device": str(device),
        "checkpoint": str(ckpt),
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": LR,
            "k": 1,
            "e_margin": E_MARGIN,
        },
        "temperature": temp_block,
        "source_test": temp_block["raw"],
    }

    for name, factory in [
        ("sar", lambda m: SAR(m, lr=LR, steps=1, e_margin=E_MARGIN)),
        ("eata_c", lambda m: EATAC(m, lr=LR, steps=1, e_margin=E_MARGIN)),
        ("dltta", lambda m: DLTTA(m, lr=LR, steps=1)),
    ]:
        m = load_model(ckpt, device)
        src = load_model(ckpt, device)
        adapter = factory(m)
        block = adapt_stream(adapter, src, test_loader, device)
        out[f"{name}_test"] = block
        a = block["after"]
        print(
            f"{name:8s} AUROC={a['auroc']:.4f} F1={a['f1']:.4f} ECE={a['ece']:.4f} "
            f"harm={block['harm']['harm_rate']:.4f} cov={block['coverage']:.3f}",
            flush=True,
        )

    # Camelyon17-C: source only, first MAX_C test patches
    xs, ys = [], []
    n = 0
    for x, y in test_loader:
        xs.append(x)
        ys.append(y.view(-1))
        n += x.size(0)
        if n >= MAX_C:
            break
    X = torch.cat(xs, 0)[:MAX_C]
    Y = torch.cat(ys, 0)[:MAX_C].numpy()
    src = load_model(ckpt, device)
    src.eval()
    c_rows = []
    with torch.no_grad():
        p_clean = []
        for i in range(0, len(X), BS):
            p_clean.append(torch.sigmoid(src(X[i : i + BS].to(device))).view(-1).cpu().numpy())
        p_clean = np.concatenate(p_clean)
        clean = metrics_from_p(Y, p_clean)
        print("C clean AUROC", round(clean["auroc"], 4), "n", len(Y), flush=True)
        c_rows.append({"corruption": "identity", "severity": 0, **{k: clean[k] for k in ("auroc", "f1", "ece")}})
        for name in DEFAULT_CORRUPTIONS:
            for sev in (1, 3):
                probs = []
                for i in range(0, len(X), BS):
                    xb = corrupt_tensor(X[i : i + BS], name, sev).to(device)
                    probs.append(torch.sigmoid(src(xb)).view(-1).cpu().numpy())
                p = np.concatenate(probs)
                m = metrics_from_p(Y, p)
                c_rows.append({"corruption": name, "severity": int(sev), "auroc": m["auroc"], "f1": m["f1"], "ece": m["ece"]})
                print("C", name, sev, "AUROC", round(m["auroc"], 4), flush=True)
    out["camelyon17_c_source"] = c_rows

    dest = Path("/kaggle/working/camelyon17_remaining_optional_s42.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", dest, flush=True)
    print("DONE. Keep negative SAR/EATA-C numbers. Do not retune.", flush=True)


if __name__ == "__main__":
    main()
