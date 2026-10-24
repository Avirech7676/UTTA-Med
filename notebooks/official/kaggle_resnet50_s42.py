#!/usr/bin/env python3
"""X1 — ResNet-50 Camelyon17 seed 42. Same protocol as ResNet-18. Not Table 1 until 5 seeds exist."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tta_kit import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_MC,
    TAU_PCT,
    ResNetMC,
    entropy_bin,
    harm_bin,
    mc_bin,
    metrics_bin,
    predict_bin,
    seed_all,
    tta_bin,
)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
BS_TRAIN = int(os.environ.get("BS_TRAIN", "32"))
BS_TTA = 32
EPOCHS = 12
PATIENCE = 4
TRAIN_H = {0, 3, 4}
VAL_H = {1}
TEST_H = {2}

TF_TRAIN = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
TF_EVAL = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class CamHF(Dataset):
    def __init__(self, hf, hospitals, transform):
        self.transform = transform
        feat = hf.features
        self.img_key = "image" if "image" in feat else "img"
        self.y_key = "label" if "label" in feat else "tumor"
        hosp = set(int(h) for h in hospitals)
        cands = [k for k in ("center", "hospital") if k in feat]
        self.c_key = cands[0] if cands else "center"
        print("keys", list(feat), "filter", self.c_key, "in", hosp)

        def keep(ex):
            c = ex.get(self.c_key)
            if c is None and isinstance(ex.get("metadata"), dict):
                c = ex["metadata"].get("center", ex["metadata"].get("hospital"))
            try:
                return int(c) in hosp
            except Exception:
                return False

        self.hf = hf.filter(keep, desc=f"hosp {sorted(hosp)}")
        print("kept", len(self.hf), hospitals)

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        row = self.hf[int(i)]
        img = row[self.img_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img)).convert("RGB")
        else:
            img = img.convert("RGB")
        y = int(row[self.y_key])
        return self.transform(img), y


def load_splits():
    from datasets import load_dataset

    print("loading Camelyon17-WILDS")
    ds = load_dataset("wltjr1007/Camelyon17-WILDS")
    print(ds)
    # train split has 0,3,4; validation has 1 (+ maybe 0,3,4); test has 2
    train_hf = ds["train"]
    val_hf = ds.get("validation", ds["train"])
    test_hf = ds["test"]
    train = CamHF(train_hf, TRAIN_H, TF_TRAIN)
    id_val = CamHF(val_hf if "validation" in ds else train_hf, TRAIN_H, TF_EVAL)
    # hospital 1 is in validation for WILDS
    ood = CamHF(val_hf, VAL_H, TF_EVAL)
    test = CamHF(test_hf, TEST_H, TF_EVAL)
    if len(ood) < 1000:
        ood = CamHF(train_hf, VAL_H, TF_EVAL)
    return train, id_val, ood, test


def main():
    seed_all(SEED)
    print("device", DEVICE, "BS_TRAIN", BS_TRAIN)
    train, id_val, ood, test = load_splits()
    print("n train/id/ood/test", len(train), len(id_val), len(ood), len(test))
    tr = DataLoader(train, batch_size=BS_TRAIN, shuffle=True, num_workers=2, pin_memory=True)
    ood_ld = DataLoader(ood, batch_size=BS_TTA, shuffle=False, num_workers=2)
    te_ld = DataLoader(test, batch_size=BS_TTA, shuffle=False, num_workers=2)

    model = ResNetMC("resnet50", n_classes=1).to(DEVICE)
    model._arch = "resnet50"
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best, best_state, bad = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in tqdm(tr, desc=f"ep{ep}"):
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logit = model(x)
                loss = F.binary_cross_entropy_with_logits(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
        yv, pv = predict_bin(model, ood_ld, DEVICE)
        met = metrics_bin(yv, pv)
        print(f"epoch {ep} loss={np.mean(losses):.4f} val_auroc={met['auroc']:.4f} val_f1={met['f1']:.4f}")
        if met["auroc"] > best:
            best = met["auroc"]
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "val_auroc": best, "epoch": ep, "arch": "resnet50"},
                OUT / "camelyon17_resnet50_source_s42_BEST.pt",
            )
            print("  saved BEST")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop")
                break
    model.load_state_dict(best_state)
    model.to(DEVICE)
    yv, pv = predict_bin(model, ood_ld, DEVICE)
    ood_m = metrics_bin(yv, pv)
    print("ASSERT OOD", ood_m["auroc"], "ckpt", best)
    yt, pt = predict_bin(model, te_ld, DEVICE)
    src = metrics_bin(yt, pt)
    print("SOURCE test", src)

    yv, pbar, u = mc_bin(model, ood_ld, DEVICE)
    tau = float(np.percentile(u, TAU_PCT))
    conf = np.maximum(pbar, 1 - pbar)
    ent = entropy_bin(pbar)
    cov = float((u < tau).mean())
    conf_t = float(np.quantile(conf, 1.0 - cov))
    ent_t = float(np.quantile(ent, cov))
    print("tau", tau, "val_cov", cov, "conf_t", conf_t, "ent_t", ent_t)

    def clone():
        m = ResNetMC("resnet50", 1).to(DEVICE)
        m._arch = "resnet50"
        m.load_state_dict(best_state)
        return m

    rows = {"source_test": src, "val_ood": ood_m}
    for gate, kw in [
        ("tent", {}),
        ("eata", {}),
        ("random", {"p_keep": cov}),
        ("confidence", {"conf_t": conf_t}),
        ("entropy", {"ent_t": ent_t}),
        ("utta", {"tau": tau}),
    ]:
        m = clone()
        y, a, b, c = tta_bin(m, te_ld, DEVICE, gate, **kw)
        met = metrics_bin(y, b)
        h = harm_bin(y, a, b)
        rows[gate + "_test"] = {**met, "coverage": c, "harm": h}
        print(f"{gate:12s} AUROC={met['auroc']:.4f} F1={met['f1']:.4f} cov={c:.3f} harm={h['harm_rate']:.4f}")

    blob = {
        "arch": "resnet50",
        "seed": SEED,
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": 1e-5,
            "n_mc": N_MC,
            "tau_rule": "70th pct val U",
        },
        "tau": tau,
        "best_val_auroc": best,
        "results": rows,
        "note": "Supplement Table 8. Not mixed into ResNet-18 5-seed mean until seeds 123/2024/7/99 exist.",
    }
    (OUT / "camelyon17_resnet50_full_s42.json").write_text(json.dumps(blob, indent=2))
    print("wrote", OUT / "camelyon17_resnet50_full_s42.json")


if __name__ == "__main__":
    main()
