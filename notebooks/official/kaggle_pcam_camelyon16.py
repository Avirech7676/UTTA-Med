#!/usr/bin/env python3
"""X2 — PCam (Camelyon16 patches). Train/val/test are WSI-disjoint.
This IS the Camelyon16 track for this paper. Full WSI CAMELYON16 is out of scope.
"""
from __future__ import annotations

import json
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
BS = 64
EPOCHS = 8
PATIENCE = 3

TF_TR = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
TF_EV = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)


class HFPcam(Dataset):
    def __init__(self, hf, transform):
        self.hf = hf
        self.transform = transform
        feat = hf.features
        self.img_key = "image" if "image" in feat else "img"
        self.y_key = "label" if "label" in feat else "tumor"

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
        if y > 1:
            y = int(y > 0)
        return self.transform(img), y


def load_pcam():
    from datasets import load_dataset

    print("HF 1aurent/PatchCamelyon")
    d = load_dataset("1aurent/PatchCamelyon")
    print(d)
    keys = list(d.keys())
    # expect train/validation/test
    def pick(*names):
        for n in names:
            if n in d:
                return d[n]
        return d[keys[0]]

    train = pick("train")
    val = pick("validation", "valid", "val")
    test = pick("test")
    return HFPcam(train, TF_TR), HFPcam(val, TF_EV), HFPcam(test, TF_EV)


def main():
    seed_all(SEED)
    print("device", DEVICE)
    print("PCam = Camelyon16 96x96 patches, WSI-disjoint splits")
    train, val, test = load_pcam()
    print("n", len(train), len(val), len(test))
    tr = DataLoader(train, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(val, batch_size=BS, shuffle=False, num_workers=2)
    te = DataLoader(test, batch_size=BS, shuffle=False, num_workers=2)

    model = ResNetMC("resnet18", 1).to(DEVICE)
    model._arch = "resnet18"
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best, st, bad = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in tqdm(tr, desc=f"ep{ep}"):
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        yv, pv = predict_bin(model, va, DEVICE)
        met = metrics_bin(yv, pv)
        print(f"epoch {ep} loss={np.mean(losses):.4f} val_auroc={met['auroc']:.4f} f1={met['f1']:.4f}")
        if met["auroc"] > best:
            best, bad = met["auroc"], 0
            st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": st, "val_auroc": best, "epoch": ep}, OUT / "pcam_resnet18_s42_BEST.pt")
            print("  saved BEST")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop")
                break
    if best < 0.90:
        raise SystemExit(f"val AUROC {best:.3f} < 0.90 — stop")
    model.load_state_dict(st)
    model.to(DEVICE)
    yt, pt = predict_bin(model, te, DEVICE)
    src = metrics_bin(yt, pt)
    print("SOURCE test", src)

    yv, pbar, u = mc_bin(model, va, DEVICE)
    tau = float(np.percentile(u, TAU_PCT))
    conf = np.maximum(pbar, 1 - pbar)
    ent = entropy_bin(pbar)
    cov = float((u < tau).mean())
    conf_t = float(np.quantile(conf, 1.0 - cov))
    ent_t = float(np.quantile(ent, cov))
    from scipy.stats import spearmanr

    sp = float(spearmanr(u, 1 - conf).correlation)
    print("tau", tau, "cov", cov, "Spearman U vs 1-conf", sp, "(binary → expect ~1)")

    def clone():
        m = ResNetMC("resnet18", 1).to(DEVICE)
        m._arch = "resnet18"
        m.load_state_dict(st)
        return m

    rows = {"source_test": src}
    for gate, kw in [
        ("tent", {}),
        ("eata", {}),
        ("random", {"p_keep": cov}),
        ("confidence", {"conf_t": conf_t}),
        ("entropy", {"ent_t": ent_t}),
        ("utta", {"tau": tau}),
    ]:
        m = clone()
        y, a, b, c = tta_bin(m, te, DEVICE, gate, **kw)
        met = metrics_bin(y, b)
        h = harm_bin(y, a, b)
        rows[gate] = {**met, "coverage": c, "harm": h}
        print(f"{gate:12s} AUROC={met['auroc']:.4f} F1={met['f1']:.4f} cov={c:.3f} harm={h['harm_rate']:.4f}")

    blob = {
        "dataset": "PCam / Camelyon16 patches",
        "seed": SEED,
        "tau": tau,
        "spearman_U_vs_1minus_conf": sp,
        "best_val_auroc": best,
        "results": rows,
        "note": "Camelyon16 track at patch level. WSI-disjoint. Binary → U≈confidence expected.",
    }
    (OUT / "pcam_full_s42.json").write_text(json.dumps(blob, indent=2))
    print("wrote", OUT / "pcam_full_s42.json")


if __name__ == "__main__":
    main()
