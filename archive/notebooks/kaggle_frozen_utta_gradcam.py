"""Stage 10 fix — post-TTA Grad-CAM with FROZEN source U (seed 42).

The previous run gated UTTA with MC-Dropout on the *adapting* model.
That collapsed on hospital 2 (AUROC 0.324). Official Gate D uses a
frozen source copy for the gate and got UTTA AUROC 0.951.

This notebook:
  - source | Tent k=1 | UTTA k=1 on the SAME hospital-2 patches
  - gate scores come from a frozen source network (never updated)
  - abort if test UTTA AUROC < 0.85 (do not write galleries)

No retrain. GPU T4, Internet ON, Add Input seed-42 BEST.pt.
Runtime ~20–30 min.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

SEED = 42
TTA_LR = 1e-5
N_MC = 20
E_MARGIN = 0.4
BS = 64
K_CAM = 6
AUROC_FLOOR = 0.85
WORK = Path("/kaggle/working")
CAMDIR = WORK / "gradcam_frozen"
CAMDIR.mkdir(parents=True, exist_ok=True)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_ckpt() -> str:
    hits = glob.glob("/kaggle/input/**/*s42*BEST*.pt", recursive=True)
    hits += glob.glob("/kaggle/working/*s42*BEST*.pt")
    if not hits:
        raise FileNotFoundError("Add Input camelyon17_resnet18_source_s42_BEST.pt")
    print("ckpt", hits[0])
    return hits[0]


def cid(v) -> int:
    return int(v.item() if hasattr(v, "item") else v)


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


class HFCam(Dataset):
    def __init__(self, split, centers, tf):
        centers = {int(c) for c in centers}
        keep = [i for i, c in enumerate(split["center"]) if cid(c) in centers]
        self.ds = split.select(keep)
        self.tf = tf
        print(" kept", len(self.ds), sorted(centers))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        x = self.tf(img.convert("RGB"))
        y = torch.tensor(float(int(r["label"])))
        return x, y, int(i)


def eval_mode(m):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def tta_mode(m):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.eval()
        elif isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def mc_mode(m):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
        elif isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def bn_affine(m):
    for p in m.parameters():
        p.requires_grad = False
    ps = []
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d) and mod.affine:
            mod.weight.requires_grad = True
            mod.bias.requires_grad = True
            ps += [mod.weight, mod.bias]
    return ps


def entropy(z):
    p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).view(-1)


@torch.no_grad()
def mc_var(src, x, n=N_MC):
    mc_mode(src)
    ps = [torch.sigmoid(src(x)).view(-1) for _ in range(n)]
    return torch.stack(ps, 0).var(0, unbiased=False)


def cls_metrics(y, p):
    y = np.asarray(y).astype(np.int32)
    p = np.asarray(p).astype(np.float64)
    pred = (p >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(tp + fn, 1)
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"n": int(len(y)), "auroc": auroc, "auprc": auprc, "recall": rec, "f1": float(
        (lambda prec, rec: 2 * prec * rec / max(prec + rec, 1e-12))(
            tp / max(tp + int(((pred == 1) & (y == 0)).sum()), 1), rec
        )
    )}


def load_net(path):
    m = Net().to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(sd, strict=True)
    eval_mode(m)
    if isinstance(ck, dict):
        print("epoch", ck.get("epoch"), "val_auroc", ck.get("val_auroc"))
    return m


def adapt_frozen_gate(src, loader, method, tau):
    """src is NEVER updated. Adapt a copy. Gate from src only."""
    m = Net().to(device)
    m.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(bn_affine(m), lr=TTA_LR)
    ys, after, n_keep, n_seen, n_skip = [], [], 0, 0, 0
    for x, y, *_ in tqdm(loader, desc=f"{method} frozen-gate"):
        x = x.to(device)
        n_seen += int(x.size(0))
        with torch.no_grad():
            eval_mode(src)
            if method == "tent":
                keep = torch.ones(x.size(0), device=device, dtype=torch.bool)
            elif method == "eata":
                keep = entropy(src(x)) < E_MARGIN
            elif method == "utta":
                keep = mc_var(src, x) < tau
            else:
                raise ValueError(method)
        w = keep.float()
        nk = int(keep.sum().item())
        if nk == 0:
            n_skip += 1
        else:
            tta_mode(m)
            opt.zero_grad(set_to_none=True)
            loss = (w * entropy(m(x))).sum() / w.sum()
            loss.backward()
            opt.step()
            n_keep += nk
        with torch.no_grad():
            eval_mode(m)
            after.append(torch.sigmoid(m(x)).view(-1).cpu())
            ys.append(y)
    y = torch.cat(ys).numpy()
    p1 = torch.cat(after).numpy()
    met = cls_metrics(y, p1)
    met["coverage"] = n_keep / max(n_seen, 1)
    met["n_skip_batch"] = n_skip
    print(
        f"{method:6s} AUROC={met['auroc']:.4f} F1={met['f1']:.4f} "
        f"rec={met['recall']:.4f} cov={met['coverage']:.3f}"
    )
    return m, met


class GradCAM:
    def __init__(self, model: Net):
        self.model = model
        layer = model.backbone.layer4
        self.acts = None
        self.grads = None
        self.h1 = layer.register_forward_hook(lambda _m, _i, o: setattr(self, "acts", o.detach()))
        self.h2 = layer.register_full_backward_hook(
            lambda _m, _gi, go: setattr(self, "grads", go[0].detach())
        )

    def close(self):
        self.h1.remove()
        self.h2.remove()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        eval_mode(self.model)
        x = x.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self.model(x).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def denorm(t):
    x = t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        x[c] = x[c] * s + m
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img, cam):
    heat = plt.cm.jet(cam)[..., :3]
    return np.clip(0.45 * img + 0.55 * heat, 0, 1)


def cam_on_cache(model, cache):
    camper = GradCAM(model)
    out = {}
    for i, (x, y) in cache.items():
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            eval_mode(model)
            p = float(torch.sigmoid(model(xb)).item())
        out[i] = {"p": p, "cam": camper(xb), "y": y}
    camper.close()
    return out


def main():
    print("device", device)
    print("GATE = frozen source copy. Adapting copy is separate.")
    ckpt = find_ckpt()
    tf = transforms.Compose(
        [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    val_ds = HFCam(raw["validation"], {1}, tf)
    test_ds = HFCam(raw["test"], {2}, tf)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    src = load_net(ckpt)

    print("\n--- tau from frozen source on hospital 1 ---")
    us = []
    for x, y, _ in tqdm(val_loader, desc="MC val"):
        us.append(mc_var(src, x.to(device)).cpu().numpy())
    U = np.concatenate(us)
    tau = float(np.quantile(U, 0.70))
    print(f"tau 70th pct = {tau:.6e}")
    eval_mode(src)

    print("\n--- TEST Tent then UTTA (frozen-source gate) ---")
    tent_m, tent_met = adapt_frozen_gate(src, test_loader, "tent", tau)
    utta_m, utta_met = adapt_frozen_gate(src, test_loader, "utta", tau)

    if utta_met["auroc"] < AUROC_FLOOR:
        raise RuntimeError(
            f"UTTA AUROC {utta_met['auroc']:.3f} < {AUROC_FLOOR}. "
            "Gate is still wrong — do not write Grad-CAM galleries."
        )
    print("UTTA survived floor. Official-like. Writing CAMs.")

    g = torch.Generator().manual_seed(SEED)
    scan = DataLoader(test_ds, batch_size=BS, shuffle=True, generator=g, num_workers=0)
    buckets = defaultdict(list)
    nseen = 0
    print("scan shuffled hospital-2 (source scores)")
    for x, y, idx in tqdm(scan, desc="fill"):
        x = x.to(device)
        with torch.no_grad():
            eval_mode(src)
            p = torch.sigmoid(src(x)).view(-1)
            u = mc_var(src, x, n=8)
        pred = (p >= 0.5).long()
        for i in range(x.size(0)):
            row = {
                "i": int(idx[i]),
                "y": int(y[i].item()),
                "p": float(p[i].item()),
                "u": float(u[i].item()),
                "pred": int(pred[i].item()),
            }
            row["ok"] = row["pred"] == row["y"]
            cat = (
                "TP" if row["ok"] and row["y"] == 1
                else "TN" if row["ok"] and row["y"] == 0
                else "FP" if row["pred"] == 1
                else "FN"
            )
            buckets[cat].append(row)
            buckets["wrong" if not row["ok"] else "correct"].append(row)
        nseen += x.size(0)
        if all(len(buckets[k]) >= K_CAM for k in ("TP", "TN", "FP", "FN")) and nseen >= 512:
            break
        if nseen >= 20000:
            break
    print("scanned", nseen, {k: len(buckets[k]) for k in ("TP", "TN", "FP", "FN")})
    if len(buckets["TP"]) == 0 or len(buckets["FN"]) == 0:
        raise RuntimeError("Need TP and FN")

    galleries = {
        "TP": buckets["TP"][:K_CAM],
        "TN": buckets["TN"][:K_CAM],
        "FP": buckets["FP"][:K_CAM],
        "FN": buckets["FN"][:K_CAM],
        "highU_wrong": sorted(buckets["wrong"], key=lambda r: -r["u"])[:K_CAM],
        "lowU_correct": sorted(buckets["correct"], key=lambda r: r["u"])[:K_CAM],
    }
    ids = sorted({r["i"] for rows in galleries.values() for r in rows})
    cache = {i: (test_ds[i][0], int(test_ds[i][1].item())) for i in ids}

    src_cam = cam_on_cache(src, cache)
    tent_cam = cam_on_cache(tent_m, cache)
    utta_cam = cam_on_cache(utta_m, cache)

    def ok(p, y):
        return int(p >= 0.5) == y

    trans = {"tent_fix": 0, "tent_harm": 0, "utta_fix": 0, "utta_harm": 0}
    for i in ids:
        y = cache[i][1]
        s, t, u = src_cam[i]["p"], tent_cam[i]["p"], utta_cam[i]["p"]
        trans["tent_fix"] += (not ok(s, y)) and ok(t, y)
        trans["tent_harm"] += ok(s, y) and (not ok(t, y))
        trans["utta_fix"] += (not ok(s, y)) and ok(u, y)
        trans["utta_harm"] += ok(s, y) and (not ok(u, y))
    print("CAM-set transitions", trans)

    for name, rows in galleries.items():
        fig, axes = plt.subplots(len(rows), 4, figsize=(10.2, 2.15 * len(rows)))
        if len(rows) == 1:
            axes = np.array([axes])
        for r, row in enumerate(rows):
            i = row["i"]
            img = denorm(cache[i][0])
            y = cache[i][1]
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"patch y={y}", fontsize=8)
            axes[r, 0].axis("off")
            for c, (lab, blob) in enumerate(
                [("source", src_cam), ("tent", tent_cam), ("utta", utta_cam)], start=1
            ):
                p = blob[i]["p"]
                axes[r, c].imshow(overlay(img, blob[i]["cam"]))
                axes[r, c].set_title(f"{lab} p={p:.2f} {'OK' if ok(p, y) else 'ERR'}", fontsize=8)
                axes[r, c].axis("off")
        fig.suptitle(
            f"Frozen-source-U Grad-CAM · {name} · hospital 2 · seed 42 (audit, not clinical)",
            fontsize=11,
        )
        fig.tight_layout()
        fp = CAMDIR / f"frozen_{name}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        print("wrote", fp)

    out = {
        "seed": SEED,
        "checkpoint": ckpt,
        "tau": tau,
        "gate": "frozen_source_U",
        "tent_test": tent_met,
        "utta_test": utta_met,
        "cam_transitions": trans,
        "note": (
            "Official Gate D protocol: MC-Dropout U from frozen source. "
            "Do not confuse with the collapsed online-U run (AUROC 0.324)."
        ),
    }
    (WORK / "camelyon17_frozen_utta_gradcam_s42.json").write_text(json.dumps(out, indent=2))
    print("wrote", WORK / "camelyon17_frozen_utta_gradcam_s42.json")
    print("PASS if UTTA AUROC is ~0.95 not ~0.32")
    print("Download /kaggle/working/gradcam_frozen/*.png")


if __name__ == "__main__":
    main()
