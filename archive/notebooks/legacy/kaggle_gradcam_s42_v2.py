"""UTTA-Med Grad-CAM v2 — stratified / shuffled sample.

The first run stopped at ~3000 patches because of a hard cap. Camelyon17
test parquet is slide-ordered, so that prefix was almost all non-tumor
(TP=0, FN=0). This version SHUFFLES hospital-2 patches so every class
appears immediately, then keeps going until TP/TN/FP/FN each have K
examples. No retraining.
"""
from __future__ import annotations

import json, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

WORK = Path("/kaggle/working")
FIG = WORK / "gradcam"
FIG.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
eval_tf = transforms.Compose(
    [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)
K_EACH = 8
N_MC = 8
# shuffle so we hit tumor slides immediately; stop when buckets are full
MIN_SCAN = 512
HARD_CAP = 20000


def find_ckpt() -> str:
    hits = glob.glob("/kaggle/input/**/*s42*BEST*.pt", recursive=True)
    hits += glob.glob("/kaggle/input/**/*source_s42*.pt", recursive=True)
    hits += glob.glob("/kaggle/working/*s42*BEST*.pt")
    if not hits:
        raise FileNotFoundError("Add Input the seed-42 BEST.pt dataset")
    print("ckpt", hits[0])
    return hits[0]


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


def load_net(path: str) -> Net:
    m = Net().to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(sd, strict=True)
    m.eval()
    if isinstance(ck, dict):
        print("epoch", ck.get("epoch"), "val_auroc", ck.get("val_auroc"))
    return m


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
        self.model.eval()
        x = x.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        logit.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def denorm(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        x[c] = x[c] * s + m
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = plt.cm.jet(cam)[..., :3]
    return np.clip(0.45 * img + 0.55 * heat, 0, 1)


class HFCam(Dataset):
    def __init__(self, split, centers):
        centers = {int(c) for c in centers}

        def cid(v):
            return int(v.item() if hasattr(v, "item") else v)

        keep = [i for i, c in enumerate(split["center"]) if cid(c) in centers]
        self.ds = split.select(keep)
        print("kept", len(self.ds), sorted(centers))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        return eval_tf(img), torch.tensor(float(int(r["label"]))), int(i)


def mc_var(model, x, n=N_MC):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    ps = []
    with torch.no_grad():
        for _ in range(n):
            ps.append(torch.sigmoid(model(x)).squeeze(1))
    P = torch.stack(ps, 0)
    return P.mean(0), P.var(0, unbiased=False)


def cat_of(row):
    if row["ok"] and row["y"] == 1:
        return "TP"
    if row["ok"] and row["y"] == 0:
        return "TN"
    if (not row["ok"]) and row["pred"] == 1:
        return "FP"
    return "FN"


def main():
    ckpt = find_ckpt()
    model = load_net(ckpt)
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    ds = HFCam(raw["test"], {2})
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=g, num_workers=0)

    buckets = defaultdict(list)
    nseen = n_pos = 0
    model.eval()
    print("SCAN: shuffled hospital-2. Do NOT stop at 3%. Wait until TP and FN > 0.")
    pbar = tqdm(loader, desc="fill TP/TN/FP/FN")
    for x, y, idx in pbar:
        x = x.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).squeeze(1)
        _, u = mc_var(model, x, n=N_MC)
        pred = (p >= 0.5).long()
        for i in range(x.size(0)):
            row = {
                "i": int(idx[i]),
                "y": int(y[i].item()),
                "p": float(p[i].item()),
                "pred": int(pred[i].item()),
                "u": float(u[i].item()),
            }
            row["ok"] = row["pred"] == row["y"]
            n_pos += row["y"]
            buckets[cat_of(row)].append(row)
            buckets["wrong" if not row["ok"] else "correct"].append(row)
        nseen += x.size(0)
        pbar.set_postfix(
            n=nseen,
            pos=n_pos,
            TP=len(buckets["TP"]),
            FN=len(buckets["FN"]),
            FP=len(buckets["FP"]),
            TN=len(buckets["TN"]),
        )
        filled = all(len(buckets[k]) >= K_EACH for k in ("TP", "TN", "FP", "FN"))
        if filled and nseen >= MIN_SCAN:
            print("buckets full at n=", nseen)
            break
        if nseen >= HARD_CAP:
            break

    print("scanned", nseen, "positives", n_pos)
    for k in ("TP", "TN", "FP", "FN"):
        print(f"  {k}: {len(buckets[k])}")
    if len(buckets["TP"]) == 0 or len(buckets["FN"]) == 0:
        raise RuntimeError(
            "Still no tumor TP/FN after scan. Check checkpoint / labels. Do not use galleries."
        )

    galleries = {
        "TP": buckets["TP"][:K_EACH],
        "TN": buckets["TN"][:K_EACH],
        "FP": buckets["FP"][:K_EACH],
        "FN": buckets["FN"][:K_EACH],
        "highU_wrong": sorted(buckets["wrong"], key=lambda r: -r["u"])[:K_EACH],
        "lowU_correct": sorted(buckets["correct"], key=lambda r: r["u"])[:K_EACH],
    }

    camper = GradCAM(model)

    def grab(i):
        x, y, _ = ds[i]
        return x.to(device).unsqueeze(0), int(y.item())

    summary = {"scanned": nseen, "positives": n_pos, "ckpt": ckpt}
    for name, rows in galleries.items():
        if not rows:
            print("skip empty", name)
            continue
        fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.05 * len(rows)))
        if len(rows) == 1:
            axes = np.array([axes])
        for r, row in enumerate(rows):
            x, y = grab(row["i"])
            cam = camper(x)
            img = denorm(x[0])
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"y={y} p={row['p']:.2f}", fontsize=8)
            axes[r, 1].imshow(cam, cmap="jet")
            axes[r, 1].set_title(f"CAM  U={row['u']:.1e}", fontsize=8)
            axes[r, 2].imshow(overlay(img, cam))
            axes[r, 2].set_title(name, fontsize=8)
            for ax in axes[r]:
                ax.axis("off")
        fig.suptitle(
            f"Source ResNet-18 seed 42  ·  {name}  ·  hospital 2 (audit, not clinical)",
            fontsize=11,
        )
        fig.tight_layout()
        fp = FIG / f"source_{name}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        print("wrote", fp)
        summary[name] = [{k: row[k] for k in ("i", "y", "p", "u", "ok", "pred")} for row in rows]

    camper.close()
    summary["note"] = (
        "Qualitative audit only. Not clinical validation. "
        "Hospital-2 patches shuffled (seed 42) so TP/FN appear; previous 3% stop was a cap on slide-ordered prefix."
    )
    (WORK / "gradcam_manifest.json").write_text(json.dumps(summary, indent=2))
    print("COUNTS  TP", len(buckets["TP"]), "TN", len(buckets["TN"]),
          "FP", len(buckets["FP"]), "FN", len(buckets["FN"]))
    print("done", FIG)
    print("Download /kaggle/working/gradcam/*.png and gradcam_manifest.json")


if __name__ == "__main__":
    main()
