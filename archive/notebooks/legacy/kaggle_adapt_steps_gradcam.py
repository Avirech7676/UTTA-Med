"""Remaining optional science (seed 42, no retrain).

Part 1 — Adaptation-step stability on OOD val (hospital 1)
  Inner SGD steps per batch k in {1, 2, 5, 10}, ONE pass over val.
  Methods: Tent, EATA, UTTA-Med.
  Official recipe stays k=1. This only tests whether longer inner
  adaptation helps or collapses on the validation hospital.

Part 2 — Post-TTA Grad-CAM on hospital 2
  Same patches, three models: source | Tent (k=1) | UTTA (k=1).
  Interpretability audit, not clinical validation.

Frozen: bn_freeze_stats, lr=1e-5, N_MC=20, tau = val 70th pct of U.

Kaggle: GPU T4, Internet ON, Add Input seed-42 BEST.pt.
Runtime ~70–100 min.
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
STEPS_GRID = (1, 2, 5, 10)
BS = 64
K_CAM = 6
WORK = Path("/kaggle/working")
FIG = WORK / "figures"
CAMDIR = WORK / "gradcam_post_tta"
FIG.mkdir(parents=True, exist_ok=True)
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
    """Official bn_freeze_stats: affine trainable, running stats frozen, dropout off."""
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


def ece_brier(y, p, n_bins=15):
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p).astype(np.float64)
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece), brier


def cls_metrics(y, p):
    y = np.asarray(y).astype(np.int32)
    p = np.asarray(p).astype(np.float64)
    pred = (p >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    acc = (tp + tn) / max(len(y), 1)
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ece, brier = ece_brier(y, p)
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "brier": brier,
    }


def harm_stats(y, p0, p1):
    y = np.asarray(y).astype(np.int32)
    b0 = (np.asarray(p0) >= 0.5).astype(np.int32)
    b1 = (np.asarray(p1) >= 0.5).astype(np.int32)
    ok0 = b0 == y
    ok1 = b1 == y
    n_wrong = int((~ok0).sum())
    n_corr = int(ok0.sum())
    correction = int((~ok0 & ok1).sum())
    harmful = int((ok0 & ~ok1).sum())
    return {
        "correction_rate": correction / max(n_wrong, 1),
        "harm_rate": harmful / max(n_corr, 1),
        "flip_rate": float((b0 != b1).mean()),
        "correction": correction,
        "harmful": harmful,
    }


def load_net(path):
    m = Net().to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m.load_state_dict(sd, strict=True)
    eval_mode(m)
    if isinstance(ck, dict):
        print("epoch", ck.get("epoch"), "val_auroc", ck.get("val_auroc"))
    return m, ck


@torch.no_grad()
def predict(model, loader):
    eval_mode(model)
    ys, ps = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x, y = batch[0].to(device), batch[1].numpy()
        p = torch.sigmoid(model(x)).view(-1).cpu().numpy()
        ys.append(y)
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def mc_on_loader(model, loader, n=N_MC):
    mc_mode(model)
    ys, means, us = [], [], []
    for batch in tqdm(loader, desc="MC val"):
        x, y = batch[0].to(device), batch[1]
        ps = [torch.sigmoid(model(x)).view(-1) for _ in range(n)]
        P = torch.stack(ps, 0)
        means.append(P.mean(0).cpu())
        us.append(P.var(0, unbiased=False).cpu())
        ys.append(y)
    eval_mode(model)
    return torch.cat(ys).numpy(), torch.cat(means).numpy(), torch.cat(us).numpy()


def adapt_stream(model, loader, method, tau, steps, before_p=None):
    """One continual pass. Gate scores from the *current* model except UTTA
    uses on-the-fly MC-Dropout (same as Gate D). Empty gate → skip batch.
    """
    tta_mode(model)
    opt = torch.optim.Adam(bn_affine(model), lr=TTA_LR)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(SEED)
    ys, after, n_keep, n_seen, n_skip = [], [], 0, 0, 0
    desc = f"{method} k={steps}"
    for batch in tqdm(loader, desc=desc):
        x, y = batch[0].to(device), batch[1]
        n_seen += int(x.size(0))
        tta_mode(model)
        with torch.no_grad():
            if method == "tent":
                keep = torch.ones(x.size(0), device=device, dtype=torch.bool)
            elif method == "eata":
                z0 = model(x)
                keep = entropy(z0) < E_MARGIN
            elif method == "utta":
                mc_mode(model)
                ps = [torch.sigmoid(model(x)).view(-1) for _ in range(N_MC)]
                u = torch.stack(ps, 0).var(0, unbiased=False)
                keep = u < tau
                tta_mode(model)
            else:
                raise ValueError(method)
        w = keep.float()
        nk = int(keep.sum().item())
        if nk == 0:
            n_skip += 1
        else:
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = (w * entropy(model(x))).sum() / w.sum()
                loss.backward()
                opt.step()
            n_keep += nk
        with torch.no_grad():
            eval_mode(model)
            after.append(torch.sigmoid(model(x)).view(-1).cpu())
            ys.append(y)
            tta_mode(model)
    y = torch.cat(ys).numpy()
    p1 = torch.cat(after).numpy()
    met = cls_metrics(y, p1)
    rec = {
        "method": method,
        "steps": steps,
        "coverage": n_keep / max(n_seen, 1),
        "n_skip_batch": n_skip,
        "after": met,
    }
    if before_p is not None:
        rec["harm"] = harm_stats(y, before_p, p1)
        rec["before"] = cls_metrics(y, before_p)
    return rec, p1


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


def denorm(t):
    x = t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        x[c] = x[c] * s + m
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img, cam):
    heat = plt.cm.jet(cam)[..., :3]
    return np.clip(0.45 * img + 0.55 * heat, 0, 1)


def main():
    print("device", device)
    ckpt = find_ckpt()
    tf = transforms.Compose(
        [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    val_ds = HFCam(raw["validation"], {1}, tf)
    test_ds = HFCam(raw["test"], {2}, tf)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    model, ck = load_net(ckpt)
    src_state = deepcopy(model.state_dict())

    print("\n=== Part 1  adaptation-step sweep on hospital 1 (val) ===")
    y_val, p_src_val, u_val = mc_on_loader(model, val_loader, n=N_MC)
    tau = float(np.quantile(u_val, 0.70))
    print(f"FROZEN tau=70th pct U={tau:.6e}  val source AUROC={roc_auc_score(y_val, p_src_val):.4f}")

    sweep = []
    for method in ("tent", "eata", "utta"):
        for k in STEPS_GRID:
            model.load_state_dict(src_state)
            rec, _ = adapt_stream(model, val_loader, method, tau, k, before_p=p_src_val)
            a = rec["after"]
            h = rec.get("harm", {})
            print(
                f"{method:6s} k={k:2d}  AUROC={a['auroc']:.4f} F1={a['f1']:.4f} "
                f"ECE={a['ece']:.4f} harm={h.get('harm_rate', float('nan')):.4f} "
                f"cov={rec['coverage']:.3f} skip={rec['n_skip_batch']}"
            )
            sweep.append(rec)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    colors = {"tent": "#d62728", "eata": "#ff7f0e", "utta": "#2ca02c"}
    for ax, key, title in zip(
        axes,
        ("auroc", "f1", "harm"),
        ("Val AUROC", "Val F1", "Val harm rate"),
    ):
        for method in ("tent", "eata", "utta"):
            xs, ys = [], []
            for rec in sweep:
                if rec["method"] != method:
                    continue
                xs.append(rec["steps"])
                if key == "harm":
                    ys.append(rec["harm"]["harm_rate"])
                else:
                    ys.append(rec["after"][key])
            ax.plot(xs, ys, "-o", color=colors[method], label=method, linewidth=2)
        ax.set_xlabel("inner SGD steps / batch")
        ax.set_title(title)
        ax.set_xticks(list(STEPS_GRID))
        ax.grid(True, alpha=0.3)
        if key != "harm":
            ax.axhline(
                cls_metrics(y_val, p_src_val)[key],
                color="#1f77b4",
                linestyle="--",
                linewidth=1,
                label="source",
            )
        ax.legend(fontsize=8)
    fig.suptitle("Adaptation-step stability · hospital 1 · seed 42 · official lr=1e-5, BN stats frozen")
    fig.tight_layout()
    fig.savefig(FIG / "fig_adapt_steps_val_s42.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_adapt_steps_val_s42.png")

    # ============================================================
    print("\n=== Part 2  post-TTA Grad-CAM (hospital 2, k=1) ===")
    model.load_state_dict(src_state)
    eval_mode(model)
    g = torch.Generator().manual_seed(SEED)
    scan_loader = DataLoader(test_ds, batch_size=BS, shuffle=True, generator=g, num_workers=0)

    buckets = defaultdict(list)
    nseen = 0
    print("scan shuffled hospital-2 for TP/TN/FP/FN (source)")
    for x, y, idx in tqdm(scan_loader, desc="fill buckets"):
        x = x.to(device)
        with torch.no_grad():
            eval_mode(model)
            p = torch.sigmoid(model(x)).view(-1)
            mc_mode(model)
            ps = [torch.sigmoid(model(x)).view(-1) for _ in range(8)]
            u = torch.stack(ps, 0).var(0, unbiased=False)
            eval_mode(model)
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
            if row["ok"] and row["y"] == 1:
                buckets["TP"].append(row)
            elif row["ok"] and row["y"] == 0:
                buckets["TN"].append(row)
            elif (not row["ok"]) and row["pred"] == 1:
                buckets["FP"].append(row)
            else:
                buckets["FN"].append(row)
            buckets["wrong" if not row["ok"] else "correct"].append(row)
        nseen += x.size(0)
        if all(len(buckets[k]) >= K_CAM for k in ("TP", "TN", "FP", "FN")) and nseen >= 512:
            break
        if nseen >= 20000:
            break
    print("scanned", nseen, {k: len(buckets[k]) for k in ("TP", "TN", "FP", "FN")})
    if len(buckets["TP"]) == 0 or len(buckets["FN"]) == 0:
        raise RuntimeError("Need TP and FN. Do not use empty galleries.")

    galleries = {
        "TP": buckets["TP"][:K_CAM],
        "TN": buckets["TN"][:K_CAM],
        "FP": buckets["FP"][:K_CAM],
        "FN": buckets["FN"][:K_CAM],
        "highU_wrong": sorted(buckets["wrong"], key=lambda r: -r["u"])[:K_CAM],
        "lowU_correct": sorted(buckets["correct"], key=lambda r: r["u"])[:K_CAM],
    }
    example_ids = sorted({r["i"] for rows in galleries.values() for r in rows})
    # cache tensors
    cache = {}
    for i in example_ids:
        x, y, _ = test_ds[i]
        cache[i] = (x, int(y.item()))

    def cam_on_cache(model):
        camper = GradCAM(model)
        out = {}
        for i, (x, y) in cache.items():
            xb = x.unsqueeze(0).to(device)
            with torch.no_grad():
                eval_mode(model)
                p = float(torch.sigmoid(model(xb)).item())
            cam = camper(xb)
            out[i] = {"p": p, "cam": cam, "y": y}
        camper.close()
        return out

    print("source CAMs")
    model.load_state_dict(src_state)
    src_cam = cam_on_cache(model)

    print("Tent k=1 on full test stream")
    model.load_state_dict(src_state)
    tent_rec, _ = adapt_stream(model, test_loader, "tent", tau, 1, before_p=None)
    print("tent test AUROC", tent_rec["after"]["auroc"])
    tent_cam = cam_on_cache(model)

    print("UTTA k=1 on full test stream")
    model.load_state_dict(src_state)
    utta_rec, _ = adapt_stream(model, test_loader, "utta", tau, 1, before_p=None)
    print("utta test AUROC", utta_rec["after"]["auroc"])
    utta_cam = cam_on_cache(model)

    def pred_of(p, y):
        return int(p >= 0.5) == y

    for name, rows in galleries.items():
        fig, axes = plt.subplots(len(rows), 4, figsize=(10.2, 2.15 * len(rows)))
        if len(rows) == 1:
            axes = np.array([axes])
        for r, row in enumerate(rows):
            i = row["i"]
            img = denorm(cache[i][0])
            y = cache[i][1]
            trio = [("source", src_cam), ("tent", tent_cam), ("utta", utta_cam)]
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"patch y={y}", fontsize=8)
            axes[r, 0].axis("off")
            for c, (lab, blob) in enumerate(trio, start=1):
                p = blob[i]["p"]
                cam = blob[i]["cam"]
                ok = pred_of(p, y)
                axes[r, c].imshow(overlay(img, cam))
                axes[r, c].set_title(f"{lab} p={p:.2f} {'OK' if ok else 'ERR'}", fontsize=8)
                axes[r, c].axis("off")
        fig.suptitle(
            f"Post-TTA Grad-CAM  ·  {name}  ·  hospital 2  ·  seed 42  (audit, not clinical)",
            fontsize=11,
        )
        fig.tight_layout()
        fp = CAMDIR / f"compare_{name}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        print("wrote", fp)

    # transition gallery: source wrong→correct / correct→wrong under UTTA vs Tent
    trans = {"tent_fix": [], "tent_harm": [], "utta_fix": [], "utta_harm": []}
    for i in example_ids:
        y = cache[i][1]
        s_ok = pred_of(src_cam[i]["p"], y)
        t_ok = pred_of(tent_cam[i]["p"], y)
        u_ok = pred_of(utta_cam[i]["p"], y)
        if (not s_ok) and t_ok:
            trans["tent_fix"].append(i)
        if s_ok and (not t_ok):
            trans["tent_harm"].append(i)
        if (not s_ok) and u_ok:
            trans["utta_fix"].append(i)
        if s_ok and (not u_ok):
            trans["utta_harm"].append(i)
    print("transitions in CAM set", {k: len(v) for k, v in trans.items()})

    out = {
        "seed": SEED,
        "checkpoint": ckpt,
        "tau": tau,
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": TTA_LR,
            "n_mc": N_MC,
            "tau_rule": "70th percentile of val U",
            "step_grid": list(STEPS_GRID),
            "step_meaning": "inner SGD steps per batch, one pass over the stream",
        },
        "val_source": cls_metrics(y_val, p_src_val),
        "step_sweep": [
            {
                "method": r["method"],
                "steps": r["steps"],
                "coverage": r["coverage"],
                "n_skip_batch": r["n_skip_batch"],
                "after_auroc": r["after"]["auroc"],
                "after_f1": r["after"]["f1"],
                "after_ece": r["after"]["ece"],
                "harm_rate": r["harm"]["harm_rate"],
                "correction_rate": r["harm"]["correction_rate"],
            }
            for r in sweep
        ],
        "test_tent_k1": tent_rec["after"],
        "test_utta_k1": utta_rec["after"],
        "cam_transitions": {k: v for k, v in trans.items()},
        "galleries": {
            name: [{kk: row[kk] for kk in ("i", "y", "p", "u", "pred", "ok")} for row in rows]
            for name, rows in galleries.items()
        },
        "note": (
            "Official TTA remains k=1. Sweep is val-only. "
            "Grad-CAM is an interpretability audit, not clinical validation."
        ),
    }
    (WORK / "camelyon17_adapt_steps_gradcam_s42.json").write_text(json.dumps(out, indent=2))
    print("wrote", WORK / "camelyon17_adapt_steps_gradcam_s42.json")
    print("Download figures/ fig_adapt_steps_val_s42.png")
    print("Download gradcam_post_tta/ compare_*.png")
    print("DONE")


if __name__ == "__main__":
    main()
