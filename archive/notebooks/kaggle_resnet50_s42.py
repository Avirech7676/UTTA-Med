#!/usr/bin/env python3
"""Phase 11 — ResNet-50 seed 42. Same frozen TTA recipe as ResNet-18.

Train source save-best on hospital 1, then Tent / EATA / confidence / UTTA on hospital 2.
Not in the 5-seed headline table. Do not retune on test.
"""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

SEED, SRC_LR, TTA_LR, N_MC = 42, 1e-4, 1e-5, 20
EPOCHS, PATIENCE, BS_TRAIN, BS_EVAL = 20, 5, 64, 32
E_MARGIN = 0.4
WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
print("If CUDA OOM during train, restart with BS_TRAIN=32")

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
eval_tf = transforms.Compose(
    [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)


def cid(v):
    return int(v.item() if hasattr(v, "item") else v)


class HFCamelyon(Dataset):
    def __init__(self, split, centers, tf):
        centers = {int(c) for c in centers}
        keep = [i for i, c in enumerate(split["center"]) if cid(c) in centers]
        self.ds = split.select(keep) if len(keep) < len(split) else split
        self.tf = tf
        print(" kept", len(self.ds), sorted(centers))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r["image"]
        img = (
            img.convert("RGB")
            if isinstance(img, Image.Image)
            else Image.fromarray(np.array(img)).convert("RGB")
        )
        return self.tf(img), torch.tensor(float(int(r["label"])))


try:
    from src.models.resnet50 import ResNet50UTTAMed as Net
except ImportError:
    class Net(nn.Module):
        """Fallback if src/ is not on PYTHONPATH (Kaggle zip layout)."""

        def __init__(self, pretrained=False):
            super().__init__()
            w = models.ResNet50_Weights.DEFAULT if pretrained else None
            bb = models.resnet50(weights=w)
            d = bb.fc.in_features
            bb.fc = nn.Identity()
            self.backbone = bb
            self.classifier = nn.Sequential(
                nn.Linear(d, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, 1)
            )

        def forward(self, x):
            return self.classifier(self.backbone(x))


def ece_score(y, p, n=15):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    bins = np.linspace(0, 1, n + 1)
    e = 0.0
    for i in range(n):
        m = (p > bins[i]) & (p <= bins[i + 1]) if i else (p >= bins[i]) & (p <= bins[i + 1])
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics_of(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    pred = (p >= 0.5).astype(int)
    return dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        sensitivity=float(recall_score(y, pred, zero_division=0)),
        specificity=float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1)),
        f1=float(f1_score(y, pred, zero_division=0)),
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        ece=ece_score(y, p),
        brier=float(np.mean((p - y) ** 2)),
    )


def flip_table(y, b, a):
    y = np.asarray(y).ravel().astype(int)
    b = (np.asarray(b).ravel() >= 0.5).astype(int)
    a = (np.asarray(a).ravel() >= 0.5).astype(int)
    return dict(
        correction_rate=float(((b != y) & (a == y)).sum() / max((b != y).sum(), 1)),
        harm_rate=float(((b == y) & (a != y)).sum() / max((b == y).sum(), 1)),
        flip_rate=float((a != b).mean()),
        correction=int(((b != y) & (a == y)).sum()),
        harmful=int(((b == y) & (a != y)).sum()),
    )


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


def eval_mode(m):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def mc_mode(m):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
        elif isinstance(mod, nn.Dropout):
            mod.train()


def tta_mode(m):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.eval()
        elif isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def entropy(z):
    p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).view(-1)


@torch.no_grad()
def mc_predict(m, x, n=N_MC):
    mc_mode(m)
    ps = [torch.sigmoid(m(x)).view(-1) for _ in range(n)]
    P = torch.stack(ps, 0)
    return P.mean(0), P.var(0, unbiased=False)


@torch.no_grad()
def evaluate(m, ld):
    eval_mode(m)
    ys, ps = [], []
    for x, y in ld:
        x = x.to(device)
        ps.append(torch.sigmoid(m(x)).view(-1).cpu().numpy())
        ys.append(y.view(-1).numpy())
    return metrics_of(np.concatenate(ys), np.concatenate(ps))


@torch.no_grad()
def collect_scores(m, ld, mc=False):
    ys, ps, us, ents, confs = [], [], [], [], []
    for x, y in tqdm(ld, desc="score"):
        x = x.to(device)
        if mc:
            p, u = mc_predict(m, x)
        else:
            eval_mode(m)
            p = torch.sigmoid(m(x)).view(-1)
            u = torch.zeros_like(p)
        p = p.clamp(1e-6, 1 - 1e-6)
        ys.append(y.view(-1).numpy())
        ps.append(p.cpu().numpy())
        us.append(u.cpu().numpy())
        ents.append((-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).cpu().numpy())
        confs.append(torch.maximum(p, 1 - p).cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(us),
        np.concatenate(ents),
        np.concatenate(confs),
    )


def adapt_stream(m, ld, gate, tau=None, target_cov=None, e_margin=None):
    """Continual 1-step BN-affine. Frozen-source scores for gates."""
    src = deepcopy(m)
    eval_mode(src)
    tta_mode(m)
    opt = torch.optim.Adam(bn_affine(m), lr=TTA_LR)
    y_all, before, after, accepted = [], [], [], []
    n_skip = 0
    rng = np.random.RandomState(SEED)
    for x, y in tqdm(ld, desc=str(gate)):
        x = x.to(device)
        with torch.no_grad():
            p0 = torch.sigmoid(src(x)).view(-1)
            if gate == "utta":
                _, u = mc_predict(src, x)
                w = (u < tau).float()
            elif gate == "confidence":
                w = (torch.maximum(p0, 1 - p0) >= tau).float()
            elif gate == "entropy":
                z = src(x)
                w = (entropy(z) < tau).float()
            elif gate == "random":
                w = torch.tensor(
                    (rng.rand(x.size(0)) < float(target_cov)).astype(np.float32),
                    device=x.device,
                )
            elif gate == "eata":
                z = src(x)
                w = (entropy(z) < e_margin).float()
            else:
                w = torch.ones(x.size(0), device=x.device)
        before.append(p0.detach().cpu().numpy())
        y_all.append(y.view(-1).numpy())
        accepted.append(w.detach().cpu().numpy())
        tta_mode(m)
        z = m(x)
        h = entropy(z)
        if float(w.sum()) < 1:
            n_skip += 1
            opt.zero_grad(set_to_none=True)
        else:
            loss = (w * h).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            eval_mode(m)
            after.append(torch.sigmoid(m(x)).view(-1).cpu().numpy())
            tta_mode(m)
    y = np.concatenate(y_all)
    b = np.concatenate(before)
    a = np.concatenate(after)
    cov = float(np.concatenate(accepted).mean())
    return {
        "gate": gate,
        "coverage": cov,
        "n_skip_batch": n_skip,
        "before": metrics_of(y, b),
        "after": metrics_of(y, a),
        "harm": flip_table(y, b, a),
    }


def main():
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    train_set = HFCamelyon(raw["train"], {0, 3, 4}, train_tf)
    ood_set = HFCamelyon(raw["validation"], {1}, eval_tf)
    test_set = HFCamelyon(raw["test"], {2}, eval_tf)
    train_loader = DataLoader(
        train_set, batch_size=BS_TRAIN, shuffle=True, num_workers=0, pin_memory=True
    )
    ood_loader = DataLoader(
        ood_set, batch_size=BS_EVAL, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BS_EVAL, shuffle=False, num_workers=0, pin_memory=True
    )

    CKPT = WORK / "camelyon17_resnet50_source_s42_BEST.pt"
    if CKPT.exists():
        st = torch.load(CKPT, map_location="cpu", weights_only=False)
        print("loaded", CKPT, "epoch", st.get("epoch"), "val_auroc", st.get("val_auroc"))
        best_state = st["model"]
    else:
        print("Training ResNet-50 source seed 42, early-stop hospital 1")
        model = Net(pretrained=True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=SRC_LR, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda")
        best, stale, best_state, best_ep = -1.0, 0, None, -1
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            run = n = 0
            for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}"):
                x, y = x.to(device), y.view(-1, 1).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.item() * x.size(0)
                n += x.size(0)
            val = evaluate(model, ood_loader)
            history.append({"epoch": epoch, "train_loss": run / max(n, 1), "val_auroc": val["auroc"], "val_f1": val["f1"]})
            print(f"epoch {epoch:02d} loss={run/max(n,1):.4f} val_auroc={val['auroc']:.4f} val_f1={val['f1']:.4f}")
            if val["auroc"] > best:
                best, stale, best_ep = val["auroc"], 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {"model": best_state, "epoch": epoch, "val_auroc": best, "seed": SEED, "backbone": "resnet50", "history": history},
                    CKPT,
                )
                print("  saved BEST", CKPT)
            else:
                stale += 1
                if stale >= PATIENCE:
                    print("early stop at", epoch, "best", best)
                    break
        st = torch.load(CKPT, map_location="cpu", weights_only=False)

    def load_fresh():
        m = Net(pretrained=False).to(device)
        m.load_state_dict(st["model"])
        return m

    src = load_fresh()
    ood_src = evaluate(src, ood_loader)
    print("ASSERT OOD", round(ood_src["auroc"], 4), "ckpt", st.get("val_auroc"))
    assert abs(ood_src["auroc"] - float(st["val_auroc"])) < 0.015, "restore mismatch"

    print("--- MC-Dropout U on hospital 1 ---")
    yv, pv, uv, ev, cv = collect_scores(src, ood_loader, mc=True)
    err = ((pv >= 0.5).astype(int) != yv.astype(int)).astype(float)
    corr = float(np.corrcoef(uv, err)[0, 1])
    tau = float(np.quantile(uv, 0.70))
    print("corr(U,err)", round(corr, 4), "tau 70th", tau)

    # matched conf/ent on val
    order_c = np.argsort(-cv)
    n_keep = int(round(0.70 * len(cv)))
    conf_t = float(cv[order_c[n_keep - 1]])
    order_e = np.argsort(ev)
    ent_t = float(ev[order_e[n_keep - 1]])
    print("conf_t", conf_t, "ent_t", ent_t)

    src_test = evaluate(src, test_loader)
    print("source test AUROC", round(src_test["auroc"], 4), "F1", round(src_test["f1"], 4))

    results = {
        "backbone": "resnet50",
        "seed": 42,
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": TTA_LR,
            "n_mc": N_MC,
            "tau_rule": "70th percentile of val U",
            "e_margin": E_MARGIN,
        },
        "best_epoch": st.get("epoch"),
        "val_ood": ood_src,
        "u_error_corr": corr,
        "tau": tau,
        "conf_threshold": conf_t,
        "entropy_threshold": ent_t,
        "source_test": src_test,
    }

    print("--- TEST once (hospital 2) ---")
    for name, kwargs in [
        ("tent", dict(gate="tent")),
        ("eata", dict(gate="eata", e_margin=E_MARGIN)),
        ("confidence", dict(gate="confidence", tau=conf_t)),
        ("utta", dict(gate="utta", tau=tau)),
    ]:
        m = load_fresh()
        rec = adapt_stream(m, test_loader, **kwargs)
        results[f"{name}_test"] = rec
        a = rec["after"]
        print(
            f"{name:12s} AUROC={a['auroc']:.4f} F1={a['f1']:.4f} ECE={a['ece']:.4f} "
            f"cov={rec['coverage']:.3f} harm={rec['harm']['harm_rate']:.4f}"
        )

    out = WORK / "camelyon17_resnet50_full_s42.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)
    print("also download", CKPT)
    print("Compare to R18 s42: source 0.9355  utta 0.9507  harm 0.003  conf 0.9508")
    print("PHASE 11: do not put these numbers in the 5-seed table.")


if __name__ == "__main__":
    main()
