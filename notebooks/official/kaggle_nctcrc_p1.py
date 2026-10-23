#!/usr/bin/env python3
"""P1 — NCT-CRC-HE 9-class, frozen Camelyon17 TTA knobs.

Train on NCT-CRC-HE-100K. τ + early-stop on 10% of 100K.
Target = CRC-VAL-HE-7K (different patients). Labels = eval only.

Goal: in 9-class, MC-Dropout U is NOT a monotone function of confidence/entropy.
Do not retune. Do not mix into Camelyon17 Table 1.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def ensure_libraries(packages: list[str]) -> None:
    """Automatically ensure required packages are installed."""
    import importlib.util
    import subprocess

    pkg_map = {
        "sklearn": "scikit-learn",
        "PIL": "pillow",
    }
    missing = []
    for pkg in packages:
        pip_name = pkg_map.get(pkg, pkg)
        if importlib.util.find_spec(pkg) is None:
            missing.append(pip_name)
    if missing:
        print(f"[*] Missing libraries detected: {missing}. Installing via pip...", flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
            print(f"[+] Successfully installed: {missing}", flush=True)
        except Exception as e:
            print(f"[!] Warning: Auto-install failed for {missing}: {e}", flush=True)


ensure_libraries(["torch", "torchvision", "numpy", "PIL", "sklearn", "scipy", "tqdm", "datasets"])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm

SEED = 42
N_CLASS = 9
NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TTA_LR = 1e-5
N_MC = 20
TAU_PCT = 70
EPOCHS = 8
PATIENCE = 3
BS_TRAIN = 64
BS_TTA = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class ResNet18MC(nn.Module):
    def __init__(self, n_classes=9, dropout=0.5):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(*list(m.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.stem(x).flatten(1)
        return self.fc(self.dropout(x))


def collect_bn(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def set_tta_mode(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # freeze running mean/var
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    return params


def softmax_np(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy_np(p, eps=1e-8):
    return -(p * np.log(p + eps)).sum(axis=-1)


def metrics_mc(y, p, n_class=9):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    # macro F1
    f1s = []
    for c in range(n_class):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    # ECE
    conf = p.max(1)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    n = len(y)
    for i in range(15):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < 14 else conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs((pred[m] == y[m]).mean() - conf[m].mean())
    # one-vs-rest AUROC (skip missing class)
    from sklearn.metrics import roc_auc_score

    try:
        auroc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "auroc_ovr": auroc,
        "ece": float(ece),
        "mean_confidence": float(conf.mean()),
        "mean_entropy": float(entropy_np(p).mean()),
    }


def harm_mc(y, p0, p1):
    y = np.asarray(y).astype(int)
    a = p0.argmax(1) == y
    b = p1.argmax(1) == y
    n_ok = max(int(a.sum()), 1)
    n_bad = max(int((~a).sum()), 1)
    return {
        "harm_rate": float((a & ~b).sum() / n_ok),
        "correction_rate": float((~a & b).sum() / n_bad),
        "harmful": int((a & ~b).sum()),
        "correction": int((~a & b).sum()),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TF_TRAIN = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
TF_EVAL = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class FolderCRC(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.samples = []
        root = Path(root)
        for i, name in enumerate(NAMES):
            d = root / name
            if not d.is_dir():
                # case-insensitive
                hits = [p for p in root.iterdir() if p.is_dir() and p.name.upper() == name]
                d = hits[0] if hits else None
            if d is None:
                continue
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                    self.samples.append((p, i))
        if not self.samples:
            raise FileNotFoundError(f"no images under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


class HFCRC(Dataset):
    def __init__(self, hf, transform, label_key="label"):
        self.hf = hf
        self.transform = transform
        self.label_key = label_key
        feat = hf.features
        self.img_key = "image" if "image" in feat else [k for k in feat if "image" in k.lower()][0]

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        row = self.hf[int(i)]
        img = row[self.img_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img)).convert("RGB")
        else:
            img = img.convert("RGB")
        y = int(row[self.label_key])
        return self.transform(img), y


def find_class_root(start: Path) -> Optional[Path]:
    """Directory that contains ADI/BACK/... subfolders (fast shallow search)."""
    if not start.exists():
        return None
    want = set(NAMES)
    for root, dirs, _ in os.walk(start):
        dir_set = {d.upper() for d in dirs}
        if want.issubset(dir_set) or len(want & dir_set) >= 8:
            return Path(root)
        # Avoid traversing into image-heavy class folders
        dirs[:] = [d for d in dirs if d.upper() not in want]
    return None


def load_crc():
    """Return (train_ds_eval_tf, target_ds). Train gets eval tf; caller wraps train tf."""
    # 1) Search mounted Kaggle datasets first
    train_root = None
    val7k_root = None
    search_dirs = []
    if os.environ.get("NCT_DIR"):
        search_dirs.append(Path(os.environ["NCT_DIR"]))
    if os.environ.get("CRC_DIR"):
        search_dirs.append(Path(os.environ["CRC_DIR"]))
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").iterdir():
            if p.is_dir():
                search_dirs.append(p)
                try:
                    search_dirs.extend([sub for sub in p.iterdir() if sub.is_dir()])
                except Exception:
                    pass
    search_dirs.extend([Path("/kaggle/working"), Path("/tmp")])

    for p in search_dirs:
        r = find_class_root(p)
        if r is not None:
            name = str(r).upper()
            if "7K" in name or "VAL" in name or "CRC-VAL" in name:
                val7k_root = r
            elif "100K" in name or "NCT" in name:
                train_root = r
            elif train_root is None:
                train_root = r
            elif val7k_root is None:
                val7k_root = r

    if train_root and val7k_root:
        print(f"[+] Loaded from mounted folders: train={train_root}, 7k={val7k_root}", flush=True)
        return FolderCRC(train_root, TF_EVAL), FolderCRC(val7k_root, TF_EVAL)

    # 2) HuggingFace (preferred online source)
    try:
        try:
            import datasets
        except ImportError:
            import subprocess
            print("[*] Installing datasets package for HuggingFace...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], timeout=120)
            import datasets

        from datasets import load_dataset

        print("[*] Loading from HuggingFace dataset '1aurent/NCT-CRC-HE'...", flush=True)
        d = load_dataset("1aurent/NCT-CRC-HE")
        keys = list(d.keys())
        print(f"[+] HuggingFace splits available: {keys}", flush=True)

        # Match 100K training split and 7K validation split
        train_key = next((k for k in keys if "100K" in k.upper() and "NONORM" not in k.upper()), None)
        if not train_key:
            train_key = next((k for k in keys if "100K" in k.upper()), None)
        val_key = next((k for k in keys if "7K" in k.upper() or "VAL" in k.upper()), None)

        if train_key and val_key:
            print(f"[+] Using HuggingFace train split: '{train_key}' ({len(d[train_key])} samples)", flush=True)
            print(f"[+] Using HuggingFace target split: '{val_key}' ({len(d[val_key])} samples)", flush=True)
            return HFCRC(d[train_key], TF_EVAL), HFCRC(d[val_key], TF_EVAL)
        elif len(keys) >= 2:
            print(f"[!] Warning: Falling back to positional splits: train='{keys[1]}', val='{keys[0]}'", flush=True)
            return HFCRC(d[keys[1]], TF_EVAL), HFCRC(d[keys[0]], TF_EVAL)
    except Exception as e:
        print(f"[!] HuggingFace load failed: {type(e).__name__}: {e}", flush=True)

    # 3) Zenodo fallback (chunked download with retries)
    zdir = Path("/kaggle/working/nctcrc")
    zdir.mkdir(parents=True, exist_ok=True)
    urls = {
        "CRC-VAL-HE-7K.zip": "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1",
        "NCT-CRC-HE-100K.zip": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1",
    }
    import subprocess
    import urllib.request
    import zipfile

    def download_file(url, path):
        print(f"[*] Downloading {path.name} from {url}...", flush=True)
        curl_bin = "curl.exe" if sys.platform == "win32" else "curl"
        try:
            cmd = [curl_bin, "-L", "--retry", "5", "--retry-delay", "2", "-C", "-", url, "-o", str(path)]
            res = subprocess.run(cmd, timeout=3600)
            if path.exists() and path.stat().st_size > 1_000_000:
                print(f"[+] Downloaded {path.name} via curl ({path.stat().st_size} bytes)", flush=True)
                return
        except Exception as e:
            print(f"[!] curl failed: {e}", flush=True)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=3600) as resp, open(path, "wb") as out_file:
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            block_size = 2 * 1024 * 1024
            last_pct = -1
            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                downloaded += len(buf)
                out_file.write(buf)
                pct = int(downloaded * 100 / total) if total > 0 else 0
                if pct != last_pct and pct % 10 == 0:
                    last_pct = pct
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total / (1024 * 1024)
                    print(f"    Progress: {pct}% ({mb:.1f}MB / {tot_mb:.1f}MB)", flush=True)

    for name, url in urls.items():
        zp = zdir / name
        if not zp.exists() or zp.stat().st_size < 1_000_000:
            download_file(url, zp)
        dest = zdir / name.replace(".zip", "")
        if not find_class_root(dest):
            print(f"[*] Unpacking {name}...", flush=True)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(dest)

    train_root = find_class_root(zdir / "NCT-CRC-HE-100K") or find_class_root(zdir)
    val7k_root = find_class_root(zdir / "CRC-VAL-HE-7K")
    if train_root is None or val7k_root is None:
        raise RuntimeError(
            "Could not load NCT-CRC-HE. Add a Kaggle dataset containing "
            "ADI/BACK/... folders for 100K and 7K, or enable internet for HF/Zenodo."
        )
    return FolderCRC(train_root, TF_EVAL), FolderCRC(val7k_root, TF_EVAL)


class RelabelTrain(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        # base already applied TF_EVAL; re-train aug needs PIL. Skip flip on tensor:
        if self.transform is TF_TRAIN:
            # x is tensor — apply tensor flips
            if random.random() < 0.5:
                x = torch.flip(x, [2])
            if random.random() < 0.5:
                x = torch.flip(x, [1])
        return x, y


def stratified_holdout(ds, frac=0.1, seed=42):
    ys = []
    # cheap label pass
    if hasattr(ds, "samples"):
        ys = np.array([y for _, y in ds.samples])
    elif hasattr(ds, "hf"):
        key = ds.label_key
        try:
            ys = np.array(ds.hf[key], dtype=int)
        except Exception:
            ys = np.array([int(ds.hf[i][key]) for i in range(len(ds))])
    else:
        ys = np.array([ds[i][1] for i in range(len(ds))])
    rng = np.random.RandomState(seed)
    val_idx, tr_idx = [], []
    for c in range(N_CLASS):
        ix = np.where(ys == c)[0]
        rng.shuffle(ix)
        n_v = max(1, int(round(len(ix) * frac)))
        val_idx.extend(ix[:n_v].tolist())
        tr_idx.extend(ix[n_v:].tolist())
    return tr_idx, val_idx, ys


# ---------------------------------------------------------------------------
# Inference / TTA
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(DEVICE)
        p = torch.softmax(model(x), 1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def mc_predict(model, loader, n_mc=N_MC):
    model.train()  # dropout on
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    ys, acc = [], []
    for x, y in tqdm(loader, desc="MC", leave=False):
        x = x.to(DEVICE)
        draws = []
        for _ in range(n_mc):
            draws.append(torch.softmax(model(x), 1).cpu())
        P = torch.stack(draws, 0).numpy()  # N,B,C
        acc.append(P)
        ys.append(y.numpy())
    P = np.concatenate(acc, 1)  # N, n, C
    y = np.concatenate(ys)
    pbar = P.mean(0)
    u = P.var(0).mean(1)  # mean class-wise MC variance
    return y, pbar, u, P


def tta_adapt(model, loader, gate, tau=None, conf_t=None, ent_t=None, p_keep=None, y_all=None):
    """One pass, k=1, bn freeze stats. gate in {tent, random, confidence, entropy, utta}."""
    params = set_tta_mode(model)
    opt = torch.optim.Adam(params, lr=TTA_LR)
    model.eval()  # deterministic eval for TTA entropy minimization (MC U comes from frozen copy)
    n = len(loader.dataset)
    p_before = np.zeros((n, N_CLASS), np.float32)
    p_after = np.zeros((n, N_CLASS), np.float32)
    y = np.zeros(n, np.int64)
    w_keep = np.zeros(n, np.float32)
    i0 = 0
    rng = np.random.RandomState(SEED)
    frozen = None
    if gate == "utta":
        frozen = ResNet18MC().to(DEVICE)
        frozen.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
        frozen.eval()

    for x, yb in tqdm(loader, desc=gate, leave=False):
        b = x.size(0)
        sl = slice(i0, i0 + b)
        x = x.to(DEVICE)
        y[sl] = yb.numpy()
        with torch.no_grad():
            p0 = torch.softmax(model(x), 1)
            p_before[sl] = p0.cpu().numpy()

        if gate == "tent":
            w = torch.ones(b, device=DEVICE)
        elif gate == "random":
            w = torch.tensor(rng.rand(b) < float(p_keep), device=DEVICE, dtype=torch.float32)
        elif gate == "confidence":
            c = p0.max(1).values
            w = (c >= float(conf_t)).float()
        elif gate == "entropy":
            ent = -(p0 * torch.log(p0.clamp_min(1e-8))).sum(1)
            w = (ent <= float(ent_t)).float()
        elif gate == "utta":
            frozen.train()
            for m in frozen.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            draws = []
            with torch.no_grad():
                for _ in range(N_MC):
                    draws.append(torch.softmax(frozen(x), 1))
            P = torch.stack(draws, 0)
            u = P.var(0).mean(1)
            w = (u < float(tau)).float()
            frozen.eval()
        else:
            raise ValueError(gate)

        w_keep[sl] = w.detach().cpu().numpy()
        if float(w.sum()) < 1:
            p_after[sl] = p_before[sl]
            i0 += b
            continue
        logits = model(x)
        p = torch.softmax(logits, 1)
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1)
        loss = (w * ent).sum() / w.sum()
        if torch.isfinite(loss):
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p_after[sl] = torch.softmax(model(x), 1).cpu().numpy()
        i0 += b
    return y, p_before, p_after, float(w_keep.mean())


def main():
    seed_all()
    print("device", DEVICE)
    train_full, target_ds = load_crc()
    print("train_full", len(train_full), "target", len(target_ds))
    tr_idx, va_idx, ys = stratified_holdout(train_full, 0.1, SEED)
    print("split train", len(tr_idx), "val", len(va_idx), Counter(ys[va_idx].tolist() if hasattr(ys, "tolist") else ys))
    train_ds = RelabelTrain(train_full, tr_idx, TF_TRAIN)
    val_ds = RelabelTrain(train_full, va_idx, TF_EVAL)
    train_loader = DataLoader(train_ds, batch_size=BS_TRAIN, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BS_TTA, shuffle=False, num_workers=2)
    tgt_loader = DataLoader(target_ds, batch_size=BS_TTA, shuffle=False, num_workers=2)

    model = ResNet18MC().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_acc, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"train {ep}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        yv, pv = predict_loader(model, val_loader)
        acc = float((pv.argmax(1) == yv).mean())
        print(f"epoch {ep} loss={np.mean(losses):.4f} val_acc={acc:.4f} val_f1={metrics_mc(yv, pv)['macro_f1']:.4f}")
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_acc": acc, "epoch": ep}, OUT / "nctcrc_resnet18_s42_BEST.pt")
            print("  saved BEST")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop")
                break
    model.load_state_dict(best_state)
    model.to(DEVICE)
    print("loaded best val_acc", best_acc, "train_min", (time.time() - t0) / 60)

    # source target
    yt, pt = predict_loader(model, tgt_loader)
    src = metrics_mc(yt, pt)
    print("SOURCE 7K", src)
    if src["accuracy"] < 0.80:
        raise SystemExit("source accuracy < 0.80 — training failed, do not TTA")

    # MC on val → τ and rank correlations (the scientific check)
    yv, pbar, u, P = mc_predict(model, val_loader, N_MC)
    conf = pbar.max(1)
    ent = entropy_np(pbar)
    err = (pbar.argmax(1) != yv).astype(np.float64)
    tau = float(np.percentile(u, TAU_PCT))
    from scipy.stats import spearmanr

    sp_u_conf = float(spearmanr(u, 1.0 - conf).correlation)
    sp_u_ent = float(spearmanr(u, ent).correlation)
    sp_conf_ent = float(spearmanr(1.0 - conf, ent).correlation)
    sp_u_err = float(spearmanr(u, err).correlation)
    print(f"tau p{TAU_PCT}={tau:.6g}")
    print(f"Spearman U vs (1-conf)={sp_u_conf:.3f}  U vs entropy={sp_u_ent:.3f}  (1-conf) vs H={sp_conf_ent:.3f}  U vs err={sp_u_err:.3f}")
    print("BINARY Camelyon had (1-conf) vs H ≈ 1.0. 9-class should be < 1.")

    # match coverage on VAL for random/conf/entropy
    target_cov = float((u < tau).mean())
    conf_t = float(np.quantile(conf, 1.0 - target_cov))
    ent_t = float(np.quantile(ent, target_cov))
    print("matched val coverage", target_cov, "conf_t", conf_t, "ent_t", ent_t)

    def clone_best():
        m = ResNet18MC().to(DEVICE)
        m.load_state_dict(best_state)
        return m

    rows = {"source": {**src, "coverage": 0.0, "harm": None}}
    for gate, kw in [
        ("tent", {}),
        ("random", {"p_keep": target_cov}),
        ("confidence", {"conf_t": conf_t}),
        ("entropy", {"ent_t": ent_t}),
        ("utta", {"tau": tau}),
    ]:
        m = clone_best()
        y, p0, p1, cov = tta_adapt(m, tgt_loader, gate, **kw)
        met = metrics_mc(y, p1)
        h = harm_mc(y, p0, p1)
        rows[gate] = {**met, "coverage": cov, "harm": h}
        print(
            f"{gate:12s} acc={met['accuracy']:.4f} F1={met['macro_f1']:.4f} "
            f"AUROC={met['auroc_ovr']:.4f} ECE={met['ece']:.3f} cov={cov:.3f} harm={h['harm_rate']:.4f}"
        )

    blob = {
        "seed": SEED,
        "dataset": "NCT-CRC-HE-100K → CRC-VAL-HE-7K",
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": TTA_LR,
            "n_mc": N_MC,
            "tau_rule": "70th percentile of val mean-class MC variance",
            "val": "10% stratified of 100K",
            "target_labels": "eval only",
        },
        "tau": tau,
        "spearman": {
            "U_vs_one_minus_confidence": sp_u_conf,
            "U_vs_entropy": sp_u_ent,
            "one_minus_confidence_vs_entropy": sp_conf_ent,
            "U_vs_error": sp_u_err,
        },
        "matched_val_coverage": target_cov,
        "conf_threshold": conf_t,
        "entropy_threshold": ent_t,
        "val_acc_best": best_acc,
        "results": rows,
        "note": "Not Camelyon17 Table 1. 7K is different patients, not a hospital. "
        "RQ6 test: if Spearman(U, 1-conf) << 1, MC-Dropout is not redundant with confidence.",
    }
    (OUT / "nctcrc_p1_s42.json").write_text(json.dumps(blob, indent=2))
    print("wrote", OUT / "nctcrc_p1_s42.json")
    print("Download nctcrc_p1_s42.json and nctcrc_resnet18_s42_BEST.pt")


if __name__ == "__main__":
    main()
