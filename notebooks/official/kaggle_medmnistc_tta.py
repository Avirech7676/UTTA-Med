#!/usr/bin/env python3
"""X3 — PathMNIST + TTA under MedMNIST-C-style corruptions.
Source-only PathMNIST-C already exists. This adds Tent / confidence / UTTA on C.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


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


ensure_libraries(["torch", "torchvision", "numpy", "PIL", "sklearn", "scipy", "tqdm", "medmnist"])

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    corrupt_batch,
    seed_all,
    set_tta_mode,
)

SEED = 42
N_CLASS = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else str(HERE.parent.parent / "results" / "optional" / "pathmnist")))
OUT.mkdir(parents=True, exist_ok=True)
BS = 128
EPOCHS = 6


def to_labels_np(y):
    """Safely convert label tensor/array to 1D integer class array."""
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)
    if y.ndim > 1:
        if y.shape[-1] == 1:
            return y.squeeze(-1).astype(int)
        elif y.shape[-1] > 1:
            return y.argmax(-1).astype(int)
    return y.astype(int)


def entropy_mc(p, eps=1e-8):
    return -(p * np.log(p + eps)).sum(-1)


def metrics9(y, p):
    y = to_labels_np(y)
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    f1s = []
    for c in range(N_CLASS):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f1s.append(2 * pr * rc / (pr + rc + 1e-8))
    from sklearn.metrics import roc_auc_score

    try:
        auroc = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(N_CLASS))))
    except Exception:
        auroc = float("nan")
    return {"n": int(len(y)), "accuracy": acc, "macro_f1": float(np.mean(f1s)), "auroc_ovr": auroc}


@torch.no_grad()
def predict9(model, loader, corrupt=None, sev=0):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        if corrupt and corrupt != "identity":
            x = corrupt_batch(x, corrupt, sev, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        p = torch.softmax(model(x), 1).cpu().numpy()
        ys.append(to_labels_np(y))
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def mc9(model, loader):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    ys, acc = [], []
    for x, y in tqdm(loader, desc="MC", leave=False):
        x = x.to(DEVICE)
        draws = [torch.softmax(model(x), 1).cpu() for _ in range(N_MC)]
        acc.append(torch.stack(draws, 0).numpy())
        ys.append(to_labels_np(y))
    P = np.concatenate(acc, 1)
    y = np.concatenate(ys)
    pbar = P.mean(0)
    u = P.var(0).mean(1)
    return y, pbar, u


def tta9(model, loader, gate, tau=None, conf_t=None, corrupt=None, sev=0):
    params = set_tta_mode(model)
    opt = torch.optim.Adam(params, lr=1e-5)
    frozen = None
    if gate == "utta":
        frozen = ResNetMC("resnet18", N_CLASS).to(DEVICE)
        frozen.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    ys, p0s, p1s, keep = [], [], [], []
    for x, y in tqdm(loader, desc=f"{gate}/{corrupt}/{sev}", leave=False):
        x = x.to(DEVICE)
        if corrupt and corrupt != "identity":
            x = corrupt_batch(x, corrupt, sev, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ynp = to_labels_np(y)
        with torch.no_grad():
            p0 = torch.softmax(model(x), 1)
        if gate == "tent":
            w = torch.ones(x.size(0), device=DEVICE)
        elif gate == "confidence":
            w = (p0.max(1).values >= float(conf_t)).float()
        elif gate == "utta":
            frozen.train()
            for m in frozen.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
            with torch.no_grad():
                P = torch.stack([torch.softmax(frozen(x), 1) for _ in range(N_MC)], 0)
            u = P.var(0).mean(1)
            w = (u < float(tau)).float()
            frozen.eval()
        else:
            raise ValueError(gate)
        keep.append(w.detach().cpu().numpy())
        if float(w.sum()) < 1:
            p1 = p0
        else:
            p = torch.softmax(model(x), 1)
            ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1)
            loss = (w * ent).sum() / w.sum()
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                p1 = torch.softmax(model(x), 1)
        ys.append(ynp)
        p0s.append(p0.detach().cpu().numpy())
        p1s.append(p1.detach().cpu().numpy())
    y = np.concatenate(ys)
    p1 = np.concatenate(p1s)
    cov = float(np.concatenate(keep).mean())
    return y, p1, cov


def ensure_pathmnist():
    """Ensure pathmnist.npz exists in ~/.medmnist/ with robust curl resume and mirror fallbacks."""
    medmnist_root = Path.home() / ".medmnist"
    medmnist_root.mkdir(parents=True, exist_ok=True)
    target = medmnist_root / "pathmnist.npz"

    if target.exists() and target.stat().st_size > 200_000_000:
        print(f"pathmnist.npz already present at {target} ({target.stat().st_size} bytes)")
        return target

    # Check search locations (e.g. attached Kaggle datasets or local workspace)
    cands = []
    if Path("/kaggle/input").exists():
        cands += list(Path("/kaggle/input").rglob("pathmnist.npz"))
    if Path("/kaggle/working").exists():
        cands += list(Path("/kaggle/working").rglob("pathmnist.npz"))
    cands += list(HERE.parent.parent.rglob("pathmnist.npz"))
    for c in cands:
        if c.is_file() and c.stat().st_size > 200_000_000:
            print(f"Found existing pathmnist.npz at {c}. Copying to {target}...")
            import shutil
            shutil.copyfile(c, target)
            return target

    # Zenodo direct download with retry and resume
    urls = [
        "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
        "https://zenodo.org/record/6496656/files/pathmnist.npz?download=1",
    ]
    for url in urls:
        print(f"Downloading pathmnist.npz directly from Zenodo ({url})...")
        import subprocess
        curl_bin = "curl.exe" if sys.platform == "win32" else "curl"
        try:
            # -C - resumes interrupted downloads; -L follows redirects; --retry 5 retries transient failures
            cmd = [curl_bin, "-L", "--retry", "5", "--retry-delay", "2", "-C", "-", url, "-o", str(target)]
            res = subprocess.run(cmd, timeout=900)
            if target.exists() and target.stat().st_size > 200_000_000:
                print(f"Successfully downloaded pathmnist.npz ({target.stat().st_size} bytes)")
                return target
        except Exception as e:
            print(f"curl attempt failed: {e}")

        # Fallback to urllib with user agent
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=900) as resp, open(target, "wb") as f:
                f.write(resp.read())
            if target.exists() and target.stat().st_size > 200_000_000:
                print(f"Successfully downloaded pathmnist.npz via urllib ({target.stat().st_size} bytes)")
                return target
        except Exception as e:
            print(f"urllib attempt failed: {e}")

    return target


def main():
    seed_all(SEED)
    print("device", DEVICE)
    try:
        import medmnist
        from medmnist import PathMNIST
    except ImportError:
        import subprocess

        print("Installing medmnist...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "medmnist"])
        import medmnist
        from medmnist import PathMNIST

    tf = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    # override kit denorm for 0.5/0.5 — corrupt_batch uses ImageNet mean.
    # PathMNIST is 28x28; we still apply C in pixel space via denorm of 0.5.
    ensure_pathmnist()
    try:
        train = PathMNIST(split="train", download=False, transform=tf)
        val = PathMNIST(split="val", download=False, transform=tf)
        test = PathMNIST(split="test", download=False, transform=tf)
    except Exception as e:
        print(f"Loading local cache failed ({e}), attempting medmnist auto-download fallback...")
        train = PathMNIST(split="train", download=True, transform=tf)
        val = PathMNIST(split="val", download=True, transform=tf)
        test = PathMNIST(split="test", download=True, transform=tf)
    print("pathmnist", len(train), len(val), len(test), medmnist.__version__)
    nw = 2 if torch.cuda.is_available() and sys.platform != "win32" else 0
    tr = DataLoader(train, batch_size=BS, shuffle=True, num_workers=nw)
    va = DataLoader(val, batch_size=BS, shuffle=False, num_workers=nw)
    te = DataLoader(test, batch_size=BS, shuffle=False, num_workers=nw)

    model = ResNetMC("resnet18", N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in tr:
            x = x.to(DEVICE)
            y = y.squeeze().long().to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        yv, pv = predict9(model, va)
        print(f"epoch {ep} loss={np.mean(losses):.4f} val_acc={metrics9(yv, pv)['accuracy']:.4f}")
    st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(st, OUT / "pathmnist_r18_s42.pt")
    yt, pt = predict9(model, te)
    src = metrics9(yt, pt)
    print("SOURCE", src)
    if src["accuracy"] < 0.85:
        raise SystemExit("source acc < 0.85")

    yv, pbar, u = mc9(model, va)
    tau = float(np.percentile(u, TAU_PCT))
    conf_t = float(np.quantile(pbar.max(1), 1.0 - (u < tau).mean()))
    print("tau", tau, "conf_t", conf_t)

    c_rows = {}
    for name, sev in [
        ("identity", 0),
        ("gaussian_noise", 1),
        ("gaussian_noise", 3),
        ("defocus_blur", 3),
        ("brightness", 3),
        ("jpeg", 3),
    ]:
        y, p = predict9(model, te, None if name == "identity" else name, sev)
        c_rows[f"source_{name}_{sev}"] = metrics9(y, p)
        print("source", name, sev, c_rows[f"source_{name}_{sev}"])

    tta_rows = {}
    for corrupt, sev in [("identity", 0), ("gaussian_noise", 3), ("jpeg", 3)]:
        for gate, kw in [("tent", {}), ("confidence", {"conf_t": conf_t}), ("utta", {"tau": tau})]:
            m = ResNetMC("resnet18", N_CLASS).to(DEVICE)
            m.load_state_dict(st)
            y, p1, cov = tta9(m, te, gate, corrupt=None if corrupt == "identity" else corrupt, sev=sev, **kw)
            met = {**metrics9(y, p1), "coverage": cov}
            tta_rows[f"{gate}_{corrupt}_{sev}"] = met
            print(gate, corrupt, sev, met)

    blob = {
        "dataset": "PathMNIST + synthetic C",
        "source": src,
        "source_C": c_rows,
        "tta_C": tta_rows,
        "tau": tau,
        "note": "Supplement. Not Camelyon17 Table 1. TTA under C; do not claim robustness if TTA fails.",
    }
    (OUT / "pathmnist_c_tta_s42.json").write_text(json.dumps(blob, indent=2))
    print("wrote", OUT / "pathmnist_c_tta_s42.json")


if __name__ == "__main__":
    main()
