"""Shared TTA helpers. Frozen: bn_freeze_stats, lr=1e-5, k=1, N_MC=20."""
from __future__ import annotations

import random
import sys


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


ensure_libraries(["sklearn", "tqdm", "torch", "torchvision", "numpy"])

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import models
from tqdm.auto import tqdm

TTA_LR = 1e-5
N_MC = 20
TAU_PCT = 70
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_all(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class ResNetMC(nn.Module):
    """1-logit if n_classes==1 else n-way. Dropout 0.5 before head."""

    def __init__(self, arch="resnet18", n_classes=1, dropout=0.5):
        super().__init__()
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            dim = 512
        elif arch == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            dim = 2048
        else:
            raise ValueError(arch)
        self.stem = nn.Sequential(*list(m.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = self.stem(x).flatten(1)
        z = self.fc(self.dropout(x))
        return z.squeeze(-1) if self.n_classes == 1 else z


def set_tta_mode(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    return params


def entropy_bin(p, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def metrics_bin(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).reshape(-1)
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    try:
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
    except Exception:
        auroc = auprc = float("nan")
    conf = np.maximum(p, 1 - p)
    bins = np.linspace(0, 1, 16)
    ece, n = 0.0, len(y)
    for i in range(15):
        hi = bins[i + 1] if i < 14 else 1.01
        m = (conf >= bins[i]) & (conf < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs((pred[m] == y[m]).mean() - conf[m].mean())
    brier = float(((p - y) ** 2).mean())
    return {
        "n": int(n),
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
        "ece": float(ece),
        "brier": brier,
    }


def harm_bin(y, p0, p1):
    y = np.asarray(y).astype(int)
    a = (p0 >= 0.5).astype(int) == y
    b = (p1 >= 0.5).astype(int) == y
    n_ok = max(int(a.sum()), 1)
    n_bad = max(int((~a).sum()), 1)
    return {
        "harm_rate": float((a & ~b).sum() / n_ok),
        "correction_rate": float((~a & b).sum() / n_bad),
        "harmful": int((a & ~b).sum()),
        "correction": int((~a & b).sum()),
    }


@torch.no_grad()
def predict_bin(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        ys.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps).reshape(-1)


@torch.no_grad()
def mc_bin(model, loader, device, n_mc=N_MC):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    ys, draws = [], []
    for x, y in tqdm(loader, desc="MC", leave=False):
        x = x.to(device)
        d = [torch.sigmoid(model(x)).cpu() for _ in range(n_mc)]
        draws.append(torch.stack(d, 0).numpy())
        ys.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
    P = np.concatenate(draws, 1)  # N, n
    y = np.concatenate(ys)
    pbar = P.mean(0).reshape(-1)
    u = P.var(0).reshape(-1)
    return y, pbar, u


def tta_bin(
    model,
    loader,
    device,
    gate,
    tau=None,
    conf_t=None,
    ent_t=None,
    p_keep=None,
    seed=42,
    lambda_anchor=0.01,
    lambda_div=0.05,
    theta_high=0.95,
    theta_low=0.65,
):
    """One pass k=1. Frozen-source U for utta with anchor & diversity stabilization."""
    params = set_tta_mode(model)
    init_params = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=TTA_LR)
    n = len(loader.dataset)
    p0 = np.zeros(n, np.float32)
    p1 = np.zeros(n, np.float32)
    y = np.zeros(n, np.int64)
    kept = np.zeros(n, np.float32)
    i0 = 0
    rng = np.random.RandomState(seed)
    frozen = None
    if gate in ("utta", "hybrid"):
        frozen = type(model)(arch=getattr(model, "_arch", "resnet18"), n_classes=1).to(device)
        frozen.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})

    for x, yb in tqdm(loader, desc=gate, leave=False):
        b = x.size(0)
        sl = slice(i0, i0 + b)
        x = x.to(device)
        y[sl] = yb.numpy() if torch.is_tensor(yb) else np.asarray(yb)
        with torch.no_grad():
            pb = torch.sigmoid(model(x))
            p0[sl] = pb.cpu().numpy()
        if gate == "tent":
            w = torch.ones(b, device=device)
        elif gate == "eata":
            ent = entropy_bin(pb.detach().cpu().numpy())
            # e_margin 0.4 on binary entropy (max ln2≈0.693)
            w = torch.tensor(ent < 0.4, device=device, dtype=torch.float32)
        elif gate == "random":
            w = torch.tensor(rng.rand(b) < float(p_keep), device=device, dtype=torch.float32)
        elif gate == "confidence":
            c = torch.maximum(pb, 1 - pb)
            w = (c >= float(conf_t)).float()
        elif gate == "entropy":
            pclip = pb.clamp(1e-8, 1 - 1e-8)
            ent = -(pclip * torch.log(pclip) + (1 - pclip) * torch.log(1 - pclip))
            w = (ent <= float(ent_t)).float()
        elif gate == "utta":
            frozen.train()
            for m in frozen.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            with torch.no_grad():
                uu = torch.stack([torch.sigmoid(frozen(x)) for _ in range(N_MC)], 0).var(0)
            w = (uu < float(tau)).float()
            frozen.eval()
        elif gate == "hybrid":
            c = torch.maximum(pb, 1 - pb)
            is_conf = (c >= float(theta_high))
            is_low = (c < float(theta_low))
            is_ambig = (~is_conf) & (~is_low)
            w = is_conf.float()
            if is_ambig.any():
                frozen.train()
                for m in frozen.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                with torch.no_grad():
                    uu_ambig = torch.stack([torch.sigmoid(frozen(x[is_ambig])) for _ in range(N_MC)], 0).var(0)
                frozen.eval()
                w[is_ambig] = (uu_ambig < float(tau)).float()
        else:
            raise ValueError(gate)
        kept[sl] = w.detach().cpu().numpy()
        if float(w.sum()) < 1:
            p1[sl] = p0[sl]
            i0 += b
            continue
        logit = model(x)
        p = torch.sigmoid(logit)
        pclip = p.clamp(1e-8, 1 - 1e-8)
        ent = -(pclip * torch.log(pclip) + (1 - pclip) * torch.log(1 - pclip))
        ent_loss = (w * ent).sum() / w.sum()
        loss = ent_loss

        # Anti-Attractor Class Diversity Regularization
        if lambda_div > 0 and float(w.sum()) > 1:
            mean_p = (w * p).sum() / w.sum()
            mean_p_clip = mean_p.clamp(1e-8, 1 - 1e-8)
            div_loss = -(mean_p_clip * torch.log(mean_p_clip) + (1 - mean_p_clip) * torch.log(1 - mean_p_clip))
            loss = loss - lambda_div * div_loss

        # Source Anchor Regularization to prevent drift / collapse
        if lambda_anchor > 0:
            anchor_loss = sum(((p_curr - p_0) ** 2).sum() for p_curr, p_0 in zip(params, init_params))
            loss = loss + lambda_anchor * anchor_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            p1[sl] = torch.sigmoid(model(x)).cpu().numpy()
        i0 += b
    return y, p0, p1, float(kept.mean())


# --- pixel corruptions (MedMNIST-C / Camelyon-C style) ---
def denorm(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean_t = x.new_tensor(mean)[:, None, None]
    std_t = x.new_tensor(std)[:, None, None]
    return (x * std_t + mean_t).clamp(0, 1)


def renorm(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean_t = x.new_tensor(mean)[:, None, None]
    std_t = x.new_tensor(std)[:, None, None]
    return (x - mean_t) / std_t


def corrupt_batch(x, name, severity, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """x: normalized BCHW. severity 1 or 3."""
    img = denorm(x, mean, std)
    if name == "identity":
        out = img
    elif name == "gaussian_noise":
        sigma = 0.08 if severity == 1 else 0.22
        out = (img + sigma * torch.randn_like(img)).clamp(0, 1)
    elif name == "defocus_blur":
        k = 5 if severity == 1 else 11
        import torch.nn.functional as F

        w = torch.ones(1, 1, k, k, device=x.device) / (k * k)
        w = w.repeat(3, 1, 1, 1)
        out = F.conv2d(img, w, padding=k // 2, groups=3).clamp(0, 1)
    elif name == "brightness":
        d = 0.1 if severity == 1 else 0.3
        out = (img + d).clamp(0, 1)
    elif name == "contrast":
        f = 0.8 if severity == 1 else 0.5
        mu = img.mean(dim=(2, 3), keepdim=True)
        out = ((img - mu) * f + mu).clamp(0, 1)
    elif name == "jpeg":
        levels = 32 if severity == 1 else 8
        out = (img * levels).round() / levels
    else:
        out = img
    return renorm(out, mean, std)
