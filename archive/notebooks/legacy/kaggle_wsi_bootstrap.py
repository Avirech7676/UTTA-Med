"""UTTA-Med Gate 10 — WSI-level bootstrap + reliability diagrams.

No retrain. Frozen protocol from the 5-seed matrix:
  BN stats frozen, TTA lr=1e-5, N_MC=20, tau=70th percentile of val U.

Kaggle: GPU T4, Internet ON.
Add Input: dataset containing camelyon17_resnet18_source_s42_BEST.pt
Optional: add the other BEST.pt files and set SEEDS = [42, 123, 2024, 7, 99].
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
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

SEEDS = [42]
TTA_LR, N_MC, E_MARGIN = 1e-5, 20, 0.4
BS, N_BOOT = 64, 1000
WORK = Path("/kaggle/working")
FIG = WORK / "figures"
DUMP = WORK / "dumps"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
METHODS = ["source", "tent", "eata", "random", "confidence", "entropy", "utta"]


def find_ckpt(seed: int) -> Path:
    hits = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        hits.extend(root.rglob(f"camelyon17_resnet18_source_s{seed}_BEST.pt"))
        hits.extend(root.rglob(f"camelyon17_resnet18_source_s{seed}.pt"))
    if not hits:
        raise FileNotFoundError(
            f"No checkpoint for seed {seed}. Add Input the dataset with "
            f"camelyon17_resnet18_source_s{seed}_BEST.pt"
        )
    hits = sorted(hits, key=lambda p: (0 if "BEST" in p.name else 1, str(p)))
    print("ckpt seed", seed, "->", hits[0])
    return hits[0]


def cid(v) -> int:
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
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        x = self.tf(img)
        y = torch.tensor(float(int(r["label"])))
        slide = r.get("slide", r.get("patient", -1))
        try:
            slide = int(slide)
        except Exception:
            slide = hash(str(slide)) % 10_000_000
        return x, y, slide


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
    out = dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        sensitivity=float(recall_score(y, pred, zero_division=0)),
        specificity=float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1)),
        f1=float(f1_score(y, pred, zero_division=0)),
        ece=ece_score(y, p),
        brier=float(np.mean((p - y) ** 2)),
    )
    try:
        out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        out["auprc"] = float("nan")
    return out


def flip_table(y, b, a):
    y = np.asarray(y).ravel().astype(int)
    b = (np.asarray(b).ravel() >= 0.5).astype(int)
    a = (np.asarray(a).ravel() >= 0.5).astype(int)
    n_wrong = int((b != y).sum())
    n_ok = int((b == y).sum())
    return dict(
        correction_rate=float(((b != y) & (a == y)).sum() / max(n_wrong, 1)),
        harm_rate=float(((b == y) & (a != y)).sum() / max(n_ok, 1)),
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
    """Official bn_freeze_stats: update γ/β, freeze running mean/var, dropout off."""
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


def load_model(ckpt_path, device):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m = Net().to(device)
    m.load_state_dict(sd)
    eval_mode(m)
    return m, st if isinstance(st, dict) else {}


def bootstrap_ci_by_slide(y, p, slides, metric_fn, n_boot=N_BOOT, alpha=0.05, seed=0):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    slides = np.asarray(slides).ravel()
    keys = np.unique(slides)
    groups = {int(s): np.where(slides == s)[0] for s in keys}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[int(s)] for s in sampled])
        try:
            stats[i] = metric_fn(y[idx], p[idx])
        except Exception:
            stats[i] = np.nan
    stats = stats[np.isfinite(stats)]
    lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
    return dict(
        mean=float(np.mean(stats)),
        std=float(np.std(stats, ddof=1) if len(stats) > 1 else 0.0),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=int(len(stats)),
        n_slides=int(len(keys)),
    )


def bootstrap_harm_by_slide(y, pb, pa, slides, n_boot=N_BOOT, seed=0):
    y = np.asarray(y).ravel().astype(int)
    pb = np.asarray(pb).ravel()
    pa = np.asarray(pa).ravel()
    slides = np.asarray(slides).ravel()
    keys = np.unique(slides)
    groups = {int(s): np.where(slides == s)[0] for s in keys}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[int(s)] for s in sampled])
        stats[i] = flip_table(y[idx], pb[idx], pa[idx])["harm_rate"]
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return dict(
        mean=float(np.mean(stats)),
        std=float(np.std(stats, ddof=1)),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
        n_slides=int(len(keys)),
    )


def per_slide_table(y, p, slides):
    y, p, slides = np.asarray(y).ravel(), np.asarray(p).ravel(), np.asarray(slides).ravel()
    rows = []
    for s in np.unique(slides):
        m = slides == s
        pred = (p[m] >= 0.5).astype(int)
        row = {"slide": int(s), "n": int(m.sum()), "pos": int(y[m].sum())}
        try:
            row["auroc"] = (
                float(roc_auc_score(y[m], p[m])) if len(np.unique(y[m])) > 1 else float("nan")
            )
        except Exception:
            row["auroc"] = float("nan")
        row["f1"] = float(f1_score(y[m], pred, zero_division=0))
        row["acc"] = float(accuracy_score(y[m], pred))
        rows.append(row)
    return rows


def reliability_diagram(y, p, title, path, n_bins=15):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, ns = [], [], []
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1]) if i else (p >= bins[i]) & (p <= bins[i + 1])
        ns.append(int(m.sum()))
        if m.any():
            accs.append(float(y[m].mean()))
            confs.append(float(p[m].mean()))
        else:
            accs.append(np.nan)
            confs.append(np.nan)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(confs, accs, "o-", color="#1f4e79", lw=2, markersize=5, label="model")
    ax.bar(
        [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)],
        [n / max(sum(ns), 1) for n in ns],
        width=1 / n_bins,
        alpha=0.25,
        color="#7f8c8d",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def choose_tau(model, ood_loader, device):
    Us, Cs, Hs = [], [], []
    for x, y, _ in tqdm(ood_loader, desc="MC val (tau)"):
        x = x.to(device)
        mean, var = mc_predict(model, x)
        p = mean.clamp(1e-6, 1 - 1e-6)
        Us.append(var.cpu().numpy())
        Cs.append(torch.maximum(p, 1 - p).cpu().numpy())
        Hs.append((-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).cpu().numpy())
    U, C, H = map(np.concatenate, (Us, Cs, Hs))
    tau = float(np.quantile(U, 0.70))
    cov = float((U < tau).mean())
    conf_t = float(np.quantile(C, 1.0 - cov))
    ent_t = float(np.quantile(H, cov))
    return dict(tau=tau, target_cov=cov, conf_t=conf_t, ent_t=ent_t)


def collect_source(model, loader, device):
    eval_mode(model)
    ys, ps, sls = [], [], []
    for x, y, sl in tqdm(loader, desc="source eval"):
        x = x.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).view(-1).cpu().numpy()
        ys.append(y.view(-1).numpy())
        ps.append(p)
        sls.append(np.asarray(sl).reshape(-1))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(sls)


def adapt_dump(ckpt_path, loader, gate, device, tau=None, target_cov=None, rng_seed=42):
    src, _ = load_model(ckpt_path, device)
    m, _ = load_model(ckpt_path, device)
    opt = torch.optim.Adam(bn_affine(m), lr=TTA_LR)
    rng = np.random.RandomState(rng_seed)
    n_seen = n_kept = n_skip = 0
    ys, pb, pa, sls = [], [], [], []
    for x, y, sl in tqdm(loader, desc=str(gate)):
        x = x.to(device)
        B = x.size(0)
        n_seen += B
        with torch.no_grad():
            eval_mode(src)
            p0 = torch.sigmoid(src(x)).view(-1)
            pb.append(p0.cpu().numpy())
            if gate == "tent":
                w = torch.ones(B, device=x.device)
            elif gate == "eata":
                w = (entropy(src(x)) < E_MARGIN).float()
            elif gate == "random":
                w = torch.tensor(rng.rand(B) < target_cov, device=x.device, dtype=torch.float32)
            elif gate == "confidence":
                w = (torch.maximum(p0, 1 - p0) > tau).float()
            elif gate == "entropy":
                w = (entropy(torch.logit(p0.clamp(1e-6, 1 - 1e-6))) < tau).float()
            elif gate == "utta":
                _, var = mc_predict(src, x)
                w = (var < tau).float()
            else:
                raise ValueError(gate)
        tta_mode(m)
        k = int(w.sum().item())
        if k == 0:
            n_skip += 1
        else:
            opt.zero_grad(set_to_none=True)
            loss = (w * entropy(m(x))).sum() / w.sum()
            loss.backward()
            opt.step()
            n_kept += k
        with torch.no_grad():
            eval_mode(m)
            pa.append(torch.sigmoid(m(x)).view(-1).cpu().numpy())
        ys.append(y.view(-1).numpy())
        sls.append(np.asarray(sl).reshape(-1))
    return dict(
        y=np.concatenate(ys),
        p_before=np.concatenate(pb),
        p_after=np.concatenate(pa),
        slide=np.concatenate(sls),
        coverage=n_kept / max(n_seen, 1),
        n_skip_batch=n_skip,
    )


def summarize_method(name, y, p, slides, pb=None):
    m = metrics_of(y, p)
    ci_auroc = bootstrap_ci_by_slide(y, p, slides, lambda yt, yp: roc_auc_score(yt, yp))
    ci_f1 = bootstrap_ci_by_slide(
        y, p, slides, lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    )
    ci_ece = bootstrap_ci_by_slide(y, p, slides, lambda yt, yp: ece_score(yt, yp))
    out = dict(
        method=name,
        point=m,
        slides=per_slide_table(y, p, slides),
        ci_auroc=ci_auroc,
        ci_f1=ci_f1,
        ci_ece=ci_ece,
    )
    if pb is not None:
        out["harm"] = flip_table(y, pb, p)
        out["ci_harm"] = bootstrap_harm_by_slide(y, pb, p, slides)
    return out


def overlay_reliability(dumps, seed, path):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    colors = dict(
        source="#4a4a4a",
        tent="#c0392b",
        eata="#e67e22",
        random="#7f8c8d",
        confidence="#1f77b4",
        entropy="#17becf",
        utta="#2ca02c",
    )
    for name in ["source", "tent", "eata", "confidence", "utta"]:
        d = dumps[name]
        y, p = d["y"], d["p_after"]
        bins = np.linspace(0, 1, 16)
        accs, confs = [], []
        for i in range(15):
            msk = (p > bins[i]) & (p <= bins[i + 1]) if i else (p >= bins[i]) & (p <= bins[i + 1])
            if msk.any():
                accs.append(y[msk].mean())
                confs.append(p[msk].mean())
        ax.plot(confs, accs, "o-", color=colors[name], lw=1.8, markersize=4, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability — seed {seed} (hospital 2)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


def run_one_seed(seed, ood_loader, test_loader, device):
    print("\n========== SEED", seed, "==========")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ckpt = find_ckpt(seed)
    model, st = load_model(ckpt, device)
    print("epoch", st.get("epoch"), "claimed val_auroc", st.get("val_auroc"))

    tau_pack = choose_tau(model, ood_loader, device)
    tau, cov, conf_t, ent_t = (
        tau_pack["tau"],
        tau_pack["target_cov"],
        tau_pack["conf_t"],
        tau_pack["ent_t"],
    )
    print(f"FROZEN tau=70th pct U={tau:.6g} val_cov={cov:.3f} conf_t={conf_t:.4f} ent_t={ent_t:.4f}")

    y_s, p_s, sl_s = collect_source(model, test_loader, device)
    print(
        "test slides",
        sorted(int(x) for x in np.unique(sl_s)),
        "n",
        len(y_s),
        "source AUROC",
        round(metrics_of(y_s, p_s)["auroc"], 4),
    )

    dumps = {
        "source": dict(y=y_s, p_after=p_s, p_before=p_s, slide=sl_s, coverage=0.0, n_skip_batch=0)
    }
    dumps["tent"] = adapt_dump(ckpt, test_loader, "tent", device, rng_seed=seed)
    dumps["eata"] = adapt_dump(ckpt, test_loader, "eata", device, rng_seed=seed)
    dumps["random"] = adapt_dump(
        ckpt, test_loader, "random", device, target_cov=cov, rng_seed=seed
    )
    dumps["confidence"] = adapt_dump(
        ckpt, test_loader, "confidence", device, tau=conf_t, rng_seed=seed
    )
    dumps["entropy"] = adapt_dump(ckpt, test_loader, "entropy", device, tau=ent_t, rng_seed=seed)
    dumps["utta"] = adapt_dump(ckpt, test_loader, "utta", device, tau=tau, rng_seed=seed)

    seed_out = {
        "seed": seed,
        "checkpoint": str(ckpt),
        "epoch": st.get("epoch"),
        "claimed_val_auroc": st.get("val_auroc"),
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": TTA_LR,
            "n_mc": N_MC,
            "tau_rule": "70th percentile of val U",
            "e_margin": E_MARGIN,
            "bootstrap": "WSI/slide with replacement",
            "n_boot": N_BOOT,
        },
        "tau": tau,
        "matched_coverage_val": cov,
        "conf_threshold": conf_t,
        "entropy_threshold": ent_t,
        "n_test_slides": int(len(np.unique(sl_s))),
        "methods": {},
    }

    print(f"\n=== SEED {seed}  point | WSI 95% CI (AUROC) ===")
    print(f"{'method':12s} {'AUROC':>8s} {'CI':>22s} {'F1':>8s} {'harm':>8s} {'cov':>6s}")
    for name in METHODS:
        d = dumps[name]
        y, p, sl = d["y"], d["p_after"], d["slide"]
        pb = None if name == "source" else d["p_before"]
        summ = summarize_method(name, y, p, sl, pb)
        seed_out["methods"][name] = {
            "point": summ["point"],
            "coverage": None if name == "source" else float(d["coverage"]),
            "harm": summ.get("harm"),
            "ci_auroc": summ["ci_auroc"],
            "ci_f1": summ["ci_f1"],
            "ci_ece": summ["ci_ece"],
            "ci_harm": summ.get("ci_harm"),
            "per_slide": summ["slides"],
        }
        ci = summ["ci_auroc"]
        harm = "—" if name == "source" else f"{summ['harm']['harm_rate']:.3f}"
        covs = "—" if name == "source" else f"{d['coverage']:.3f}"
        print(
            f"{name:12s} {summ['point']['auroc']:.4f}  [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]  "
            f"{summ['point']['f1']:.4f}  {harm:>8s}  {covs:>6s}"
        )
        np.savez_compressed(
            DUMP / f"s{seed}_{name}.npz",
            y=y,
            p=p,
            p_before=d["p_before"],
            slide=sl,
        )
        reliability_diagram(
            y, p, title=f"seed {seed}  {name}", path=FIG / f"reliability_s{seed}_{name}.png"
        )

    overlay_reliability(dumps, seed, FIG / f"reliability_overlay_s{seed}.png")

    slides = np.unique(sl_s)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = np.arange(len(slides))
    src_s = [
        next(r["auroc"] for r in seed_out["methods"]["source"]["per_slide"] if r["slide"] == int(s))
        for s in slides
    ]
    utt_s = [
        next(r["auroc"] for r in seed_out["methods"]["utta"]["per_slide"] if r["slide"] == int(s))
        for s in slides
    ]
    ax.bar(xs - 0.18, src_s, 0.36, label="source", color="#4a4a4a")
    ax.bar(xs + 0.18, utt_s, 0.36, label="UTTA-Med", color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(s)) for s in slides], fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("test WSI id")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    ax.set_title(f"Per-slide AUROC — seed {seed} (n_slides={len(slides)})")
    fig.tight_layout()
    fig.savefig(FIG / f"per_slide_auroc_s{seed}.png", dpi=170)
    plt.close(fig)

    jp = WORK / f"camelyon17_wsi_bootstrap_s{seed}.json"
    jp.write_text(json.dumps(seed_out, indent=2))
    print("wrote", jp)
    return seed_out


def main():
    assert torch.cuda.is_available(), "Enable GPU T4"
    device = torch.device("cuda")
    print("GPU", torch.cuda.get_device_name(0))
    WORK.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    DUMP.mkdir(exist_ok=True)

    eval_tf = transforms.Compose(
        [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    print("loading HuggingFace Camelyon17-WILDS")
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    ood_set = HFCamelyon(raw["validation"], {1}, eval_tf)
    test_set = HFCamelyon(raw["test"], {2}, eval_tf)
    ood_loader = DataLoader(ood_set, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

    all_seed_out = []
    for seed in SEEDS:
        all_seed_out.append(run_one_seed(seed, ood_loader, test_loader, device))

    summary = {
        "n_boot": N_BOOT,
        "protocol": "WSI-level bootstrap, hospital-2 slides with replacement",
        "seeds": all_seed_out,
    }
    (WORK / "camelyon17_wsi_bootstrap_all.json").write_text(json.dumps(summary, indent=2))
    print("\nDONE. Download:")
    print("  /kaggle/working/camelyon17_wsi_bootstrap_s*.json")
    print("  /kaggle/working/figures/*.png")
    print("  /kaggle/working/dumps/*.npz")
    print("CIs are wide because hospital 2 has only 10 WSIs. That is the honest unit of inference.")


if __name__ == "__main__":
    main()
