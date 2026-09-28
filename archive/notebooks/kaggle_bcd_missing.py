"""Complete remaining B + D items. No retrain. Seed-42 BEST.pt only.

B  Feature-space shift: UMAP of frozen ResNet-18 embeddings
   (id_val hospitals 0/3/4 vs OOD val 1 vs target 2).

D1 MC-Dropout N sensitivity on hospital 1: one 50-pass run, then
   U from the first N in {5,10,20,30,50}.

D2 Random gate at UTTA's *test* coverage (0.562), so random is
   coverage-matched on hospital 2, not only on val 0.70.

C  Mandatory Tent/EATA is already done. SAR/DLTTA are cut.
   Tent-stability figure is plotted locally from the salvage JSON.

Kaggle: GPU T4, Internet ON, Add Input seed-42 BEST.pt.
Runtime ~40–70 min.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm

SEED = 42
TTA_LR, N_MC_MAX, E_MARGIN = 1e-5, 50, 0.4
# Locked from seed-42 Gate D: UTTA test coverage
UTTA_TEST_COV = 0.5620429374279869
BS, N_UMAP = 64, 2500
N_LIST = (5, 10, 20, 30, 50)
WORK = Path("/kaggle/working")
FIG = WORK / "figures"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def find_ckpt(seed: int = SEED) -> Path:
    hits = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        hits.extend(root.rglob(f"camelyon17_resnet18_source_s{seed}_BEST.pt"))
    if not hits:
        raise FileNotFoundError("Add Input camelyon17_resnet18_source_s42_BEST.pt")
    print("ckpt", hits[0])
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
        x = self.tf(img.convert("RGB"))
        y = torch.tensor(float(int(r["label"])))
        center = cid(r["center"])
        slide = r.get("slide", r.get("patient", -1))
        try:
            slide = int(slide)
        except Exception:
            slide = hash(str(slide)) % 10_000_000
        return x, y, center, slide


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

    def features(self, x):
        return self.backbone(x)


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


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m = Net().to(device)
    m.load_state_dict(sd)
    eval_mode(m)
    return m, st if isinstance(st, dict) else {}


def subset(ds, n, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return Subset(ds, idx.tolist())


@torch.no_grad()
def embed(model, loader, device, max_n=None):
    eval_mode(model)
    feats, ys, cs = [], [], []
    n = 0
    for batch in tqdm(loader, desc="embed"):
        x, y, center = batch[0].to(device), batch[1], batch[2]
        z = model.features(x).cpu().numpy()
        feats.append(z)
        ys.append(y.view(-1).numpy())
        cs.append(np.asarray(center).reshape(-1))
        n += x.size(0)
        if max_n and n >= max_n:
            break
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(cs)


def umap_or_tsne(X):
    X = X.astype(np.float32)
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="euclidean", random_state=SEED)
        print("UMAP")
        return reducer.fit_transform(X), "UMAP"
    except Exception as e:
        print("UMAP failed, sklearn TSNE:", e)
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(X), "t-SNE"


def plot_shift(xy, labels, title, path, cmap="tab10"):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    uniq = sorted(set(int(v) for v in labels))
    colors = plt.get_cmap(cmap)
    for i, u in enumerate(uniq):
        m = labels == u
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.55, color=colors(i / max(len(uniq), 1)), label=str(u))
    ax.legend(frameon=False, markerscale=3, fontsize=8, title="group")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


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
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    pred = (p >= 0.5).astype(int)
    return dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
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
        harm_rate=float(((b == y) & (a != y)).sum() / max((b == y).sum(), 1)),
        correction_rate=float(((b != y) & (a == y)).sum() / max((b != y).sum(), 1)),
        harmful=int(((b == y) & (a != y)).sum()),
        correction=int(((b != y) & (a == y)).sum()),
    )


@torch.no_grad()
def mc_stack(model, loader, device, n_passes, collect_y=True):
    """One pass collecting n_passes stochastic probabilities. Returns P [N, n], y."""
    mc_mode(model)
    ps, ys = [], []
    t0 = time.time()
    for x, y, *_ in tqdm(loader, desc=f"MC x{n_passes}"):
        x = x.to(device)
        batch = []
        for _ in range(n_passes):
            batch.append(torch.sigmoid(model(x)).view(-1).cpu())
        ps.append(torch.stack(batch, 0).numpy())  # [N, B]
        if collect_y:
            ys.append(y.view(-1).numpy())
    P = np.concatenate(ps, axis=1)  # [N, n]
    y = np.concatenate(ys) if collect_y else None
    dt = time.time() - t0
    return P, y, dt


def n_sensitivity(P, y):
    """P shape [Nmax, n]."""
    err_full = None
    rows = []
    U50 = P.var(axis=0, ddof=0)
    err = ((P.mean(axis=0) >= 0.5).astype(int) != y.astype(int)).astype(float)
    for n in N_LIST:
        Pn = P[:n]
        mean = Pn.mean(axis=0)
        U = Pn.var(axis=0, ddof=0)
        pred_err = ((mean >= 0.5).astype(int) != y.astype(int)).astype(float)
        corr = float(np.corrcoef(U, pred_err)[0, 1])
        # rank agreement with N=50
        from scipy.stats import spearmanr

        try:
            rho = float(spearmanr(U, U50).correlation)
        except Exception:
            rho = float(np.corrcoef(U, U50)[0, 1])
        order = np.argsort(U)
        bins = []
        for i, idx in enumerate(np.array_split(order, 10)):
            bins.append(dict(bin=i + 1, n=int(len(idx)), err_rate=float(pred_err[idx].mean()), mean_U=float(U[idx].mean())))
        rows.append(
            dict(
                N=n,
                u_error_corr=corr,
                spearman_vs_N50=rho,
                auroc=float(roc_auc_score(y, mean)),
                bins=bins,
            )
        )
        print(f"N={n:2d}  corr(U,err)={corr:.3f}  Spearman(U,U50)={rho:.3f}  AUROC={rows[-1]['auroc']:.4f}")
    return rows


def adapt_random(ckpt, loader, device, target_cov, rng_seed=SEED):
    src, _ = load_model(ckpt, device)
    m, _ = load_model(ckpt, device)
    opt = torch.optim.Adam(bn_affine(m), lr=TTA_LR)
    rng = np.random.RandomState(rng_seed)
    n_seen = n_kept = 0
    ys, pb, pa = [], [], []
    for x, y, *_ in tqdm(loader, desc=f"random cov={target_cov:.3f}"):
        x = x.to(device)
        B = x.size(0)
        n_seen += B
        with torch.no_grad():
            eval_mode(src)
            p0 = torch.sigmoid(src(x)).view(-1)
            pb.append(p0.cpu().numpy())
            w = torch.tensor(rng.rand(B) < target_cov, device=x.device, dtype=torch.float32)
        tta_mode(m)
        k = int(w.sum().item())
        if k > 0:
            opt.zero_grad(set_to_none=True)
            loss = (w * entropy(m(x))).sum() / w.sum()
            loss.backward()
            opt.step()
            n_kept += k
        with torch.no_grad():
            eval_mode(m)
            pa.append(torch.sigmoid(m(x)).view(-1).cpu().numpy())
        ys.append(y.view(-1).numpy())
    y = np.concatenate(ys)
    b = np.concatenate(pb)
    a = np.concatenate(pa)
    return dict(
        target_cov=target_cov,
        coverage=n_kept / max(n_seen, 1),
        before=metrics_of(y, b),
        after=metrics_of(y, a),
        harm=flip_table(y, b, a),
    )


def main():
    assert torch.cuda.is_available(), "Enable GPU T4"
    device = torch.device("cuda")
    print("GPU", torch.cuda.get_device_name(0))
    WORK.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    eval_tf = transforms.Compose(
        [transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    raw = load_dataset("wltjr1007/Camelyon17-WILDS")
    id_set = HFCamelyon(raw["validation"], {0, 3, 4}, eval_tf)
    ood_set = HFCamelyon(raw["validation"], {1}, eval_tf)
    test_set = HFCamelyon(raw["test"], {2}, eval_tf)
    id_loader = DataLoader(subset(id_set, N_UMAP), batch_size=BS, shuffle=False, num_workers=0)
    ood_s_loader = DataLoader(subset(ood_set, N_UMAP), batch_size=BS, shuffle=False, num_workers=0)
    test_s_loader = DataLoader(subset(test_set, N_UMAP), batch_size=BS, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_set, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

    ckpt = find_ckpt()
    model, st = load_model(ckpt, device)
    print("epoch", st.get("epoch"), "val_auroc", st.get("val_auroc"))

    # ----- B: embeddings -----
    print("\n=== B  feature shift (UMAP/t-SNE) ===")
    f_id, y_id, c_id = embed(model, id_loader, device)
    f_ood, y_ood, c_ood = embed(model, ood_s_loader, device)
    f_te, y_te, c_te = embed(model, test_s_loader, device)
    X = np.concatenate([f_id, f_ood, f_te])
    split = np.array(["id_val"] * len(f_id) + ["ood_val"] * len(f_ood) + ["test"] * len(f_te))
    hospital = np.concatenate([c_id, c_ood, c_te])
    xy, method = umap_or_tsne(X)
    plot_shift(xy, split, f"{method} by split (frozen ResNet-18)", FIG / "fig_b_umap_split.png")
    plot_shift(xy, hospital, f"{method} by hospital", FIG / "fig_b_umap_hospital.png")
    np.savez_compressed(WORK / "embeddings_s42.npz", xy=xy, split=split, hospital=hospital, y=np.concatenate([y_id, y_ood, y_te]))

    # ----- D1: MC N sensitivity (shared 50 passes) -----
    print("\n=== D1  MC-Dropout N sensitivity (hospital 1) ===")
    P, y_ood_full, dt = mc_stack(model, ood_loader, device, N_MC_MAX)
    print(f"50 MC passes on val took {dt/60:.1f} min; P={P.shape}")
    try:
        from scipy.stats import spearmanr  # noqa: F401
    except Exception:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "scipy"])
    n_rows = n_sensitivity(P, y_ood_full)
    # plot corr vs N
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([r["N"] for r in n_rows], [r["u_error_corr"] for r in n_rows], "o-", color="#1f4e79", label="corr(U, error)")
    ax.plot([r["N"] for r in n_rows], [r["spearman_vs_N50"] for r in n_rows], "s--", color="#e67e22", label="Spearman vs N=50")
    ax.set_xlabel("MC passes N")
    ax.set_ylabel("correlation")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("MC-Dropout N sensitivity (hospital 1, seed 42)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_mc_n_sensitivity.png", dpi=170)
    plt.close(fig)

    # ----- D2: random at UTTA test coverage -----
    print("\n=== D2  random gate at UTTA test coverage", UTTA_TEST_COV, "===")
    rand = adapt_random(ckpt, test_loader, device, UTTA_TEST_COV, rng_seed=SEED)
    print(
        "random@0.562  AUROC={:.4f} F1={:.4f} ECE={:.4f} cov={:.3f} harm={:.4f}".format(
            rand["after"]["auroc"],
            rand["after"]["f1"],
            rand["after"]["ece"],
            rand["coverage"],
            rand["harm"]["harm_rate"],
        )
    )
    print("compare: seed-42 UTTA test AUROC=0.9507 F1=0.8697 harm=0.0032 cov=0.562")
    print("compare: seed-42 random@0.70 AUROC=0.9223 F1=0.8051 harm=0.0617")

    out = {
        "seed": SEED,
        "checkpoint": str(ckpt),
        "epoch": st.get("epoch"),
        "umap_method": method,
        "n_umap_per_split": N_UMAP,
        "mc_n_sensitivity": n_rows,
        "mc_50_minutes": dt / 60.0,
        "random_matched_test_coverage": rand,
        "utta_test_coverage_target": UTTA_TEST_COV,
        "note": "C mandatory (Tent/EATA) already locked. SAR/DLTTA not run (cut-first).",
    }
    p = WORK / "camelyon17_bcd_missing_s42.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)
    print("Download figures/ and the JSON.")


if __name__ == "__main__":
    main()
