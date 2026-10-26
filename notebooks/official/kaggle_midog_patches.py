#!/usr/bin/env python3
"""X4 — MIDOG 2021 mitosis vs impostor patches, leave-one-scanner-out TTA.

Not RetinaNet / MIDOG++ 7-tumor detection. Classification proxy for scanner shift.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
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


ensure_libraries(["torch", "torchvision", "numpy", "PIL", "sklearn", "scipy", "tqdm"])

import numpy as np
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset  # type: ignore
from torchvision import transforms  # type: ignore
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tta_kit import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
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
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else str(HERE.parent.parent / "results" / "optional" / "midog")))
OUT.mkdir(parents=True, exist_ok=True)
PATCH = 64
HOLD_SCANNER = None  # set to a name, else auto = last scanner by count


class PatchDS(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        if len(item) == 2:
            crop, y = item
            if isinstance(crop, (Path, str)):
                try:
                    crop = Image.open(crop).convert("RGB")
                except Exception:
                    crop = Image.new("RGB", (PATCH, PATCH), (128, 128, 128))
        else:
            path, y, box = item
            try:
                raw = Image.open(path).convert("RGB")
                if box[2] > box[0]:
                    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                else:
                    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
                a = PATCH // 2
                l, t = int(cx - a), int(cy - a)
                crop = raw.crop((l, t, l + PATCH, t + PATCH))
            except Exception:
                crop = Image.new("RGB", (PATCH, PATCH), (128, 128, 128))

        if crop.size != (PATCH, PATCH):
            crop = crop.resize((PATCH, PATCH))
        return self.transform(crop), int(y)


ZENODO_MIDOG_BASE = "https://zenodo.org/records/4643381/files"


def download_zenodo_file(fname: str, dest_dir: Path, idx: int = 1, total: int = 1) -> Path:
    dest_path = dest_dir / fname
    if dest_path.exists() and dest_path.stat().st_size > 10000:
        return dest_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{ZENODO_MIDOG_BASE}/{fname}?download=1"
    print(f"[{idx}/{total}] Downloading {fname} directly from Zenodo ({url})...", flush=True)

    import subprocess
    import time
    import urllib.request
    curl_bin = "curl.exe" if sys.platform == "win32" else "curl"

    for attempt in range(1, 4):
        # 1. Try curl with auto-resume/retry
        try:
            cmd = [curl_bin, "-sL", "--connect-timeout", "30", "--max-time", "300", "--retry", "3", url, "-o", str(dest_path)]
            res = subprocess.run(cmd, timeout=360)
            if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 10000:
                print(f"  Successfully downloaded {fname} ({dest_path.stat().st_size} bytes)", flush=True)
                return dest_path
        except Exception:
            pass

        # 2. Fallback to urllib with chunked streaming
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest_path.exists() and dest_path.stat().st_size > 10000:
                print(f"  Successfully downloaded {fname} ({dest_path.stat().st_size} bytes)", flush=True)
                return dest_path
        except Exception as e:
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except Exception:
                    pass
            if attempt < 3:
                print(f"  Attempt {attempt} failed ({e}). Retrying in 5s...", flush=True)
                time.sleep(5)
            else:
                raise e
    return dest_path


def find_or_download_midog(zdir: Path):
    search_dirs = []
    if os.environ.get("MIDOG_DIR"):
        search_dirs.append(Path(os.environ["MIDOG_DIR"]))
    if Path("/kaggle/input").exists():
        search_dirs.append(Path("/kaggle/input"))
    if Path("/kaggle/working").exists():
        search_dirs.append(Path("/kaggle/working"))
    search_dirs.append(HERE.parent.parent)

    roots = []
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.name.lower() in ("midog.json", "midog_coco.json"):
                roots.append(p)
                break
        if roots:
            break

    if roots:
        js = roots[0]
        print("MIDOG.json found locally at:", js)
        return js.parent, json.loads(js.read_text(encoding="utf-8"))

    # Direct import from Zenodo
    print("MIDOG.json not found locally. Importing directly from Zenodo (record 4643381)...")
    js_path = download_zenodo_file("MIDOG.json", zdir)
    return js_path.parent, json.loads(js_path.read_text(encoding="utf-8"))


def scanner_of(fname: str) -> str:
    u = fname.upper()
    for s in ("HAMAMATSU-XR", "HAMAMATSU-S360", "HAMAMATSU", "APERIO", "LEICA", "PHILIPS", "3DHISTECH", "VENTANA"):
        if s in u:
            return s
    # Challenge case ranges: 1-50: Hamamatsu XR, 51-100: Hamamatsu S360, 101-150: Aperio CS2, 151-200: Leica GT450
    m = re.search(r"(\d+)", Path(fname).stem)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 50:
            return "HAMAMATSU-XR"
        elif 51 <= num <= 100:
            return "HAMAMATSU-S360"
        elif 101 <= num <= 150:
            return "APERIO-CS2"
        elif 151 <= num <= 200:
            return "LEICA-GT450"
    return "UNK"


def main():
    seed_all(SEED)
    print("device", DEVICE)
    zdir = OUT / "midog_zenodo"
    zdir.mkdir(exist_ok=True, parents=True)
    root, coco = find_or_download_midog(zdir)
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    cat_mit = None
    for c in coco.get("categories", []):
        n = c["name"].lower()
        if "mitot" in n and "not" not in n and "non" not in n and "impost" not in n and "hard" not in n:
            cat_mit = c["id"]
    if cat_mit is None:
        cat_mit = 1
    print("categories", coco.get("categories"), "mit_id", cat_mit)

    # 1. Fast local discovery of any mounted/downloaded TIFF files
    local_tiff_cache = {}
    scan_search_dirs = [root]
    if Path("/kaggle/input").exists():
        scan_search_dirs.append(Path("/kaggle/input"))
    if Path("/kaggle/working").exists():
        scan_search_dirs.append(Path("/kaggle/working"))
    scan_search_dirs.append(HERE.parent.parent)

    print("[*] Indexing local TIFF files...")
    for sd in scan_search_dirs:
        if sd.exists():
            for p in sd.rglob("*.tif*"):
                if p.is_file() and p.stat().st_size > 1000:
                    local_tiff_cache[p.name.lower()] = p
    print(f"[*] Found {len(local_tiff_cache)} local TIFF files.")

    # 2. Scanner cap logic: default to 3 images per scanner when downloading from Zenodo
    # to complete in ~10 min and prevent Kaggle 12-hour timeout.
    env_max = os.environ.get("MAX_IMAGES_PER_SCANNER")
    if env_max is not None:
        max_per_scanner = int(env_max)
    else:
        max_per_scanner = 0 if len(local_tiff_cache) >= 10 else 3
    print(f"[*] max_images_per_scanner: {max_per_scanner} (0 = all available)")

    download_missing = os.environ.get("AUTO_DOWNLOAD_ZENODO", "1") == "1"

    by_scan_images = defaultdict(set)
    counts_by_fn = defaultdict(lambda: {"mit": 0, "non_mit": 0})
    for ann in coco["annotations"]:
        fn = id2file[ann["image_id"]]
        by_scan_images[scanner_of(fn)].add(fn)
        if ann["category_id"] == cat_mit:
            counts_by_fn[fn]["mit"] += 1
        else:
            counts_by_fn[fn]["non_mit"] += 1

    # 3. Collect targets and acquire files (prioritize slides with both classes)
    files_to_obtain = []
    for scan_name, fnames in by_scan_images.items():
        def balance_score(fn):
            m = counts_by_fn[fn]["mit"]
            nm = counts_by_fn[fn]["non_mit"]
            has_both = 1 if (m > 0 and nm > 0) else 0
            return (has_both, min(m, nm), m + nm)

        sorted_fns = sorted(fnames, key=balance_score, reverse=True)
        if max_per_scanner > 0:
            sorted_fns = sorted_fns[:max_per_scanner]
        for fn in sorted_fns:
            files_to_obtain.append((scan_name, fn))

    file_to_path = {}
    total_needed = len(files_to_obtain)
    for idx, (scan_name, fn) in enumerate(files_to_obtain, 1):
        path = local_tiff_cache.get(Path(fn).name.lower())
        if path is None:
            cand = root / fn
            if cand.is_file() and cand.stat().st_size > 1000:
                path = cand
        if path is None and download_missing:
            try:
                path = download_zenodo_file(fn, zdir, idx=idx, total=total_needed)
            except Exception as e:
                print(f"Failed to download {fn} from Zenodo: {e}", flush=True)

        if path is not None and path.is_file():
            file_to_path[fn] = path

    print(f"[*] Successfully acquired {len(file_to_path)} slides across scanners.")

    # 4. Extract 64x64 patches once in memory (eliminates redundant multi-gigabyte TIFF I/O)
    print(f"[*] Extracting {PATCH}x{PATCH} patches from acquired slides into memory...")
    anns_by_fn = defaultdict(list)
    for ann in coco["annotations"]:
        fn = id2file.get(ann["image_id"])
        if fn in file_to_path:
            anns_by_fn[fn].append(ann)

    by_scan = defaultdict(list)
    for fn, path in tqdm(file_to_path.items(), desc="Extracting patches"):
        anns = anns_by_fn.get(fn, [])
        if not anns:
            continue
        try:
            slide = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open {path}: {e}")
            continue

        scan_name = scanner_of(fn)
        for ann in anns:
            box = ann["bbox"]
            y = 1 if ann["category_id"] == cat_mit else 0
            if box[2] > box[0]:
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            else:
                cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
            a = PATCH // 2
            l, t = int(cx - a), int(cy - a)
            crop = slide.crop((l, t, l + PATCH, t + PATCH))
            if crop.size != (PATCH, PATCH):
                crop = crop.resize((PATCH, PATCH))
            by_scan[scan_name].append((crop.copy(), y))
        del slide

    print("scanners", {k: len(v) for k, v in by_scan.items()})
    scans = sorted(by_scan, key=lambda s: -len(by_scan[s]))
    scans = [s for s in scans if s != "UNK" or len(scans) == 1]
    if len(scans) < 2:
        raise SystemExit(f"need ≥2 scanners, got {list(by_scan)}")
    hold = HOLD_SCANNER if (HOLD_SCANNER and HOLD_SCANNER in scans) else scans[-1]
    train_items, test_items = [], []
    for s, items in by_scan.items():
        (test_items if s == hold else train_items).extend(items)
    print("hold-out scanner", hold, "n_train", len(train_items), "n_test", len(test_items))
    rng = np.random.RandomState(SEED)
    rng.shuffle(train_items)
    n_val = max(10, min(250, len(train_items) // 10))
    val_items, tr_items = train_items[:n_val], train_items[n_val:]

    tf_tr = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tf_ev = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    nw = 2 if torch.cuda.is_available() and sys.platform != "win32" else 0
    tr = DataLoader(PatchDS(tr_items, tf_tr), batch_size=32, shuffle=True, num_workers=nw)
    va = DataLoader(PatchDS(val_items, tf_ev), batch_size=32, shuffle=False, num_workers=nw)
    te = DataLoader(PatchDS(test_items, tf_ev), batch_size=32, shuffle=False, num_workers=nw)

    model = ResNetMC("resnet18", 1).to(DEVICE)
    model._arch = "resnet18"
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    best, st = -1.0, None
    for ep in range(1, 8):
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
        print(f"epoch {ep} loss={np.mean(losses):.4f} val_auroc={met['auroc']:.4f}")
        if met["auroc"] > best or st is None:
            best = met["auroc"]
            st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(st, OUT / "midog_resnet18_s42_BEST.pt")
    model.load_state_dict(st)
    model.to(DEVICE)
    yt, pt = predict_bin(model, te, DEVICE)
    src = metrics_bin(yt, pt)
    print("SOURCE hold-scanner", hold, src)

    yv, pbar, u = mc_bin(model, va, DEVICE)
    tau = float(np.percentile(u, TAU_PCT))
    conf = np.maximum(pbar, 1 - pbar)
    ent = entropy_bin(pbar)
    cov = float((u < tau).mean())
    conf_t = float(np.quantile(conf, 1.0 - cov))
    ent_t = float(np.quantile(ent, cov))

    def clone():
        m = ResNetMC("resnet18", 1).to(DEVICE)
        m._arch = "resnet18"
        m.load_state_dict(st)
        return m

    rows = {"source": src}
    for gate, kw in [
        ("tent", {}),
        ("random", {"p_keep": cov}),
        ("confidence", {"conf_t": conf_t}),
        ("utta", {"tau": tau}),
    ]:
        m = clone()
        y, a, b, c = tta_bin(m, te, DEVICE, gate, **kw)
        met = metrics_bin(y, b)
        h = harm_bin(y, a, b)
        rows[gate] = {**met, "coverage": c, "harm": h}
        print(gate, met["auroc"], met["f1"], "harm", h["harm_rate"])

    blob = {
        "dataset": "MIDOG2021 mitosis vs impostor patches",
        "holdout_scanner": hold,
        "n_train_pool_total": len(train_items),
        "n_val_heldout": len(val_items),
        "n_train": len(tr_items),
        "n_test": len(test_items),
        "tau": tau,
        "results": rows,
        "note": "Scanner-shift classification proxy. Not MIDOG++ detection F1.",
    }
    (OUT / "midog_patches_lodo_s42.json").write_text(json.dumps(blob, indent=2))
    print("wrote", OUT / "midog_patches_lodo_s42.json")


if __name__ == "__main__":
    main()
