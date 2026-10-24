"""Official architecture freeze: 1-logit, dropout 0.5, Adam, BN-affine count."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
import torch.optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_resnet18_one_logit_dropout_half():
    from src.models.resnet18 import OFFICIAL_DROPOUT_P, build_resnet18

    m = build_resnet18(pretrained=False)
    x = torch.randn(2, 3, 96, 96)
    y = m(x)
    assert y.shape == (2, 1)
    assert m.classifier[-1].weight.shape == (1, 512)
    assert m.dropout_p == 0.5 == OFFICIAL_DROPOUT_P
    assert m.num_classes == 1
    drops = [mod.p for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]
    assert drops == [0.5]


def test_resnet18_rejects_two_class():
    from src.models.resnet18 import ResNet18Classifier
    import pytest

    with pytest.raises(ValueError, match="1-logit"):
        ResNet18Classifier(pretrained=False, num_classes=2)


import pytest

@pytest.mark.skip(reason="ResNet-50 replacement architecture archived to archive/unrequested-research-extensions per charter")
def test_resnet50_uttamed_import_and_shape():
    from src.models import ResNet50UTTAMed, build_resnet50
    from src.models.resnet50 import ResNet50UTTAMed as Alias

    assert ResNet50UTTAMed is Alias
    m = build_resnet50(pretrained=False)
    assert isinstance(m, ResNet50UTTAMed)
    y = m(torch.randn(2, 3, 96, 96))
    assert y.shape == (2, 1)
    assert m.dropout_p == 0.5


def test_source_optimizer_is_adam_not_adamw():
    from src.training import train_loop

    src = inspect.getsource(train_loop.train_source)
    assert "torch.optim.Adam(" in src
    assert "AdamW" not in src


def test_tta_optimizer_is_adam():
    from src.tta import tent, uncertainty_gated, eata

    assert "torch.optim.Adam(" in inspect.getsource(tent.Tent.__init__)
    assert "AdamW" not in inspect.getsource(tent.Tent.__init__)
    assert "torch.optim.Adam(" in inspect.getsource(uncertainty_gated.GatedTTA.__init__)
    assert "AdamW" not in inspect.getsource(eata.EATA.__init__)


def test_resnet18_bn_affine_count():
    from src.models.resnet18 import build_resnet18
    from src.tta.bn import collect_bn_affine

    m = build_resnet18(pretrained=False)
    params = collect_bn_affine(m)
    assert len(params) == 40  # 20 BN modules × (γ, β); NOT 106 (that is ResNet-50)


@pytest.mark.skip(reason="ResNet-50 replacement architecture archived to archive/unrequested-research-extensions per charter")
def test_resnet50_bn_affine_count():
    from src.models.resnet50 import build_resnet50
    from src.tta.bn import collect_bn_affine

    m = build_resnet50(pretrained=False)
    params = collect_bn_affine(m)
    assert len(params) == 106


def test_official_tau_not_0p05():
    import yaml

    tau = yaml.safe_load((ROOT / "configs" / "tau_by_seed.yaml").read_text())["tau"]
    for seed, v in tau.items():
        assert float(v) < 1e-3, f"seed {seed} tau={v} looks like coverage, not U"
        assert float(v) > 0


def test_canonical_camelyon17_transforms():
    """Verify src.data.transforms is the single canonical transform implementation."""
    from torchvision import transforms
    import src.data.transforms as canonical_tf
    import src.data.camelyon17 as loader_tf

    # Verify functions are shared / canonical
    assert loader_tf.get_camelyon17_train_transform is canonical_tf.get_camelyon17_train_transform
    assert loader_tf.get_camelyon17_eval_transform is canonical_tf.get_camelyon17_eval_transform

    train_tf = loader_tf.get_transform(train=True)
    eval_tf = loader_tf.get_transform(train=False)

    train_classes = [type(t) for t in train_tf.transforms]
    eval_classes = [type(t) for t in eval_tf.transforms]

    # No RandomRotation anywhere in active pipeline
    assert transforms.RandomRotation not in train_classes
    assert transforms.RandomRotation not in eval_classes

    # Canonical Train pipeline: Resize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, ToTensor, Normalize
    expected_train = [
        transforms.Resize,
        transforms.RandomHorizontalFlip,
        transforms.RandomVerticalFlip,
        transforms.ColorJitter,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert train_classes == expected_train

    # Canonical Eval pipeline: Resize, ToTensor, Normalize
    expected_eval = [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert eval_classes == expected_eval

    # Check ColorJitter parameters: brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
    cj = [t for t in train_tf.transforms if isinstance(t, transforms.ColorJitter)][0]
    assert cj.brightness == (0.9, 1.1)
    assert cj.contrast == (0.9, 1.1)
    assert cj.saturation == (0.9, 1.1)
    assert cj.hue == (-0.05, 0.05)

    # Check ImageNet normalization parameters
    for tf_obj in (train_tf, eval_tf):
        norm = [t for t in tf_obj.transforms if isinstance(t, transforms.Normalize)][0]
        assert tuple(norm.mean) == (0.485, 0.456, 0.406)
        assert tuple(norm.std) == (0.229, 0.224, 0.225)


def test_benchmark_training_configuration():
    """Verify scripts/benchmark_training.py uses authoritative training configuration."""
    from scripts import benchmark_training
    import inspect
    import torch.nn as nn
    from src.models.resnet18 import build_resnet18, OFFICIAL_DROPOUT_P, OFFICIAL_NUM_CLASSES

    # Verify parser defaults match authoritative source-training protocol
    p = benchmark_training.parse_args([])
    assert p.lr == 1e-4
    assert p.weight_decay == 1e-4
    assert p.max_epochs == 20
    assert p.patience == 5

    # Verify source inspection of benchmark_training:
    src = inspect.getsource(benchmark_training)
    assert "nn.BCEWithLogitsLoss()" in src
    assert "CrossEntropyLoss" not in src
    assert "torch.optim.Adam(" in src
    assert "AdamW" not in src
    assert "Camelyon17Dataset" not in src

    # Verify model built matches 1-logit, dropout 0.5, ResNet-18
    m = build_resnet18(pretrained=False, dropout_p=OFFICIAL_DROPOUT_P, num_classes=OFFICIAL_NUM_CLASSES)
    assert m.num_classes == 1
    assert m.dropout_p == 0.5
    out = m(torch.randn(2, 3, 96, 96))
    assert out.shape == (2, 1)

    # Verify loss computation works with (B, 1) logits and labels
    crit = nn.BCEWithLogitsLoss()
    dummy_labels = torch.tensor([0.0, 1.0]).view(-1, 1)
    loss = crit(out, dummy_labels)
    assert loss.dim() == 0


def test_canonical_configuration_and_reproducibility():
    """Step 5: Enforce canonical configuration files, seed sets, and reproduction commands."""
    import yaml
    from pathlib import Path

    repo_root = ROOT
    canonical_configs = [
        repo_root / "configs" / "camelyon17.yaml",
        repo_root / "configs" / "authoritative_resnet18_utta.yaml",
        repo_root / "configs" / "tau_by_seed.yaml",
        repo_root / "configs" / "FINAL" / "source_resnet18.yaml",
        repo_root / "configs" / "FINAL" / "tta_utta_med.yaml",
        repo_root / "configs" / "FINAL" / "evaluation.yaml",
    ]

    # 1. Verify canonical configs exist and parse cleanly
    for cfg_p in canonical_configs:
        assert cfg_p.exists(), f"Canonical config missing: {cfg_p}"
        loaded = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict) and len(loaded) > 0, f"Config empty: {cfg_p}"

    # 2. Verify official 5-seed set is preserved
    expected_seeds = {42, 123, 2024, 7, 99}
    cam17_cfg = yaml.safe_load((repo_root / "configs" / "camelyon17.yaml").read_text())
    auth_cfg = yaml.safe_load((repo_root / "configs" / "authoritative_resnet18_utta.yaml").read_text())
    tau_cfg = yaml.safe_load((repo_root / "configs" / "tau_by_seed.yaml").read_text())

    assert set(cam17_cfg["protocol"]["seeds"]) == expected_seeds
    assert set(auth_cfg["protocol"]["seeds"]) == expected_seeds
    assert set(tau_cfg["tau"].keys()) == expected_seeds

    # 3. Verify tau values are loaded from canonical config and remain positive floats < 1e-3
    for s in expected_seeds:
        tau_val = float(tau_cfg["tau"][s])
        assert 0 < tau_val < 1e-3, f"Seed {s} tau={tau_val} out of bounds"

    # 4. Verify source-training default optimizer is Adam
    src_cfg = yaml.safe_load((repo_root / "configs" / "FINAL" / "source_resnet18.yaml").read_text())
    assert src_cfg["training"]["optimizer"].lower() == "adam"
    assert auth_cfg["source"]["optimizer"].lower() == "adam"

    # 5. Verify no stale train_baseline.py in active reproduction documents
    active_docs = [
        repo_root / "reproducibility" / "reproduction_commands.md",
        repo_root / "docs" / "REPRODUCIBILITY.md",
        repo_root / "README.md",
        repo_root / "START_HERE.txt",
        repo_root / "LOCAL_RUN.md",
    ]
    for doc in active_docs:
        if doc.exists():
            content = doc.read_text(encoding="utf-8")
            assert "train_baseline.py" not in content, f"Stale train_baseline.py reference in {doc.name}"
            assert "train_baseline," not in content, f"Stale train_baseline reference in {doc.name}"

    # 6. Verify reproduction commands do not contain explicit --tau CLI override
    repro_cmds = (repo_root / "reproducibility" / "reproduction_commands.md").read_text(encoding="utf-8")
    assert " --tau " not in repro_cmds and " --tau=" not in repro_cmds, "reproduction_commands.md must resolve tau automatically from config, not via --tau override"


def test_dependency_specifications_harmonized():
    """Step 6: Enforce harmonization between root and reproducibility dependency files."""
    repo_root = ROOT

    root_req_p = repo_root / "requirements.txt"
    repro_req_p = repo_root / "reproducibility" / "requirements.txt"
    lock_req_p = repo_root / "reproducibility" / "requirements-lock.txt"
    env_p = repo_root / "environment.yml"
    repro_env_p = repo_root / "reproducibility" / "environment.yml"

    for p in [root_req_p, repro_req_p, lock_req_p, env_p, repro_env_p]:
        assert p.exists(), f"Dependency specification missing: {p}"

    def parse_req_packages(text: str) -> set:
        packages = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg = line.split(">=")[0].split("==")[0].split("<=")[0].strip()
            packages.add(pkg)
        return packages

    root_pkgs = parse_req_packages(root_req_p.read_text(encoding="utf-8"))
    repro_pkgs = parse_req_packages(repro_req_p.read_text(encoding="utf-8"))

    # requirements.txt and reproducibility/requirements.txt must be synchronized
    assert root_pkgs == repro_pkgs, f"Mismatch between requirements.txt and reproducibility/requirements.txt: {root_pkgs ^ repro_pkgs}"

    # Active code does not import umap; umap-learn must not linger in requirements
    assert "umap-learn" not in root_pkgs
    assert "umap-learn" not in repro_pkgs

    # Core active runtime packages must all be declared
    required_core = {
        "torch", "torchvision", "numpy", "pandas", "pyarrow", "scipy",
        "scikit-learn", "matplotlib", "seaborn", "pyyaml", "tqdm", "pillow", "pytest",
    }
    assert required_core.issubset(root_pkgs), f"Missing core package in requirements: {required_core - root_pkgs}"

    # Locked frozen versions must match authoritatively established pinned versions
    lock_text = lock_req_p.read_text(encoding="utf-8")
    assert "torch==2.1.2" in lock_text
    assert "torchvision==0.16.2" in lock_text
    assert "numpy==1.26.4" in lock_text
    assert "pandas==2.2.2" in lock_text
    assert "pyarrow==16.1.0" in lock_text
    assert "scipy==1.13.1" in lock_text
    assert "scikit-learn==1.5.0" in lock_text
    assert "wilds==2.0.0" in lock_text
