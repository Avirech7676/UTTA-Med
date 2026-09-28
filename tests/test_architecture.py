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
