"""Verification script for Camelyon17 canonical configuration."""

from pathlib import Path
import yaml

CONFIG_PATH = Path("configs/camelyon17.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Dataset & Leakage
assert config["dataset"] == "camelyon17", f"Expected 'camelyon17', got {config.get('dataset')}"
assert config["leakage_free"] is True, "Config must be marked leakage_free"

# Splits & Domain IDs
sp = config["splits"]
assert sp["train"]["domain_ids"] == [0, 3, 4], "Train must be centers {0, 3, 4}"
assert sp["id_val"]["domain_ids"] == [0, 3, 4], "ID Val must be centers {0, 3, 4}"
assert sp["val"]["domain_ids"] == [1], "OOD Val must be center 1"
assert sp["test"]["domain_ids"] == [2], "Test target must be center 2"

# Verification of disjoint hospital partition
train_domains = set(sp["train"]["domain_ids"])
val_domains = set(sp["val"]["domain_ids"])
test_domains = set(sp["test"]["domain_ids"])
assert train_domains.isdisjoint(val_domains), "Train and Val domains must be disjoint"
assert train_domains.isdisjoint(test_domains), "Train and Test domains must be disjoint"
assert val_domains.isdisjoint(test_domains), "Val and Test domains must be disjoint"

# Protocol parameters
proto = config["protocol"]
assert proto["backbone"] == "resnet18", "Official backbone is resnet18"
assert proto["head"] == "1-logit BCE", "Official head is 1-logit BCE"
assert proto["dropout"] == 0.5, "Official dropout is 0.5"
assert proto["tta_bn_mode"] == "bn_freeze_stats", "Official BN mode is bn_freeze_stats"
assert proto["n_mc"] == 20, "Official MC passes is 20"
assert proto["seeds"] == [42, 123, 2024, 7, 99], "Official 5 seeds required"

print("=" * 60)
print("Camelyon17 configuration verification")
print("=" * 60)
print("Status: PASS (Zero Leakage Verified)")
print("Source hospitals (train):", sp["train"]["domain_ids"])
print("OOD validation hospital :", sp["val"]["domain_ids"])
print("Target test hospital    :", sp["test"]["domain_ids"])
print("Official seeds          :", proto["seeds"])
print("Architecture            :", proto["backbone"], proto["head"], f"dropout={proto['dropout']}")
print("TTA BN mode             :", proto["tta_bn_mode"])
print("=" * 60)
