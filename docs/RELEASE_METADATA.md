# UTTA-Med: Official Release Metadata

- **GitHub Repository**: https://github.com/Avirech7676/UTTA-Med.git
- **Tracking Branch**: `master`
- **Release State**: `v1.0-official-5seed-freeze`
- **requirements.txt SHA256**: `f57f4e3e2ee2975408d6525fb3f6c7e43e2188d2ef799e53aadcd7154559f7f5`
- **environment.yml SHA256**: `9f92575ed3e5309d790468b880cdebc7da39cb112f5592e831ed591849b1a812`
- **Artifact Manifest**: [`reproducibility/ARTIFACT_MANIFEST.csv`](../reproducibility/ARTIFACT_MANIFEST.csv)
- **Frozen Protocol**:
  - Centers: Source {0, 3, 4} -> OOD Val {1} -> Target Test {2}
  - Adaptation: BN-affine parameters only with frozen BN running statistics (bn_freeze_stats)
  - TTA LR: 1e-5
  - MC passes: 20
  - Threshold tau: 70th percentile of Center 1 validation uncertainty, independently per seed
  - Seeds: 42, 123, 2024, 7, 99

## Reproduction Commands

```bash
# 1. Clone repository
git clone https://github.com/Avirech7676/UTTA-Med.git
cd UTTA-Med

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full verification test suite
pytest tests/ -q

# 4. Reproduce adaptation for Seed 42 (tau loaded from configs/tau_by_seed.yaml)
python scripts/adapt.py --checkpoint checkpoints/camelyon17_resnet18_source_s42_BEST.pt --method utta --seed 42 --n-passes 20 --bn-mode bn_freeze_stats --lr 1e-5
```
