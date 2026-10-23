# Official Reproduction Commands

## 1. Environment Setup
```bash
git clone https://github.com/Avirech7676/UTTA-Med.git
cd UTTA-Med
git checkout 17ec6839825528d32e8db2fd919a754b9ed140b0
conda env create -f reproducibility/environment.yml
# Or with pip:
pip install -r reproducibility/requirements.txt
```

## 2. Test Suite & Leakage Guardrails
```bash
pytest tests/ -v
```

## 3. Source Training (Seeds 42, 123, 2024, 7, 99)
```bash
python scripts/train_source.py --seed 42 --config configs/camelyon17.yaml
```

## 4. Test-Time Adaptation on Target Hospital 2
```bash
# Tent (Unconstrained, bn_freeze_stats, lr=1e-5)
python scripts/adapt.py --checkpoint checkpoints/camelyon17_resnet18_source_s42_BEST.pt --method tent --bn-mode bn_freeze_stats --lr 1e-5

# EATA (e_margin=0.4, bn_freeze_stats, lr=1e-5)
python scripts/adapt.py --checkpoint checkpoints/camelyon17_resnet18_source_s42_BEST.pt --method eata --e-margin 0.4 --bn-mode bn_freeze_stats --lr 1e-5

# UTTA-Med (MC-Dropout N=20, tau resolved from configs/tau_by_seed.yaml, bn_freeze_stats, lr=1e-5)
python scripts/adapt.py --checkpoint checkpoints/camelyon17_resnet18_source_s42_BEST.pt --method utta --seed 42 --n-passes 20 --bn-mode bn_freeze_stats --lr 1e-5
```

## 5. Statistical Master Aggregation & Table Generation
```bash
python scripts/build_master_dataset.py
```
