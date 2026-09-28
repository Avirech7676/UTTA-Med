# Experiment Matrix — UTTA-Med

## Mandatory (LOCKED, 5 seeds)

| ID  | Model     | Method                    | Gate                  | Status |
|-----|-----------|---------------------------|-----------------------|--------|
| B1  | ResNet-18 | Source-only               | None                  | DONE |
| B4  | ResNet-18 | Tent                      | None                  | DONE |
| B5  | ResNet-18 | EATA                      | Entropy filter        | DONE |
| B6  | ResNet-18 | Gated TTA                 | Random                | DONE |
| B7  | ResNet-18 | Gated TTA                 | Softmax confidence    | DONE |
| B10 | ResNet-18 | Gated TTA                 | Predictive entropy    | DONE |
| B8  | ResNet-18 | UTTA-Med                  | MC-Dropout variance   | DONE |

## Optional (CODE READY — seed 42 GPU)

| ID  | Model     | Method                    | Status |
|-----|-----------|---------------------------|--------|
| —   | ResNet-18 | Temperature scaling       | CODE READY |
| —   | ResNet-18 | SAR                       | CODE READY |
| —   | ResNet-18 | EATA-C                    | CODE READY |
| —   | ResNet-18 | DLTTA                     | CODE READY |
| —   | ResNet-18 | Camelyon17-C              | CODE READY |
| B2  | ResNet-50 | Source-only               | STRETCH (code exists) |
| B9  | ResNet-50 | UTTA-Med                  | STRETCH |
| B3  | ResNet-18 | Strong augmentation       | SKIP unless reviewers ask |

## Ablation (LOCKED)

| Ablation | TTA | Gate                   |
|----------|-----|------------------------|
| A        | ✗   | —                      |
| B        | ✓   | None (= Tent)          |
| C        | ✓   | Random                 |
| D        | ✓   | Confidence             |
| E        | ✓   | Entropy                |
| F        | ✓   | MC-Dropout (UTTA-Med)  |
