# Experiment Matrix — UTTA-Med

## Mandatory Experiments
| ID  | Model     | Method                    | Gate                  | Purpose                          |
|-----|-----------|---------------------------|-----------------------|----------------------------------|
| B1  | ResNet-18 | Source-only               | None                  | Lower-bound / shift damage       |
| B4  | ResNet-18 | Tent                      | None                  | Standard TTA baseline            |
| B5  | ResNet-18 | EATA                      | Method-specific       | Strong selective TTA baseline    |
| B6  | ResNet-18 | Gated TTA                 | Random                | Gate-control                     |
| B7  | ResNet-18 | Gated TTA                 | Softmax confidence    | Gate-control                     |
| B10 | ResNet-18 | Gated TTA                 | Predictive entropy    | Gate-control                     |
| B8  | ResNet-18 | UTTA-Med                  | MC-Dropout variance   | Proposed method                  |

## Optional Experiments
| ID  | Model     | Method                    | Notes                            |
|-----|-----------|---------------------------|----------------------------------|
| B2  | ResNet-50 | Source-only               | Capacity check                   |
| B3  | ResNet-18 | Strong augmentation       | Rules out “just need more aug”   |
| B9  | ResNet-50 | UTTA-Med                  | Scaling check                    |
| —   | ResNet-18 | SAR                       | Optional strong baseline         |
| —   | ResNet-18 | DLTTA / EATA-C            | Medical-specific / calibrated    |

## Ablation Matrix (Gate Controls)
| Ablation | TTA | Gate                   |
|----------|-----|------------------------|
| A        | ✗   | —                      |
| B        | ✓   | None (= Tent)          |
| C        | ✓   | Random                 |
| D        | ✓   | Confidence             |
| E        | ✓   | Entropy                |
| F        | ✓   | MC-Dropout (UTTA-Med)  |

All gated methods report coverage. Preferred comparison uses matched coverage.
