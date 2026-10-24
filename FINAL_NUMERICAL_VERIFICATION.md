# FINAL NUMERICAL VERIFICATION REPORT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Branch:** `research-project-2` | **Audit Snapshot Tag:** `aeaddb7`  
**Auditor:** Senior Computational Pathology Researcher & Statistical Auditor  

---

## 1. RECALCULATION OF FROZEN SOURCE AUROC STANDARD DEVIATION

A numerical discrepancy existed in previous documentation regarding the standard deviation of AUROC for the Frozen ResNet-18 Source ($A_0$):
- **Value A:** $0.0092$
- **Value B:** $0.0163$

### Authentic Five-Seed AUROC Values ($A_0$, Center 2, $N=85,054$ per seed):
- **Seed 42:** `0.9355`
- **Seed 123:** `0.9349`
- **Seed 2024:** `0.9320`
- **Seed 7:** `0.9134`
- **Seed 99:** `0.9328`

### Statistical Calculations:
$$\text{Mean AUROC} = \frac{0.9355 + 0.9349 + 0.9320 + 0.9134 + 0.9328}{5} = \mathbf{0.92972}$$
$$\text{Sample Standard Deviation } (s, ddof=1) = \sqrt{\frac{1}{4} \sum_{i=1}^5 (x_i - \bar{x})^2} = \mathbf{0.0092367} \approx \mathbf{0.0092}$$
$$\text{Population Standard Deviation } (\sigma, ddof=0) = \sqrt{\frac{1}{5} \sum_{i=1}^5 (x_i - \bar{x})^2} = \mathbf{0.0082615} \approx \mathbf{0.0083}$$

### Mathematical Resolution of the Discrepancy:
The lowest AUROC occurs on **Seed 7** ($0.9134$).
Calculating the maximum downward range deviation from the mean yields:
$$\Delta_{\text{down}} = \text{Mean} - \min(x) = 0.92972 - 0.9134 = \mathbf{0.01632} \approx \mathbf{0.0163}$$
An earlier reporting step inadvertently recorded this **maximum downward deviation ($0.0163$)** in place of the **sample standard deviation ($0.0092$)**.
- **Canonical True Report:** $\mathbf{\text{AUROC} = 0.9297 \pm 0.0092}$.

---

## 2. INSPECTION OF CONFIDENCE GATING ADMISSION CONDITION

The exact sample admission logic was audited in [`src/tta/gates.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/gates.py#L15-L21) and [`src/tta/uncertainty_gated.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/uncertainty_gated.py#L69-L71):

```python
# src/tta/gates.py
def score_confidence(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Softmax/sigmoid confidence = max(p, 1-p). Higher = more confident."""
    model.eval()
    p = torch.sigmoid(model(x)).view(-1)
    return torch.maximum(p, 1.0 - p)

# src/tta/uncertainty_gated.py
if self.gate == "confidence":
    return score_confidence(self.model, x) > float(self.tau)
```

### Key Verification Findings:
1. **Maximum-Class Confidence:** The admission condition threshold $\tau = 0.80$ is evaluated on $\mathbf{\max(p, 1-p)}$, **NOT** on the positive-class probability $p$.
2. **Two-Sided Admission Zone:** A patch is admitted to adapt Batch Normalization affine parameters if:
   $$p \ge 0.80 \quad (\text{high-confidence tumor}) \quad \text{OR} \quad p \le 0.20 \quad (\text{high-confidence normal stroma})$$
3. **Ambiguity Rejection:** Only patches in the boundary interval $p \in (0.20, 0.80)$ (where prediction certainty is low) are rejected from parameter updates.
4. **Protocol Uniformity:** Target patch IDs, class definitions (tumor=1, normal=0), stream ordering, and source checkpoints are 100% identical across all baseline evaluations.
