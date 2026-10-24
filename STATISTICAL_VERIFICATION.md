# STATISTICAL VERIFICATION & METRIC RECONCILIATION

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Auditor:** Senior Statistical Auditor & Computational Pathology Researcher  

---

## 1. RECONCILING THE PRECISION-RECALL DISCREPANCY

A major discrepancy existed in previous documentation regarding the performance of the Confidence Gating baseline:
- **Earlier Documentation:** $\text{Precision} = 0.9855, \quad \text{Recall} = 0.7200, \quad F_1 = 0.8310$.
- **Recent Turn Documentation:** $\text{Precision} = 0.8124, \quad \text{Recall} = 0.8506, \quad F_1 = 0.8310$.

### Forensic Root Cause Analysis:
By recomputing the confusion matrix directly from the raw target prediction arrays ($N=85,054$ per seed, $42,527$ tumor, $42,527$ normal), the discrepancy was traced to an inadvertent column-header transposition:
1. **The True Physical Values:**
   - Across the 5 seeds, Confidence Gating achieves:
     - Seed 42: $\text{TP}=33,530, \text{FP}=960, \text{FN}=8,997, \text{TN}=41,567 \implies \text{Prec}=0.9722, \text{Rec}=0.7884, F_1=0.8707$
     - Seed 123: $\text{TP}=32,222, \text{FP}=641, \text{FN}=10,305, \text{TN}=41,886 \implies \text{Prec}=0.9805, \text{Rec}=0.7577, F_1=0.8548$
     - Seed 2024: $\text{TP}=30,161, \text{FP}=336, \text{FN}=12,366, \text{TN}=42,191 \implies \text{Prec}=0.9890, \text{Rec}=0.7092, F_1=0.8261$
     - Seed 7: $\text{TP}=27,921, \text{FP}=136, \text{FN}=14,606, \text{TN}=42,391 \implies \text{Prec}=0.9952, \text{Rec}=0.6565, F_1=0.7911$
     - Seed 99: $\text{TP}=29,261, \text{FP}=277, \text{FN}=13,266, \text{TN}=42,250 \implies \text{Prec}=0.9906, \text{Rec}=0.6881, F_1=0.8121$
   - **Mean Precision:** $(0.9722 + 0.9805 + 0.9890 + 0.9952 + 0.9906) / 5 = \mathbf{0.9855 \pm 0.0091}$.
   - **Mean Recall:** $(0.7884 + 0.7577 + 0.7092 + 0.6565 + 0.6881) / 5 = \mathbf{0.7200 \pm 0.0531}$.
   - **Mean Accuracy:** $(0.8829 + 0.8713 + 0.8507 + 0.8267 + 0.8408) / 5 = \mathbf{0.8545 \pm 0.0227}$.
2. **Origin of the False Values ($0.8124$ and $0.8506$):**
   - The value $0.8506$ is the exact accuracy of seed 2024 ($30,161 + 42,191) / 85,054 = 0.850659...$.
   - The value $0.8124$ was an unadapted source baseline F1 value from `scripts/generate_phase18_figures.py`.
   - In a recent markdown summary table, the row labels were misaligned, placing the accuracy value into the recall row and the figure baseline into the precision row.

### Canonical Reconciliation:
The canonical metric values for Confidence Gating ($A_1$) on the authentic 5-seed benchmark are:
$$\mathbf{Precision = 0.9855 \pm 0.0091}, \quad \mathbf{Recall = 0.7200 \pm 0.0531}, \quad \mathbf{Binary\ F_1 = 0.8310 \pm 0.0321}$$
$$\mathbf{Accuracy = 0.8545 \pm 0.0227}, \quad \mathbf{Specificity = 0.9889 \pm 0.0078}, \quad \mathbf{AUROC = 0.9355 \pm 0.0218}$$
All values are published in `CANONICAL_ORIGINAL_METRICS.csv`.

---

## 2. RIGOROUS PAIRED STATISTICAL HYPOTHESIS TESTING ($df=4$)

Because all methods were evaluated on the identical 5 source checkpoints and the exact same 85,054 target patches, paired difference testing is required.

### Table 1: Paired Seed-Level Statistical Comparisons

| Hypothesis Comparison | Seed Differences ($\Delta F_1$) | Mean Difference ($\mu_d$) | Sample Std ($s_d$) | Standard Error ($\text{SE}$) | Paired $t$-Statistic | Degrees of Freedom | Two-Sided $p$-Value | 95% Confidence Interval | Wilcoxon $p$-Value | Statistical Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$A_2$ (Original UTTA) vs. $A_1$ (Conf Gating)** | `[-0.0010, -0.0010, -0.0011, -0.0022, -0.0014]` | **-0.001340** | `0.000508` | `0.000227` | **-5.8990** | 4 | **0.004131** | `[-0.001971, -0.000709]` | 0.0625 | **$A_1$ significantly superior ($p < 0.01$)** |
| **$A_3$ (Improved UTTA) vs. $A_1$ (Conf Gating)** | `[-0.0005, -0.0004, -0.0005, -0.0011, -0.0007]` | **-0.000640** | `0.000279` | `0.000125` | **-5.1241** | 4 | **0.006867** | `[-0.000987, -0.000293]` | 0.0625 | **$A_1$ significantly superior ($p < 0.01$)** |
| **$A_3$ (Improved UTTA) vs. $A_2$ (Original UTTA)** | `[+0.0005, +0.0006, +0.0006, +0.0011, +0.0007]` | **+0.000700** | `0.000235` | `0.000105` | **+6.6742** | 4 | **0.002619** | `[+0.000409, +0.000991]` | 0.0625 | **$A_3$ significantly superior ($p < 0.01$)** |

---

## 3. METHODOLOGICAL ASSUMPTIONS & SCIENTIFIC LIMITATIONS

1. **Seed-Level vs. Patient-Level Uncertainty:**
   - The paired $t$-tests above evaluate **algorithmic stability across stochastic training initializations**. They prove that under the identical source model initialization, Confidence Gating consistently outperforms UTTA-Med ($p < 0.01$).
   - However, seed-level tests do **not** quantify patient-level sampling uncertainty across unseen clinical populations. In pathology whole-slide imaging, inter-patient variance (slide-level F1 range: $0.62 - 0.94$) is substantially larger than seed variance ($s = 0.032$). Claims of generalizability must rely on multicenter cluster sampling, not random seed replicates.
2. **Small-Sample Limitations ($N=5$):**
   - With $N=5$ degrees of freedom ($df=4$), the critical $t$-value at $\alpha=0.05$ is $t_{\text{crit}} = 2.776$. While the observed effect sizes are large ($|t| > 5.1$), statistical certainty should not be overstated. The practical difference in F1 ($\Delta = 0.0013$) is modest in magnitude.
