# Research Highlights: UTTA-Med v2 / SafeAdapt

*Submission to IEEE Transactions on Medical Imaging (TMI) / Medical Image Analysis (MedIA)*

- **Reverses TTA Attractor Collapse:** Optimal Transport prototype alignment eliminates single-class collapse on digital pathology WSIs ($Q^T \mathbf{1}_B = \mathbf{r}$).
- **Overcomes Negative Transfer Dilemma:** Elevates 5-seed Camelyon17 Macro AUROC to $0.9495 \pm 0.0134$ ($p = 0.0312$, paired Wilcoxon test).
- **Recovers Extreme Outlier WSIs:** Lifts catastrophic outlier Slide 29 AUROC from $0.0521$ (Tent) and $0.8592$ (Source) to $0.9324$.
- **Cuts Calibration Error by 43.2%:** Reduces Expected Calibration Error (ECE) from $0.1281$ to $0.0745$ and slashes harm rate to $0.0018$.
- **Decouples Acellular Stroma in Mitosis Detection:** Cellularity-Mitosis Ratio (CMR) eliminates all 14 false silence alarms, achieving $F_1 = 0.5160$ on MIDOG Scanner 3.
- **Lightweight Clinical Edge Profiling:** Operates at $53.5\text{ ms}$ latency per 32-patch batch and $474.1\text{ MB}$ peak VRAM on Nvidia Tesla T4 GPU.
