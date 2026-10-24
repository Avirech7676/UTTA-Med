# UTTA-Med — Stage H4: Controlled Target Adaptation Audit Report

**Execution Timestamp:** 2026-10-16T14:24:02.161327Z  
**Compute Platform:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (0.0 GB VRAM)  
**Evaluated Target Windows:** 10 tiles across 1 cases  
**Target Mitoses Evaluated:** 118 certified mitotic figures  
**Primary Matching Radius:** 30.0 px (7.5 µm at 0.25 µm/px)  
**Primary Estimand Determination:** `EQUIVALENT_STRICT_MARGIN`  

## 1. Primary Seed-Macro Evaluation Matrix

| Arm | Description | Mean F1 | Mean Precision | Mean Recall | Updates | Alarms | Restorations | Fallbacks |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Arm_A | Arm_A | **0.0000** | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 |
| Arm_B | Arm_B | **0.0000** | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 |
| Arm_C | Arm_C | **0.0000** | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 |
| Arm_D | Arm_D | **0.0000** | 0.0000 | 0.0000 | 0 | 8 | 0 | 0 |
| Arm_E | Arm_E | **0.0000** | 0.0000 | 0.0000 | 0 | 3 | 1 | 1 |

## 2. Clustered Bootstrap Contrasts

| Contrast | Mean $\Delta F_1$ | 95% Bootstrap CI | Boot Tail $\le 0$ | Verdict |
|---|:---:|:---:|:---:|:---:|
| `Delta_Arm_E_vs_Arm_A_Primary_Macro` | +0.0000 | [+0.0000, +0.0000] | 1.0000 | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_E_vs_Arm_A_Secondary_Micro` | +0.0000 | [+0.0000, +0.0000] | 1.0000 | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_E_vs_Arm_D_Macro` | +0.0000 | [+0.0000, +0.0000] | 1.0000 | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_B_vs_Arm_A_Macro` | +0.0000 | [+0.0000, +0.0000] | 1.0000 | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_D_vs_Arm_A_Macro` | +0.0000 | [+0.0000, +0.0000] | 1.0000 | `EQUIVALENT_STRICT_MARGIN` |
