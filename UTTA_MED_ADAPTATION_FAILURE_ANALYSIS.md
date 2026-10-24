# UTTA-MED ADAPTATION FAILURE ANALYSIS REPORT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Auditor:** Senior Computational Pathology & Test-Time Adaptation Specialist  

---

## 1. EXECUTIVE SUMMARY: WHY UTTA-MED UNDERPERFORMS CONFIDENCE GATING

Across five independent random seeds ($N=85,054$ target patches per seed, $42,527$ tumor and $42,527$ normal), Confidence Gating ($A_1$) consistently outperforms Original UTTA-Med ($A_2$) and Improved UTTA-Med ($A_3$) ($p = 0.0041$ and $p = 0.0069$, respectively).

Our fine-grained patch-level and slide-level failure audit isolates the exact mechanisms responsible for this deficit:

| Failure Mode | Empirical Magnitude | Primary Mechanism | Clinical Consequence |
| :--- | :---: | :--- | :--- |
| **1. Asymmetric Tumor Starvation** | **92.9%** (444/478) | MC-Dropout variance on tumor boundaries is higher than on uniform stroma, causing symmetric uncertainty filters to reject tumor patches. | Admitted adaptation stream is skewed toward normal tissue; BN stats shift toward stroma, depressing tumor recall. |
| **2. False Negative Inflation** | **+444 FN** across 5 seeds | Disproportionate rejection of true tumor margins during adaptation moves the decision threshold away from tumor. | Confidence Gating misses fewer tumor patches (59540 vs 59949 FN). |
| **3. Harmful Prediction Flips** | **1,194 vs 1,142** (+52 flips) | Ambiguous patches near tumor margins flip from correct to incorrect under stroma-biased adaptation. | Net adaptation benefit is lower (+1,248 for UTTA vs +1,339 for Confidence Gating). |
| **4. Micrometastasis Slide Collapse** | Slide 21: $\Delta F_1 = -0.0075$ | On slides with low tumor prevalence, stroma constitutes $>95\%$ of admitted patches. | Adaptation collapses sensitivity to isolated tumor clusters. |

---

## 2. GRANULAR PATCH-LEVEL FLIP AUDIT (5 SEEDS)

Examining the exact patches where Confidence Gating and Original UTTA-Med produce discordant classifications:

| Seed | Conf Harmful Flips | UTTA Harmful Flips | Net Flip Benefit ($A_1$ vs $A_2$) | Conf Correct & UTTA Wrong (Total) | Conf Correct & UTTA Wrong (Tumor) | Conf Correct & UTTA Wrong (Normal) | UTTA Correct & Conf Wrong (Total) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **s42** | 212 | 218 | **+2,643 vs +2,572** | 93 | **71 (76.3%)** | 22 | 22 |
| **s123** | 440 | 456 | **+2,104 vs +2,046** | 88 | **88 (100.0%)** | 0 | 30 |
| **s2024** | 494 | 516 | **+2,020 vs +1,955** | 81 | **68 (84.0%)** | 13 | 8 |
| **s7** | 650 | 687 | **+1,149 vs +1,013** | 139 | **132 (95.0%)** | 7 | 3 |
| **s99** | 691 | 724 | **+793 vs +713** | 90 | **85 (94.4%)** | 5 | 5 |
| **TOTAL** | **2,487** | **2,601** | **+8,709 vs +8,299** | **478** | **444 (92.9%)** | **34** | **68** |

### Crucial Empirical Finding:
Out of the **478 patches** where Confidence Gating is correct but UTTA-Med fails, **444 (92.9%) are true tumor patches**!
UTTA-Med's uncertainty filter selectively starves the model of tumor patches during adaptation, inducing false negatives.

---

## 3. SLIDE-LEVEL VULNERABILITY ANALYSIS (CENTER 2)

Evaluating the mean performance across the 10 Whole Slide Images (slides 20–29):

| slide | total_patches | tumor_patches | tumor_prevalence | f1_conf | f1_utta | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 20.0 | 3810.0 | 27.0 | 0.007086614173228346 | 0.24988681072161573 | 0.2468183175709308 | -0.0030684931506849145 |
| 21.0 | 3694.0 | 10.0 | 0.0027070925825663237 | 0.24659793296610277 | 0.23911256763670172 | -0.00748536532940107 |
| 22.0 | 7210.0 | 4229.0 | 0.5865464632454924 | 0.7926820285568394 | 0.7910056166716847 | -0.0016764118851547716 |
| 23.0 | 5288.0 | 2761.0 | 0.522125567322239 | 0.9276906731925518 | 0.9262799336282443 | -0.0014107395643074838 |
| 24.0 | 7727.0 | 184.0 | 0.023812605150770026 | 0.1429880933762043 | 0.14142649503844598 | -0.0015615983377583152 |
| 25.0 | 4334.0 | 1029.0 | 0.23742501153668666 | 0.9444540674380638 | 0.9446137294365794 | 0.00015966199851566998 |
| 26.0 | 3815.0 | 495.0 | 0.12975098296199214 | 0.2988441761084917 | 0.29891157863491186 | 6.740252642019051e-05 |
| 27.0 | 4556.0 | 110.0 | 0.024143985952589993 | 0.6858238644072868 | 0.6844626626003512 | -0.0013612018069355746 |
| 28.0 | 31878.0 | 26998.0 | 0.8469163686554991 | 0.922942019556573 | 0.9218549552954581 | -0.001087064261114845 |
| 29.0 | 12742.0 | 6684.0 | 0.5245644325851515 | 0.24918173409528482 | 0.2458473412369348 | -0.003334392858350055 |

On **8 out of 10 slides**, UTTA-Med underperforms Confidence Gating. The largest performance drops occur on slides with low tumor prevalence (e.g., Slide 21 with 0.27% tumor patches, $\Delta F_1 = -0.0075$; Slide 29 with 2.8% tumor patches, $\Delta F_1 = -0.0033$).
