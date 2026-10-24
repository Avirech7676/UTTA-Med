# Statistical Validation Methodology & Significance Protocol

**Document Version:** 1.0  
**Investigator:** Lead Medical Imaging & Biostatistical Specialist  
**Evaluation Setting:** Camelyon17-WILDS Multi-Center Cross-Hospital Generalization  
**Sample Sizes:** $N_{\text{target}} = 85{,}054$ patches across $K = 10$ patients/slides, $S = 5$ random seeds

---

## 1. Biostatistical Evaluation Design

To eliminate the statistical ambiguities identified in the previous audit (e.g. inconsistent degrees of freedom, conflation of one-sided vs. two-sided tests, and overstating confidence interval coverage), all statistical comparisons follow the established standards for medical AI validation:

### 1.1 Two-Sided Paired Student's t-Test
For any two paired methods $A$ and $B$ evaluated across $S = 5$ independent random seeds, the per-seed performance difference is:
$$d_s = M_A(s) - M_B(s), \quad s \in \{42, 123, 2024, 7, 99\}$$
The sample mean difference $\bar{d}$ and sample standard deviation $s_d$ are:
$$\bar{d} = \frac{1}{S} \sum_{s=1}^S d_s, \qquad s_d = \sqrt{\frac{1}{S-1} \sum_{s=1}^S (d_s - \bar{d})^2}$$
The standard error of the mean difference is:
$$\text{SE}(\bar{d}) = \frac{s_d}{\sqrt{S}}$$
The test statistic follows Student's t-distribution with $df = S - 1 = 4$ degrees of freedom:
$$t = \frac{\bar{d}}{\text{SE}(\bar{d})}$$
The exact two-sided $p$-value is calculated as:
$$p = 2 \cdot (1 - F_{t, df}(|t|))$$
where $F_{t, df}$ is the cumulative distribution function of Student's t-distribution with $df=4$.

---

## 2. Confidence Interval Construction & Correct Interpretation

The 95% Confidence Interval for the mean metric value across 5 seeds is constructed as:
$$\text{CI}_{95\%} = \left[ \bar{M} - t_{0.025, 4} \cdot \frac{s_M}{\sqrt{5}}, \quad \bar{M} + t_{0.025, 4} \cdot \frac{s_M}{\sqrt{5}} \right]$$
where the two-sided critical value for $df = 4$ is:
$$t_{0.025, 4} \approx 2.7764$$

### Mandatory Interpretation Rule (Audit Resolution):
If the lower bound of the 95% confidence interval falls below a target threshold (e.g. $0.8500$), the result **does not prove that the underlying population performance exceeds $0.8500$ at the 95% confidence level**, even if the sample mean exceeds $0.8500$. In such cases, the result must be reported transparently as an observed point estimate with overlapping confidence bounds.

---

## 3. Patient-Level Clustered Evaluation Considerations

The 85,054 target patches in Center 2 originate from $10$ distinct patients/WSIs (patients 040, 041, 042, 044, 045, 046, 048, 051, 052).
Because patches extracted from the same whole-slide image share common tissue preparation, fixation, staining, and scanning artifacts, intra-cluster correlation exists.

To address this:
1. **Case-Boundary Resets:** Adaptation state is reset to the source checkpoint between WSIs, ensuring that adaptation on one patient cannot corrupt or leak into subsequent patients.
2. **Slide-Stratified Metrics:** In addition to global pooled metrics across the 85,054 patches, slide-level Macro $F_1$ is evaluated to ensure robust performance across low-burden and high-burden slides alike.
