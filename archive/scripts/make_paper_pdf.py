#!/usr/bin/env python3
"""Team-shareable PDF of the frozen UTTA-Med paper (no LaTeX required)."""
from pathlib import Path
from fpdf import FPDF

ROOT = Path("/workspace/artifacts/UTTA-Med")
FIGS = ROOT / "figures"
IEEE = ROOT / "paper/ieee/figs"
OUT = ROOT / "paper/UTTA-Med_draft.pdf"


class Paper(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(80, 80, 80)
        self.set_x(self.l_margin)
        self.cell(0, 6, "UTTA-Med  |  frozen 5-seed draft  |  do not edit Table 1", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, str(self.page_no()), align="C")

    def h1(self, t):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(13, 33, 55)
        self.set_x(self.l_margin)
        self.multi_cell(0, 7, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, t):
        self.ln(1)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(27, 78, 121)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def body(self, t):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def italic(self, t):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def fig(self, path, caption, w=180):
        p = Path(path)
        if not p.exists():
            self.italic("[missing figure] " + str(path))
            return
        self.ln(1)
        # keep aspect
        try:
            from PIL import Image
            im = Image.open(p)
            iw, ih = im.size
            h = w * ih / iw
            # cap height
            if h > 105:
                h = 105
                w = h * iw / ih
        except Exception:
            h = 70
        # page break if needed
        if self.get_y() + h + 12 > 280:
            self.add_page()
        x = (210 - w) / 2
        self.image(str(p), x=x, w=w)
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4, caption, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def main():
    pdf = Paper(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(13, 33, 55)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, "Uncertainty-Gated Test-Time Adaptation for Reliable Cross-Hospital Histopathology Classification", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, "UTTA-Med  |  Camelyon17-WILDS  |  ResNet-18  |  5 seeds {42, 123, 2024, 7, 99}", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, "Numbers frozen. Target hospital labels used only for final evaluation.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h2("Abstract")
    pdf.body(
        "Models trained on histopathology patches from source hospitals degrade on an unseen hospital. "
        "Test-time adaptation (TTA) can recover some of that drop, but entropy minimization on every target sample can also amplify errors. "
        "We restrict TTA to low-uncertainty samples under real Camelyon17-WILDS shift (source hospitals {0,3,4} to target hospital 2). "
        "We compare source-only, Tent, EATA, and four gates (random, softmax confidence, predictive entropy, MC-Dropout variance) "
        "under a frozen protocol: BatchNorm affine updates only, running statistics frozen, TTA learning rate 1e-5, "
        "and gate thresholds chosen on hospital 1."
    )
    pdf.body(
        "On hospital 2, source AUROC is 0.930 +/- 0.009. Fair Tent is unstable: it collapses on 2/5 seeds "
        "(mean AUROC 0.698 +/- 0.324, harm 0.187 +/- 0.179). Selective gates never collapse, raise F1 on all five seeds, "
        "and cut harm to 0.007 +/- 0.003. UTTA-Med (MC-Dropout) does not outperform softmax confidence "
        "(delta AUROC < 0.001 on every seed). Random gating also collapses on 2/5 seeds, so the benefit is which samples "
        "are used, not merely using fewer of them."
    )
    pdf.italic(
        "Takeaway. Gating is what makes TTA reliable; MC-Dropout is a valid error signal but is redundant with confidence. "
        "The contribution is a leakage-controlled reliability profile, not a claim that Bayesian uncertainty uniquely wins."
    )

    pdf.h1("1. Introduction")
    pdf.body(
        "A classifier trained on labeled patches from source hospitals is often deployed on a new scanner, stain, and patient population. "
        "TTA updates a frozen source checkpoint using unlabeled target images, typically by entropy minimization on BatchNorm affine parameters (Tent). "
        "The failure mode is harmful adaptation: a wrong prediction becomes an update signal and the model degrades."
    )
    pdf.body(
        "UTTA-Med uses a hard gate: adapt only if MC-Dropout predictive variance U(x) < tau, otherwise skip. "
        "We do not claim the first uncertainty-aware TTA method. Closest prior work: EATA (entropy + diversity) and SAR (sharpness-aware TTA). "
        "Our difference: (i) an explicit MC-Dropout variance gate, (ii) matched-coverage controls against random, confidence, and entropy, "
        "(iii) joint reporting of coverage, harm rate, and calibration on a leakage-controlled Camelyon17-WILDS split."
    )

    pdf.h1("2. Method (frozen)")
    pdf.body(
        "Splits (verified from the official loader, not from documents): train hospitals 0/3/4 (302,436 patches, 30 WSI); "
        "id-val 0/3/4 (33,560); OOD val hospital 1 (34,904, 10 WSI) for tau / lr only; test hospital 2 (85,054, 10 WSI). "
        "Train intersect OOD-val/test slides is empty."
    )
    pdf.body(
        "Source: ResNet-18, 1 logit, BCE-with-logits, Dropout 0.5, Adam 1e-4, save-best on hospital-1 AUROC. "
        "TTA: continual one pass, BN gamma/beta only, running stats frozen, lr=1e-5. Empty gate: skip backward and step. "
        "tau = 70th percentile of hospital-1 U (rule frozen; numeric tau re-estimated per seed). N_MC = 20."
    )

    pdf.h1("3. Results")
    pdf.h2("Table 1. Hospital 2, mean +/- std, n = 5 seeds")
    # simple table
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(0, 0, 0)
    headers = ["Method", "AUROC", "F1", "ECE", "Harm", "Cov."]
    widths = [38, 32, 30, 30, 30, 20]
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.5)
    rows = [
        ["Source-only", "0.930 +/- 0.009", "0.807 +/- 0.025", "0.128 +/- 0.026", "--", "--"],
        ["Tent", "0.698 +/- 0.324", "0.506 +/- 0.411", "0.284 +/- 0.179", "0.187 +/- 0.179", "1.00"],
        ["EATA", "0.921 +/- 0.026", "0.809 +/- 0.031", "0.152 +/- 0.024", "0.025 +/- 0.015", "0.89"],
        ["Random gate", "0.697 +/- 0.304", "0.487 +/- 0.385", "0.298 +/- 0.166", "0.194 +/- 0.170", "0.70"],
        ["Confidence", "0.936 +/- 0.022", "0.831 +/- 0.032", "0.130 +/- 0.030", "0.007 +/- 0.003", "0.58"],
        ["UTTA-Med", "0.935 +/- 0.022", "0.830 +/- 0.032", "0.131 +/- 0.031", "0.007 +/- 0.003", "0.58"],
    ]
    for i, r in enumerate(rows):
        if i in (4, 5):
            pdf.set_font("Helvetica", "B", 8.5)
        else:
            pdf.set_font("Helvetica", "", 8.5)
        for c, w in zip(r, widths):
            pdf.cell(w, 6, c, border=1, align="C")
        pdf.ln()
    pdf.ln(2)
    pdf.italic("Entropy equals confidence in binary classification (omitted as a duplicate row). Tent/random collapse on seeds 7 and 99.")

    pdf.h2("Per-seed AUROC (hospital 2)")
    pdf.set_font("Helvetica", "B", 8.5)
    headers = ["Method", "42", "123", "2024", "7", "99"]
    widths = [38, 28, 28, 28, 28, 28]
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.5)
    rows = [
        ["Source", "0.935", "0.935", "0.932", "0.913", "0.933"],
        ["Tent", "0.929", "0.941", "0.933", "0.343", "0.344"],
        ["EATA", "0.936", "0.940", "0.932", "0.877", "0.920"],
        ["Random", "0.922", "0.937", "0.897", "0.354", "0.373"],
        ["Confidence", "0.951", "0.942", "0.956", "0.902", "0.927"],
        ["UTTA-Med", "0.951", "0.942", "0.955", "0.902", "0.928"],
    ]
    for r in rows:
        for c, w in zip(r, widths):
            pdf.cell(w, 6, c, border=1, align="C")
        pdf.ln()
    pdf.ln(3)

    pdf.body(
        "RQ1: shift is real (AUROC drop 0.070 +/- 0.009; F1 drop ~0.18). Sensitivity is the failure mode. "
        "RQ3: U vs error r = 0.40 +/- 0.02 on hospital 1; decile 1 error ~0%, decile 10 ~35-42%. "
        "RQ2: Tent collapses on 2/5 seeds even at lr=1e-5 with frozen BN stats. EATA never collapsed but does not beat source AUROC. "
        "RQ4: gating cuts harm ~25x; random at 70% coverage still collapses -- the mechanism is which samples, not fewer samples. "
        "RQ5: gated F1 improves on 5/5 seeds; AUROC mixed (up on 3/5). ECE is not a win vs source. "
        "RQ6: UTTA-Med ~ confidence (delta < 0.001 every seed)."
    )

    pdf.add_page()
    pdf.h1("4. Figures")
    pdf.fig(FIGS / "fig1_architecture.png",
            "Figure 1. UTTA-Med pipeline. Target labels never enter tau, lr, or model selection.")
    pdf.fig(FIGS / "fig3_u_error_5seed.png",
            "Figure 3. Uncertainty vs error on hospital 1, all five seeds. Mandatory gate before using U for TTA.")
    pdf.fig(FIGS / "fig5_per_seed_auroc.png",
            "Figure 5. Per-seed hospital-2 AUROC. Collapse of Tent/random on seeds 7 and 99 is the reliability result.")
    pdf.fig(FIGS / "fig8_harm_5seed.png",
            "Figure 8. Harm rate. Gating prevents the Correct-to-Wrong flips that destroy Tent.")

    pdf.add_page()
    pdf.h1("5. Grad-CAM audit (RQ7)")
    pdf.body(
        "Seed-42 source model, 512 shuffled hospital-2 patches (TP 215, TN 226, FP 15, FN 56). "
        "Interpretability audit, not clinical validation. 96x96 maps are coarse. "
        "Blank CAMs on confident normals are a 1-logit Grad-CAM artifact."
    )
    pdf.fig(FIGS / "gradcam/source_lowU_correct.png",
            "Figure 7a. Low-uncertainty correct: CAM on dense nuclei for tumors; fat/stroma often blank.")
    pdf.fig(FIGS / "gradcam/source_highU_wrong.png",
            "Figure 7b. High-uncertainty errors (p near 0.5, edge CAMs) -- the population the gate skips.")
    pdf.body(
        "Residual failure mode: confident false positives on lymphocyte-dense tissue (p > 0.96) and confident false negatives (p ~ 0.01) still pass a hard gate."
    )

    pdf.add_page()
    pdf.h1("6. Discussion and limitations")
    pdf.body(
        "Positive statement: unrestricted entropy-minimization TTA is unsafe under real hospital shift; "
        "restricting updates to high-confidence / low-variance samples makes it safe and improves F1. "
        "Honest qualifier: MC-Dropout variance did not beat softmax confidence for this binary ResNet-18."
    )
    pdf.body(
        "Limitations: (i) one dataset, one backbone, binary labels; "
        "(ii) seed-level CIs, not yet WSI-level bootstrap (10 test slides); "
        "(iii) Grad-CAM is source-only; (iv) test coverage of learned gates is below the val-matched 0.70; "
        "(v) Tent lr was selected on hospital 1 after larger lrs collapsed -- not Tent's original ImageNet hyperparameter; "
        "(vi) confident errors still pass a hard gate."
    )
    pdf.h1("7. What not to claim")
    pdf.body(
        "Do not write: first uncertainty-aware TTA for medical images; UTTA-Med outperforms confidence; "
        "Tent is useless without showing the 3 non-collapsed seeds; clinical validity of Grad-CAM; "
        "drop seeds 7 and 99."
    )
    pdf.h1("Reproducibility")
    pdf.body(
        "Data: HuggingFace wltjr1007/Camelyon17-WILDS parquet, filter by center. "
        "Checkpoints: camelyon17_resnet18_source_s{seed}_BEST.pt. "
        "Protocol: bn_freeze_stats, TTA lr 1e-5, N_MC=20, tau = val 70th percentile of U. "
        "Every headline number traces to camelyon17_resnet18_full_s{seed}.json. "
        "IEEE LaTeX: paper/ieee/utta_med.tex (Overleaf)."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
