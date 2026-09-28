#!/usr/bin/env python3
"""Generate UTTA-Med Complete Project Guide + 4-Week Action Plan PDF."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os

OUTPUT = "/home/workdir/artifacts/UTTA-Med/UTTA-Med_Project_Guide_and_4Week_Plan.pdf"

# Colors
PRIMARY = HexColor("#1a365d")      # dark blue
SECONDARY = HexColor("#2b6cb0")    # medium blue
ACCENT = HexColor("#c53030")       # red for warnings
LIGHT_BG = HexColor("#edf2f7")     # light gray-blue
SUCCESS = HexColor("#276749")      # green
TABLE_HEADER = HexColor("#2c5282")
TABLE_ALT = HexColor("#ebf8ff")

def make_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=4,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        name="SectionHead",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold",
        borderPadding=3,
    ))
    styles.add(ParagraphStyle(
        name="SubHead",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=12,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        name="BodyTight",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=3,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        name="BulletText",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        leftIndent=12,
        spaceAfter=2,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        backColor=LIGHT_BG,
        borderPadding=4,
        spaceBefore=4,
        spaceAfter=6,
        fontName="Courier"
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=white,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="Warning",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=ACCENT,
        fontName="Helvetica-Bold",
        spaceBefore=4,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=HexColor("#718096"),
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=HexColor("#4a5568"),
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=8
    ))
    return styles


def header_footer(canvas, doc):
    canvas.saveState()
    # Header
    canvas.setStrokeColor(PRIMARY)
    canvas.setLineWidth(1.5)
    canvas.line(0.7*inch, A4[1] - 0.55*inch, A4[0] - 0.7*inch, A4[1] - 0.55*inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(PRIMARY)
    canvas.drawString(0.7*inch, A4[1] - 0.45*inch, "UTTA-Med — Project Guide & 4-Week Action Plan")
    canvas.drawRightString(A4[0] - 0.7*inch, A4[1] - 0.45*inch, "Master Plan v4 (Frozen)")

    # Footer
    canvas.setStrokeColor(HexColor("#cbd5e0"))
    canvas.setLineWidth(0.5)
    canvas.line(0.7*inch, 0.55*inch, A4[0] - 0.7*inch, 0.55*inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#718096"))
    canvas.drawString(0.7*inch, 0.4*inch, "Confidential — Research Use")
    canvas.drawRightString(A4[0] - 0.7*inch, 0.4*inch, f"Page {doc.page}")
    canvas.restoreState()


def make_table(headers, rows, col_widths):
    styles = make_styles()
    data = [[Paragraph(h, styles["TableHeader"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), styles["TableCell"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#a0aec0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf():
    styles = make_styles()
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=0.7*inch,
        rightMargin=0.7*inch,
        topMargin=0.75*inch,
        bottomMargin=0.7*inch,
        title="UTTA-Med Project Guide and 4-Week Action Plan",
        author="UTTA-Med Research Team",
    )

    story = []

    # ========== COVER ==========
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("UTTA-Med", styles["CoverTitle"]))
    story.append(Paragraph(
        "Uncertainty-Aware Test-Time Adaptive Deep Learning<br/>for Reliable Medical Image Classification Under Distribution Shift",
        styles["CoverSubtitle"]
    ))
    story.append(Spacer(1, 0.25*inch))
    story.append(HRFlowable(width="80%", thickness=2, color=PRIMARY, spaceBefore=4, spaceAfter=12, hAlign="CENTER"))
    story.append(Paragraph("<b>Complete Project Explanation</b><br/>+ <b>4-Week Team Action Plan</b>", styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Master Plan v4 — Frozen for Implementation", styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.4*inch))

    meta = [
        ["Primary Dataset", "Camelyon17-WILDS"],
        ["Primary Model", "ResNet-18"],
        ["Core Method", "MC-Dropout Uncertainty-Gated TTA"],
        ["Mandatory Baselines", "Source-only + Tent + EATA"],
        ["Gate Controls", "Random / Confidence / Entropy"],
        ["Statistics", "≥5 seeds + WSI-level bootstrap"],
        ["Document Date", datetime.now().strftime("%Y-%m-%d")],
    ]
    meta_data = [[Paragraph(f"<b>{k}</b>", styles["TableCell"]), Paragraph(v, styles["TableCell"])] for k, v in meta]
    meta_table = Table(meta_data, colWidths=[2.2*inch, 3.5*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#a0aec0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "<i>This document consolidates the full conceptual foundation and a realistic 4-week execution plan for a research team. "
        "All experimental rules from Master Plan v4 are binding.</i>",
        styles["Caption"]
    ))
    story.append(PageBreak())

    # ========== PART 1 ==========
    story.append(Paragraph("Part 1 — Complete Project Explanation", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph("1.1 One-Sentence Goal", styles["SubHead"]))
    story.append(Paragraph(
        "We want to answer: <b>Can we make test-time adaptation safer and more reliable for medical image classification "
        "under real hospital distribution shift by only adapting on low-uncertainty samples?</b>",
        styles["Body"]
    ))

    story.append(Paragraph("1.2 The Real-World Problem", styles["SubHead"]))
    story.append(Paragraph(
        "A deep learning model trained on histopathology patches from Hospitals A, B, C usually performs worse when "
        "deployed on Hospital D (different scanner, staining protocol, laboratory, patient population). This is called "
        "<b>distribution shift</b> (or domain shift).",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Formally: <font face='Courier'>P_source(X,Y) ≠ P_target(X,Y)</font>. Accuracy, AUROC, and especially "
        "<b>confidence calibration</b> degrade. The model becomes overconfident on wrong predictions.",
        styles["Body"]
    ))

    story.append(Paragraph("1.3 Test-Time Adaptation (TTA)", styles["SubHead"]))
    story.append(Paragraph(
        "TTA tries to fix this <b>at deployment time</b> using only unlabeled target images (no labels from the new hospital):",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Source hospitals (labeled) → Train model → Freeze checkpoint<br/>"
        "  ↓<br/>"
        "Unlabeled target hospital images → Adapt model (usually by minimizing entropy)<br/>"
        "  ↓<br/>"
        "Make predictions on target",
        styles["CodeBlock"]
    ))
    story.append(Paragraph(
        "<b>Classic method (Tent)</b>: Minimize the entropy of the model’s predictions while updating only BatchNorm "
        "affine parameters (γ, β). This is simple and strong, but it has a known failure mode: the model can confidently "
        "make a <b>wrong</b> prediction, then adapt using that wrong signal, and become even worse (error amplification / "
        "harmful adaptation).",
        styles["Body"]
    ))

    story.append(Paragraph("1.4 Our Core Idea (UTTA-Med)", styles["SubHead"]))
    story.append(Paragraph(
        "We estimate how uncertain the model is about each target sample using <b>Monte Carlo Dropout</b> (keep dropout "
        "active at inference and run multiple forward passes).",
        styles["Body"]
    ))
    story.append(Paragraph(
        "If uncertainty U(x) is high → skip the sample (do not adapt on it).<br/>"
        "If uncertainty is low → allow the sample to contribute to the entropy-minimization update.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Gate:  w(x) = 1 if U(x) &lt; τ, else 0<br/><br/>"
        "Adaptation loss (only on accepted samples):<br/>"
        "L = Σ(wᵢ · H(pᵢ)) / Σ wᵢ    (if Σ wᵢ = 0, skip the batch entirely — no epsilon smoothing)",
        styles["CodeBlock"]
    ))

    story.append(Paragraph("1.5 Why This Design Is Strong for a Best-Paper Attempt", styles["SubHead"]))
    story.append(Paragraph(
        "We do <b>not</b> claim “first uncertainty + TTA for medical images”. The literature is too close (EATA, EATA-C, "
        "SAR, COME, CertainTTA, etc.). Our defensible contributions are:",
        styles["Body"]
    ))
    bullets = [
        "<b>Reproducible leakage-free Camelyon17-WILDS protocol</b> for TTA.",
        "<b>Hard MC-Dropout predictive-variance gate</b> (Bayesian-style epistemic uncertainty) instead of point-estimate entropy or confidence.",
        "<b>Gate-control ablations</b> (Random / Confidence / Entropy) + <b>matched-coverage</b> protocol → isolates whether the uncertainty signal itself matters.",
        "<b>Joint reliability evaluation</b>: accuracy + calibration (ECE, Brier) + coverage + harmful-adaptation rate + Grad-CAM.",
        "<b>WSI-level bootstrap statistics</b> (patches from the same slide are correlated — patch-level stats are invalid)."
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["BulletText"]))

    story.append(Paragraph("1.6 Primary Dataset — Camelyon17-WILDS", styles["SubHead"]))
    story.append(Paragraph(
        "• 96×96 RGB patches from lymph-node whole-slide images.<br/>"
        "• Binary label: tumor vs non-tumor (based on the central 32×32 region).<br/>"
        "• 5 hospitals → natural multi-center shift.<br/>"
        "• Official WILDS splits must be used (never re-split yourself).<br/>"
        "• <b>Critical</b>: Hospital numbering differs across documents → we load the official <font face='Courier'>wilds</font> "
        "package, inspect the real domain IDs, and freeze the mapping in <font face='Courier'>configs/camelyon17.yaml</font>. "
        "That file becomes the single source of truth.",
        styles["Body"]
    ))

    story.append(Paragraph("1.7 Key Technical Concepts the Whole Team Must Master", styles["SubHead"]))
    concept_headers = ["Concept", "What it is", "Why it matters here"]
    concept_rows = [
        ["Domain / Distribution Shift", "Source and target data come from different distributions", "The entire problem"],
        ["Test-Time Adaptation (TTA)", "Adapt a frozen model using unlabeled target data at inference", "Core technique family"],
        ["Entropy Minimization (Tent)", "Force the model to make confident predictions", "Standard TTA baseline"],
        ["Monte Carlo Dropout", "Approximate Bayesian uncertainty by multiple stochastic forward passes", "Our uncertainty estimator"],
        ["Epistemic vs Aleatoric Uncertainty", "Model uncertainty vs data noise", "We target epistemic uncertainty"],
        ["Calibration (ECE, Brier)", "Does 80% confidence really mean 80% accuracy?", "Reliability is as important as accuracy"],
        ["Harmful Adaptation", "Correct prediction becomes wrong after TTA", "Central reliability metric"],
        ["Coverage", "Fraction of target samples that pass the gate", "Prevents “we just adapted on more data” confound"],
        ["Matched Coverage", "Force competing gates to accept ~same % of samples", "Fair comparison"],
        ["WSI-level Bootstrap", "Resample whole slides, not individual patches", "Correct statistical inference for histopathology"],
        ["Grad-CAM", "Visualize which regions the model looks at", "Explainability audit (not clinical validation)"],
        ["BatchNorm Affine Only", "Update only γ and β of BatchNorm layers", "Fair update-scope parity across all methods"],
    ]
    story.append(make_table(concept_headers, concept_rows, [1.6*inch, 2.5*inch, 2.2*inch]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1.8 Mandatory Baselines & Controls", styles["SubHead"]))
    base_headers = ["ID", "Method", "Gate / Notes"]
    base_rows = [
        ["B1", "Source-only (ResNet-18)", "None — lower bound"],
        ["B4", "Tent", "None — standard TTA"],
        ["B5", "EATA", "Method-specific (frozen second baseline)"],
        ["B6", "Random-gated TTA", "Random"],
        ["B7", "Confidence-gated TTA", "Softmax confidence"],
        ["B10", "Entropy-gated TTA", "Predictive entropy"],
        ["B8", "UTTA-Med", "MC-Dropout predictive variance (proposed)"],
    ]
    story.append(make_table(base_headers, base_rows, [0.7*inch, 2.4*inch, 3.2*inch]))
    story.append(Paragraph(
        "All TTA methods update <b>only BatchNorm affine parameters</b> (update-scope parity).",
        styles["Body"]
    ))

    story.append(Paragraph("1.9 Absolute Rules (Never Break)", styles["SubHead"]))
    story.append(Paragraph(
        "1. Target labels are used <b>only</b> for final evaluation.<br/>"
        "2. τ and all hyperparameters are selected on the OOD validation hospital only.<br/>"
        "3. Official WILDS splits only — never re-split.<br/>"
        "4. No epsilon smoothing for empty gates — explicitly skip the batch.<br/>"
        "5. Report coverage for every gated method.<br/>"
        "6. Primary statistics = WSI-level bootstrap.<br/>"
        "7. Minimum 5 seeds (fixed list: 42, 123, 2024, 7, 99).",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Violation of any of the above invalidates the scientific claims of the paper.",
        styles["Warning"]
    ))

    story.append(Paragraph("1.10 What Counts as Success", styles["SubHead"]))
    story.append(Paragraph(
        "We do <b>not</b> require UTTA-Med to win every metric. Positive, mixed, or negative results are all scientifically "
        "valid if the experiments are rigorous and honestly reported. The paper wins by clarity of analysis, not by forced superiority.",
        styles["Body"]
    ))

    story.append(PageBreak())

    # ========== PART 2 ==========
    story.append(Paragraph("Part 2 — Complete 4-Week Team Action Plan", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    story.append(Paragraph(
        "<b>Assumptions</b>: Team of 3–5 people; one strong PyTorch lead; one data/statistics lead; one evaluation/results lead; "
        "shared Git repo (already initialized); at least one GPU (8–16 GB VRAM recommended; can start on CPU/subset).",
        styles["Body"]
    ))

    # Week 1
    story.append(Paragraph("Week 1 — Foundation, Data, Source Model, Shift Demonstration", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Goal</b>: Everything needed to prove the problem exists and the pipeline is leakage-free.",
        styles["Body"]
    ))
    w1_headers = ["Day", "Focus", "Deliverables", "Owner"]
    w1_rows = [
        ["1–2", "Infrastructure", "Python 3.10+ env, pip install -r requirements.txt, Git workflow, basic tracking helper, dummy run logged", "Infra lead"],
        ["2–3", "Data Layer", "PathMNIST + PCam loaders; Camelyon17 official wilds loader; inspect real domain IDs and freeze configs/camelyon17.yaml", "Data lead"],
        ["3", "Leakage Tests", "tests/test_no_leakage.py passes (WSI intersection = empty)", "Data lead"],
        ["3–4", "Dataset Exploration", "Class balance, hospital/WSI distributions, sample images → results/data/", "Data lead"],
        ["4–5", "Source Model", "ResNet-18 + classification head, source-only training, source metrics", "Model lead"],
        ["5–6", "Distribution Shift", "Target evaluation of frozen source model; Drop metrics; optional UMAP/t-SNE", "All"],
        ["6–7", "Buffer + Review", "Fix any leakage or split issues; Stage 0–5 acceptance gates passed", "Team review"],
    ]
    story.append(make_table(w1_headers, w1_rows, [0.6*inch, 1.3*inch, 3.3*inch, 1.1*inch]))
    story.append(Paragraph(
        "<b>Week 1 Exit Gate</b>: Source model trains, official splits verified, no leakage, clear performance drop on target hospital documented.",
        styles["Body"]
    ))

    # Week 2
    story.append(Paragraph("Week 2 — Standard TTA + Uncertainty Estimation", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Goal</b>: Working Tent + EATA + reliable MC-Dropout uncertainty.",
        styles["Body"]
    ))
    w2_headers = ["Day", "Focus", "Deliverables"]
    w2_rows = [
        ["1–2", "Tent", "Correct BN-affine-only Tent implementation; same update scope as all future methods"],
        ["2–3", "EATA", "EATA baseline (restrict to BN-affine if needed); document any deviation"],
        ["3–4", "MC-Dropout", "Uncertainty head / dropout kept active; N-pass predictive variance; sensitivity for N = 5,10,20,30,50"],
        ["4–5", "Uncertainty Validation", "Bin uncertainty → error rate plot (must show U↑ ⇒ Error↑). Do not proceed to gating if signal is weak"],
        ["5–6", "Basic Adaptation Loop", "Shared adaptation engine that all gated methods will reuse"],
        ["6–7", "Buffer + Code Review", "Clean, tested Tent / EATA / MC-Dropout; uncertainty–error relationship confirmed"],
    ]
    story.append(make_table(w2_headers, w2_rows, [0.7*inch, 1.6*inch, 4.0*inch]))
    story.append(Paragraph(
        "<b>Week 2 Exit Gate</b>: Tent and EATA improve over source-only (or the failure is understood); MC-Dropout uncertainty correlates with error.",
        styles["Body"]
    ))

    # Week 3
    story.append(Paragraph("Week 3 — UTTA-Med Core + Gate Controls + Reliability", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Goal</b>: Full proposed method + all controls + calibration + harmful-adaptation analysis.",
        styles["Body"]
    ))
    w3_headers = ["Day", "Focus", "Deliverables"]
    w3_rows = [
        ["1", "UTTA-Med Gate", "Hard gate + zero-accepted-batch skip; τ sweep on validation only"],
        ["1–2", "Gate Controls", "Random, Confidence, Entropy gates with matched-coverage protocol"],
        ["2–3", "Harmful Adaptation", "Before/After vs Ground-Truth analysis; Correction Rate + Harm Rate"],
        ["3–4", "Calibration", "ECE, Brier, reliability diagrams for every method"],
        ["4–5", "τ Sensitivity + Stability", "Full τ curves; adaptation-step sweep (1/2/5/10)"],
        ["5–6", "Ablation Matrix", "Complete A–F table (including B10 Entropy)"],
        ["6–7", "Multi-seed Dry Run", "At least 2–3 seeds on the main matrix to catch bugs"],
    ]
    story.append(make_table(w3_headers, w3_rows, [0.7*inch, 1.8*inch, 3.8*inch]))
    story.append(Paragraph(
        "<b>Week 3 Exit Gate</b>: UTTA-Med end-to-end works; all gate controls run; coverage, harm rate, ECE reported; τ frozen from validation.",
        styles["Body"]
    ))

    # Week 4
    story.append(Paragraph("Week 4 — Full Experiments, Statistics, Figures, Paper Skeleton", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Goal</b>: Publication-ready results + first complete paper draft.",
        styles["Body"]
    ))
    w4_headers = ["Day", "Focus", "Deliverables"]
    w4_rows = [
        ["1–3", "Full Multi-seed Matrix", "≥5 seeds on all mandatory methods (B1, B4, B5, B6, B7, B10, B8)"],
        ["2–3", "WSI-level Statistics", "bootstrap_ci_by_slide() implemented and used for all headline CIs"],
        ["3–4", "Grad-CAM", "Correct / Incorrect / High-U / Wrong→Correct / Correct→Wrong examples"],
        ["4", "Figures & Tables", "All required figures + main results table populated with real numbers"],
        ["4–5", "Reproducibility Audit", "Every number in tables traceable to experiment ID + Git commit + config"],
        ["5–6", "Paper Skeleton", "Introduction, Related Work (explicit EATA/SAR differentiation), Method, Experiments, Ablation, Discussion, Limitations"],
        ["6–7", "Final Review + Buffer", "Team walkthrough of results honesty; optional MedMNIST-C if time remains"],
    ]
    story.append(make_table(w4_headers, w4_rows, [0.7*inch, 1.7*inch, 3.9*inch]))
    story.append(Paragraph(
        "<b>Week 4 Exit Gate</b>: All mandatory experiments finished, statistics correct, figures ready, first full paper draft written, repository publicly releasable.",
        styles["Body"]
    ))

    story.append(PageBreak())

    # Roles & Risk
    story.append(Paragraph("Team Role Suggestions (Flexible)", styles["SubHead"]))
    story.append(Paragraph(
        "• <b>Data & Protocol Lead</b>: Camelyon17 mapping, leakage tests, WSI bootstrap, statistical analysis.<br/>"
        "• <b>Core Method Lead</b>: ResNet, Tent, EATA, MC-Dropout, UTTA-Med gate, adaptation engine.<br/>"
        "• <b>Evaluation & Results Lead</b>: Metrics, calibration, harmful adaptation, Grad-CAM, tables, figures.<br/>"
        "• <b>Paper Lead</b>: Related Work positioning, writing, reproducibility checklist.<br/>"
        "• Everyone shares experiment running and debugging.",
        styles["Body"]
    ))

    story.append(Paragraph("Risk Buffer Rules", styles["SubHead"]))
    story.append(Paragraph(
        "If something slips, <b>never cut</b>: leakage control, matched coverage, gate controls, WSI-level stats, 5 seeds, target-label isolation.<br/>"
        "Cut first: ResNet-50, SAR, DLTTA, EATA-C, MedMNIST-C, temperature scaling, 10-seed extension.",
        styles["Body"]
    ))

    story.append(Paragraph("Daily Stand-up Suggestion (15 min)", styles["SubHead"]))
    story.append(Paragraph(
        "1. What gate did we pass yesterday?<br/>"
        "2. What is blocked today?<br/>"
        "3. Any risk of target-label leakage or statistical incorrectness?",
        styles["Body"]
    ))

    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceBefore=8, spaceAfter=10))
    story.append(Paragraph("Final Status", styles["SubHead"]))
    story.append(Paragraph(
        "Master Plan v4 is frozen. Repository Stage 0 is complete (docs, configs, Git). "
        "The team can begin Week 1 Day 1 immediately: environment setup → official WILDS loader inspection → freeze "
        "<font face='Courier'>configs/camelyon17.yaml</font> → leakage tests → source model.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Negative or mixed experimental results remain scientifically valid and must be reported honestly. "
        "The scientific objective is to establish whether MC-Dropout predictive uncertainty provides useful information "
        "for safer and more reliable TTA — not to force UTTA-Med to win every metric.",
        styles["Body"]
    ))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "— End of Document —",
        styles["Caption"]
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF written to: {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build_pdf()
