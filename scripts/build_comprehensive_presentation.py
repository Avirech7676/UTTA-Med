"""
Comprehensive SafeAdapt Presentation Generator (16:9 Widescreen)
Builds SafeAdapt_Presentation.pptx and paper/SafeAdapt_Google_Slides.pptx.

Design Principles:
- Self-explanatory, publication-grade academic presentation.
- Beautiful clinical AI palette (Deep Navy #0F172A, Clinical Teal #0D9488, Emerald #059669, Alert Red #DC2626).
- Structured Bento cards with bold lead-ins and quantitative metrics.
- High-resolution publication figures embedded directly with preserved aspect ratios.
- Styled comparison and benchmark tables for CAMELYON17, MIDOG 2021, Hardware Latency, and Ablation.
- Complete, professional speaker notes with intuitive real-world analogies embedded into every slide.
"""

import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Clinical AI Color Palette
    NAVY = RGBColor(15, 23, 42)          # #0F172A
    SLATE_DARK = RGBColor(30, 41, 59)    # #1E293B
    TEAL = RGBColor(13, 148, 136)        # #0D9488
    TEAL_DARK = RGBColor(15, 118, 110)   # #0F766E
    TEAL_BG = RGBColor(240, 253, 250)    # #F0FDFA
    TEAL_BORDER = RGBColor(153, 246, 228)# #99F6E4
    CARD_BG = RGBColor(255, 255, 255)    # White
    CARD_BORDER = RGBColor(226, 232, 240)# #E2E8F0
    TEXT_DARK = RGBColor(15, 23, 42)     # #0F172A
    TEXT_MUTED = RGBColor(100, 116, 139) # #64748B
    ACCENT_RED = RGBColor(220, 38, 38)   # #DC2626
    RED_BG = RGBColor(254, 242, 242)     # #FEF2F2
    ACCENT_GREEN = RGBColor(5, 150, 105) # #059669
    GREEN_BG = RGBColor(236, 253, 245)   # #ECFDF5
    SLATE_LIGHT = RGBColor(248, 250, 252)# #F8FAFC

    def set_slide_background(slide, color=SLATE_LIGHT):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="METHODOLOGY & CLINICAL TRANSLATION", takeaway_banner=""):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = TEAL

        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.733), Inches(0.55))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY

        if takeaway_banner:
            banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.22), Inches(11.733), Inches(0.42))
            banner.fill.solid()
            banner.fill.fore_color.rgb = TEAL_BG
            banner.line.color.rgb = TEAL_BORDER
            banner.line.width = Pt(1.0)
            tf_b = banner.text_frame
            tf_b.word_wrap = True
            tf_b.margin_left = Inches(0.15)
            tf_b.margin_top = Inches(0.04)
            tf_b.margin_right = Inches(0.15)
            tf_b.margin_bottom = Inches(0.04)
            p_b = tf_b.paragraphs[0]
            p_b.text = f"KEY TAKEAWAY:  {takeaway_banner}"
            p_b.font.size = Pt(11)
            p_b.font.bold = True
            p_b.font.color.rgb = TEAL_DARK

    def add_card(slide, left, top, width, height, title, items, badge="", accent_color=None, bg_color=CARD_BG):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = accent_color if accent_color else CARD_BORDER
        card.line.width = Pt(1.5 if accent_color else 1.0)

        pad = Inches(0.16)
        tb = slide.shapes.add_textbox(left + pad, top + pad, width - (2 * pad), height - (2 * pad))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.size = Pt(13)
        p_head.font.bold = True
        p_head.font.color.rgb = accent_color if accent_color else NAVY
        p_head.space_after = Pt(8)

        for item in items:
            p = tf.add_paragraph()
            p.space_after = Pt(6)
            if isinstance(item, tuple):
                lead_in, text = item
                run1 = p.add_run()
                run1.text = f"•  {lead_in}: "
                run1.font.bold = True
                run1.font.size = Pt(10.5)
                run1.font.color.rgb = NAVY

                run2 = p.add_run()
                run2.text = text
                run2.font.size = Pt(10.5)
                run2.font.color.rgb = SLATE_DARK
            else:
                p.text = f"•  {item}"
                p.font.size = Pt(10.5)
                p.font.color.rgb = SLATE_DARK

        if badge:
            badge_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + width - Inches(1.8), top + Inches(0.12), Inches(1.65), Inches(0.28))
            badge_shape.fill.solid()
            badge_shape.fill.fore_color.rgb = TEAL_BG if not accent_color else (RED_BG if accent_color == ACCENT_RED else GREEN_BG)
            badge_shape.line.color.rgb = accent_color if accent_color else TEAL
            badge_shape.line.width = Pt(1.0)
            tf_badge = badge_shape.text_frame
            tf_badge.margin_left = tf_badge.margin_top = tf_badge.margin_right = tf_badge.margin_bottom = 0
            p_bg = tf_badge.paragraphs[0]
            p_bg.text = badge.upper()
            p_bg.font.size = Pt(8.5)
            p_bg.font.bold = True
            p_bg.font.color.rgb = accent_color if accent_color else TEAL_DARK
            p_bg.alignment = PP_ALIGN.CENTER

    def add_image_box(slide, image_path, left, top, max_w, max_h, caption=""):
        if not os.path.exists(image_path):
            return None
        im = Image.open(image_path)
        orig_w, orig_h = im.size
        aspect = orig_w / orig_h
        box_aspect = max_w / max_h

        if aspect > box_aspect:
            w = max_w
            h = max_w / aspect
            x = left
            y = top + (max_h - h) / 2
        else:
            h = max_h
            w = max_h * aspect
            x = left + (max_w - w) / 2
            y = top

        frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x - Inches(0.06), y - Inches(0.06), w + Inches(0.12), h + Inches(0.12) + (Inches(0.3) if caption else 0))
        frame.fill.solid()
        frame.fill.fore_color.rgb = CARD_BG
        frame.line.color.rgb = CARD_BORDER
        frame.line.width = Pt(1.0)

        pic = slide.shapes.add_picture(image_path, x, y, w, h)

        if caption:
            cap_box = slide.shapes.add_textbox(x, y + h + Inches(0.05), w, Inches(0.25))
            tf_c = cap_box.text_frame
            tf_c.word_wrap = True
            tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
            p_c = tf_c.paragraphs[0]
            p_c.text = caption
            p_c.font.size = Pt(8.5)
            p_c.font.italic = True
            p_c.font.color.rgb = TEXT_MUTED
            p_c.alignment = PP_ALIGN.CENTER

        return pic

    def add_styled_table(slide, left, top, width, height, headers, rows, col_widths, highlight_row_idx=None, alert_row_idx=None):
        table_shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height)
        table = table_shape.table

        for col_idx, cw in enumerate(col_widths):
            table.columns[col_idx].width = cw

        for col_idx, h in enumerate(headers):
            cell = table.cell(0, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
            cell.text = h
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)

        for r_idx, row in enumerate(rows):
            is_hl = (highlight_row_idx is not None and r_idx == highlight_row_idx)
            is_al = (alert_row_idx is not None and r_idx == alert_row_idx)

            bg_c = TEAL_BG if is_hl else (RED_BG if is_al else (CARD_BG if r_idx % 2 == 0 else SLATE_LIGHT))
            text_c = TEAL_DARK if is_hl else (ACCENT_RED if is_al else NAVY)

            for col_idx, val in enumerate(row):
                cell = table.cell(r_idx + 1, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg_c
                cell.text = str(val)
                p = cell.text_frame.paragraphs[0]
                p.font.size = Pt(9.5)
                p.font.bold = is_hl or is_al or (col_idx == 0)
                p.font.color.rgb = text_c
                p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.04)

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, NAVY)

    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(4.5), Inches(0.35))
    pill.fill.solid(); pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    pill.line.color.rgb = TEAL; pill.line.width = Pt(1.0)
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = "IEEE TRANSACTIONS ON MEDICAL IMAGING (TMI)"
    p_pill.font.size = Pt(10); p_pill.font.bold = True; p_pill.font.color.rgb = TEAL
    p_pill.alignment = PP_ALIGN.CENTER

    t_box1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(1.8))
    tf1 = t_box1.text_frame; tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0
    p1 = tf1.paragraphs[0]
    p1.text = "SafeAdapt"
    p1.font.size = Pt(46); p1.font.bold = True; p1.font.color.rgb = TEAL
    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Robust Test-Time Adaptation via Optimal Transport and Conformal Risk Control in Digital Pathology"
    p1_sub.font.size = Pt(22); p1_sub.font.bold = True; p1_sub.font.color.rgb = RGBColor(248, 250, 252); p1_sub.space_before = Pt(6)

    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(11.3), Inches(1.0))
    tf_m = meta_box.text_frame
    tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = 0
    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "Author: Avinash Reddy Cheerapareddy   |   Co-Author: Gunnam Sanjana"
    p_m1.font.size = Pt(15); p_m1.font.bold = True; p_m1.font.color.rgb = RGBColor(226, 232, 240)
    p_m2 = tf_m.add_paragraph()
    p_m2.text = "Department of Computer Science & Engineering, SRM University-AP   •   Final Thesis & Scientific Defense"
    p_m2.font.size = Pt(13); p_m2.font.color.rgb = RGBColor(148, 163, 184); p_m2.space_before = Pt(4)

    stat_w = Inches(2.7)
    stat_h = Inches(1.9)
    stats_data = [
        ("AUROC GAIN", "+14.2%", "CAMELYON17 multicenter shifts (0.8312 → 0.9495)", TEAL),
        ("COLLAPSE RATE", "0.0%", "Eliminates 40% collapse seen in classical TTA", ACCENT_GREEN),
        ("GPU ROLLBACK", "1.8 ms", "Sub-frame fail-closed state recovery", TEAL),
        ("PATIENT ISOLATION", "HIPAA/GDPR", "Deterministic episodic slide boundary resets", ACCENT_GREEN)
    ]
    for idx, (lbl, val, desc, col) in enumerate(stats_data):
        bx = Inches(1.0) + idx * Inches(2.85)
        by = Inches(4.7)
        card_s = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, stat_w, stat_h)
        card_s.fill.solid(); card_s.fill.fore_color.rgb = RGBColor(30, 41, 59)
        card_s.line.color.rgb = col; card_s.line.width = Pt(1.5)
        tf_s = card_s.text_frame
        tf_s.margin_left = tf_s.margin_right = Inches(0.12)
        tf_s.margin_top = Inches(0.15)
        p_l = tf_s.paragraphs[0]
        p_l.text = lbl; p_l.font.size = Pt(10); p_l.font.bold = True; p_l.font.color.rgb = RGBColor(148, 163, 184)
        p_l.alignment = PP_ALIGN.CENTER
        p_v = tf_s.add_paragraph()
        p_v.text = val; p_v.font.size = Pt(26); p_v.font.bold = True; p_v.font.color.rgb = col
        p_v.alignment = PP_ALIGN.CENTER; p_v.space_before = Pt(4)
        p_d = tf_s.add_paragraph()
        p_d.text = desc; p_d.font.size = Pt(9.5); p_d.font.color.rgb = RGBColor(203, 213, 225)
        p_d.alignment = PP_ALIGN.CENTER; p_d.space_before = Pt(4)

    s1.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Respected committee members, faculty, and examiners: Good morning. My name is Avinash Reddy Cheerapareddy, "
        "and today I am presenting our research: SafeAdapt: Robust Test-Time Adaptation via Optimal Transport and "
        "Conformal Risk Control in Digital Pathology. In clinical medicine, AI models perform well in the development lab, "
        "but the moment they are deployed to a new hospital, scanner variations and stain differences cause severe diagnostic failure. "
        "We introduce SafeAdapt, a fail-closed architecture that adapts models in real time during scanning without ground-truth labels "
        "and with certified safety.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Think of a self-driving car tuned exclusively for sunny California roads. When shipped to Chicago in a snowstorm, it encounters "
        "ice and snow. If it refuses to adapt, it crashes. But if it tries to adapt blindly without guardrails, it hallucinates lane markers "
        "and steers off a bridge. SafeAdapt provides an intelligent co-pilot and an automatic emergency brake for clinical AI.\n\n"
        "KEY HIGHLIGHTS: +14.2% AUROC gain, 0.0% collapse rate, and 1.8 ms GPU rollback."
    )

    # ==================== SLIDE 2: MOTIVATION & CLINICAL PROBLEM ====================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "The Multi-Center Domain Shift Dilemma in Pathology AI", "Clinical Motivation",
               "Diagnostic models trained on one hospital fail silently when deployed across multi-scanner clinical networks.")

    card_y = Inches(1.8)
    card_h = Inches(5.1)
    card_w = Inches(3.75)

    add_card(s2, Inches(0.8), card_y, card_w, card_h, "1. Hardware & Scanner Drift", [
        ("Multi-Vendor Scanners", "Pathology centers use Hamamatsu NanoZoomer, Philips Ultra Fast, and Leica Aperio digitizers."),
        ("Sensor Discrepancies", "Divergent optical lenses, CCD/CMOS sensors, and LED vs. halogen illumination spectra produce drastic chromatic variations."),
        ("Physical Artifacts", "Variations in tissue thickness (3-5 µm), micro-vibrations, and focus plane drifts alter cellular texture patterns across centers.")
    ], badge="Optical Heterogeneity")

    add_card(s2, Inches(4.8), card_y, card_w, card_h, "2. Chemical & Staining Shifts", [
        ("Reagent Protocols", "Different laboratories use variable concentrations of Hematoxylin (nuclei) and Eosin (cytoplasm/stroma)."),
        ("Pre-Analytical Factors", "Tissue fixation delays, alcohol dehydration variations, and paraffin temperature alter staining uptake."),
        ("Consequence", "Even identical biological specimens digitized at two different hospitals exhibit non-overlapping pixel value manifolds.")
    ], badge="Chemical Heterogeneity")

    add_card(s2, Inches(8.8), card_y, card_w, card_h, "3. Silent Diagnostic Failure", [
        ("The Performance Drop", "A ResNet-50 trained at Radboud achieving >0.98 AUROC plummets to 0.73 AUROC when deployed at CWZ or UMC Utrecht."),
        ("The Clinical Danger", "The model does NOT output low confidence—it fails silently with 99% confidence on misclassified tissue."),
        ("Urgent Requirement", "An autonomous, unsupervised adaptation method that operates in real-time during Whole Slide Image (WSI) streaming.")
    ], badge="Patient Safety Risk", accent_color=ACCENT_RED)

    s2.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "In digital pathology, cancer diagnosis begins by digitizing tissue slices into multi-gigabyte Whole-Slide Images. "
        "Across hospitals, hardware digitizers (Hamamatsu, Philips, Leica), lens calibrations, and chemical staining concentrations "
        "vary significantly. A model trained at Radboud University achieving over 0.98 AUROC plummets to 0.73 AUROC when tested at "
        "CWZ or UMC Utrecht. Crucially, the model does not express uncertainty—it fails silently with high confidence on misclassified tissue.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Imagine an expert human pathologist walking into a clinic where lights are tinted deep yellow and the microscope lenses are tinted blue. "
        "The human doctor pauses, adjusts their mental color balance, and continues diagnosing accurately. But a standard computer vision model "
        "trained on exact RGB values looks at the same image and thinks it is an alien planet—a 10% shift in dye concentration completely blinds the AI.\n\n"
        "KEY HIGHLIGHTS: AUROC drop from >0.98 to 0.73 across centers; 3-5 µm slice thickness variability."
    )

    # ==================== SLIDE 3: FAILURE OF CURRENT TTA ====================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Why Standard Test-Time Adaptation Collapses in Pathology", "Prior Art Breakdown",
               "Entropy minimization creates an unsupervised feedback loop that pulls models into degenerate single-class attractors.")

    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.8), Inches(2.4), "The Entropy Minimization Trap (Tent / EATA)", [
        ("Core Assumption", "Minimizing prediction entropy H(p) assumes that confident predictions are predominantly correct."),
        ("Histopathology Reality", "Severe stain drift causes false predictions to have moderate entropy; minimizing entropy actively reinforces false class assignments."),
        ("Feedback Loop", "Corrupted normalization parameters bias the next batch, leading to runaway confirmation bias.")
    ], badge="Flawed Premise", accent_color=ACCENT_RED)

    add_card(s3, Inches(0.8), Inches(4.4), Inches(5.8), Inches(2.5), "Catastrophic Attractor State Collapse", [
        ("Single-Class Attractor", "Batch Normalization running statistics drift until all features collapse into a single degenerate cluster."),
        ("Clinical Consequence", "The network predicts 100% of tissue patches as normal, driving cancer detection F1-score from 0.85 down to 0.05."),
        ("40% Collapse Rate", "On CAMELYON17 Center 4, standard Tent collapses on 40% of patient slides (e.g., Slide 29 drops to 0.51 AUROC).")
    ], badge="40% Collapse", accent_color=ACCENT_RED)

    add_image_box(s3, "paper/ieee/figs/fig_tent_collapse_mechanism.png", Inches(6.9), Inches(1.8), Inches(5.6), Inches(5.1),
                  "Empirical Collapse: Tent entropy optimization destroys latent feature margins across streaming WSI batches.")

    s3.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "In computer vision, the standard technique to adapt models during inference is Test-Time Adaptation, like Tent or EATA, "
        "which minimize Shannon prediction entropy. In digital pathology, this logic is fatal. Extreme stain shifts produce moderate "
        "entropy on false predictions; forcing entropy down reinforces the wrong class assignments. This creates an unsupervised "
        "feedback loop that pulls all feature embeddings into a single degenerate 'attractor state'. On CAMELYON17 Center 4, Tent "
        "collapsed on 40% of slides, driving cancer detection F1 down to 0.05.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Echo Chamber / Overconfident Medical Intern: Imagine a junior intern who misidentifies normal tissue as cancer. If the attending "
        "physician says, 'Whatever you do, be 100% decisive!', the intern assumes their mistake was correct and becomes aggressively "
        "overconfident. In the next room, they repeat the error with even higher confidence. Within two hours, the intern diagnoses every "
        "single patient on the floor with cancer. That is what entropy minimization does.\n\n"
        "KEY HIGHLIGHTS: 40.0% collapse rate on CAMELYON17; complete loss of diagnostic margins on Slide 29."
    )

    # ==================== SLIDE 4: THEORETICAL ROOT CAUSE ====================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Proposition 1: The Variance-Confidence Paradox", "Theoretical Foundation",
               "We prove that predictive entropy is asymptotic to epistemic uncertainty, blinding scalar confidence gating to catastrophic drift.")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Taylor Expansion Proof", [
        ("Weight Perturbation", "Consider parameter uncertainty delta_theta ~ N(0, Sigma_theta) around adapted weights."),
        ("Prediction Variance", "Var[p(y|x)] = grad_theta p(y|x)^T Sigma_theta grad_theta p(y|x) + O(||Sigma||^2)."),
        ("Vanishing Gradient", "As softmax probability p -> 1 or p -> 0, the derivative grad_theta p vanishes quadratically: grad p = p(1 - p) grad z."),
        ("Fatal Consequence", "Predictive entropy approaches zero exactly when epistemic weight variance reaches its global peak!")
    ], badge="Mathematical Proof", accent_color=NAVY)

    add_card(s4, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. Rank Inversion: rho = -0.9917", [
        ("Spearman Rank Test", "We measured the empirical rank correlation between epistemic variance (MC-Dropout) and Shannon entropy."),
        ("Inverse Equivalence", "Spearman rank coefficient rho = -0.9917 reveals asymptotic rank equivalence."),
        ("The Paradox", "When corrupted patches drive representations into collapsed feature space, the model outputs 99.8% confidence while being totally wrong."),
        ("Takeaway", "Gating updates based on scalar entropy or confidence is mathematically incapable of detecting collapse.")
    ], badge="Rank Inversion", accent_color=ACCENT_RED)

    add_card(s4, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Failure of Prior Regularizers", [
        ("EATA (Fisher Regularization)", "Adds an EWC penalty on parameter drift. However, static penalties cannot undo corrupted running statistics once a bad batch passes."),
        ("SAR (Sharpness-Aware)", "Monitors gradient norm spikes. But gradual stain shift creates small, consistent, low-norm gradients that silently bypass SAR's detector."),
        ("Our Solution", "We need a geometry-preserving transport metric and finite-sample conformal statistical risk bounds.")
    ], badge="Prior Art Gaps", accent_color=TEAL)

    s4.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "In Proposition 1, we prove the theoretical root cause: the Variance-Confidence Paradox. Through a second-order Taylor expansion, "
        "predictive variance depends directly on the logit gradient, which scales as p * (1 - p). As confidence approaches 1, this gradient "
        "vanishes quadratically! We measured the empirical rank correlation between epistemic weight variance (via MC-Dropout) and entropy: "
        "Spearman rho is -0.9917. The model outputs 99.8% confidence precisely when its epistemic uncertainty is at its maximum! That is why "
        "scalar confidence filters fail.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Dunning-Kruger Effect on an Exam: A student sits for a medical board exam without studying. Instead of leaving blanks, they mark "
        "option 'B' on all 200 questions with 100% confidence. Their predictive entropy is zero—they are totally decisive. But their actual knowledge "
        "is zero. If a system only checks how confident the student is, it gives them top marks for decisiveness while they fail the exam.\n\n"
        "KEY HIGHLIGHTS: Spearman rho = -0.9917; vanishing gradient formula grad p = p(1-p) grad z."
    )

    # ==================== SLIDE 5: SAFEADAPT ARCHITECTURE ====================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "SafeAdapt: A Deterministic, Fail-Closed TTA Architecture", "Proposed Framework",
               "SafeAdapt introduces geometry-preserving optimal transport, contextual artifact rejection, and finite-sample risk bounds.")

    add_image_box(s5, "paper/ieee/figs/fig_architecture.png", Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.1),
                  "SafeAdapt System Architecture: Dual-Stream Sinkhorn OT, Cross-Attention Memory Routing (CMR), and Conformal Rollback.")

    box_w = Inches(2.78)
    box_h = Inches(1.85)
    modules = [
        ("1. Dual-Stream Anchor", "Maintains frozen source anchor theta_0 in parallel with adaptive student theta_t to preserve diagnostic geometry.", TEAL),
        ("2. Sinkhorn Optimal Transport", "Aligns streaming patch feature marginals to class prototypes without label-dependent reinforcement.", TEAL),
        ("3. Memory Routing (CMR)", "Cross-attention memory routes OOD tissue debris and non-neoplastic mimics to a dedicated slack prototype.", NAVY),
        ("4. Conformal Risk Rollback", "Finite-sample risk monitor halts adaptation and restores theta_0 in 1.8 ms if parameter drift exceeds alpha.", ACCENT_GREEN)
    ]
    for idx, (m_title, m_desc, m_col) in enumerate(modules):
        bx = Inches(0.8) + idx * Inches(2.98)
        by = Inches(5.1)
        card_m = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, box_w, box_h)
        card_m.fill.solid(); card_m.fill.fore_color.rgb = CARD_BG
        card_m.line.color.rgb = m_col; card_m.line.width = Pt(1.5)
        tf_m = card_m.text_frame
        tf_m.margin_left = tf_m.margin_right = Inches(0.12)
        tf_m.margin_top = Inches(0.12)
        p_mt = tf_m.paragraphs[0]
        p_mt.text = m_title; p_mt.font.size = Pt(11); p_mt.font.bold = True; p_mt.font.color.rgb = m_col
        p_md = tf_m.add_paragraph()
        p_md.text = m_desc; p_md.font.size = Pt(9.5); p_md.font.color.rgb = SLATE_DARK
        p_md.space_before = Pt(4)

    s5.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Here is our proposed solution: SafeAdapt. SafeAdapt replaces uncontrolled entropy minimization with four interlocking safeguards: "
        "1) A Dual-Stream feature extractor pairing a frozen source anchor with an adaptive student; 2) Sinkhorn Optimal Transport to align feature "
        "marginals without label feedback; 3) Cross-Attention Memory Routing (CMR) to isolate staining artifacts into a slack channel; and 4) Finite-Sample "
        "Conformal Risk Control that triggers an instantaneous 1.8 ms GPU rollback if parameter drift breaches safety limits.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Commercial Airliner Dual-Pilot Cockpit: You have a senior Captain (the frozen anchor) who knows the flight envelope. You have a First Officer "
        "(the adaptive student) who flies the plane and adjusts for wind shear. A flight computer (CMR) ignores weather radar noise. And an envelope "
        "protection system steps in: if the co-pilot banks the plane beyond 30 degrees, the flight computer instantly takes control and levels the wings.\n\n"
        "KEY HIGHLIGHTS: 4-stage pipeline; Dual-Stream separation of Frozen Anchor vs. Adaptive Student."
    )

    # ==================== SLIDE 6: MECHANISM 1: SINKHORN OT ====================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Mechanism 1: Geometry-Preserving Sinkhorn Optimal Transport", "Technical Deep Dive",
               "Replaces unconstrained entropy minimization with exact marginal transport, mathematically preventing single-class collapse.")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Optimal Transport Polytope", [
        ("Wasserstein Discrepancy", "Measures the geometric distance between streaming batch features Z in R^(B x d) and anchor prototype dictionary C in R^(K x d)."),
        ("Transport Optimization", "min_{P in U(B, K)} <P, M> - epsilon H(P), where cost matrix M_ik = ||z_i - c_k||_2^2."),
        ("Marginal Constraints", "Row constraint P 1_K = (1/B) 1_B ensures every patch is mapped; column constraint P^T 1_B = (1/K) 1_K enforces balanced class capacity.")
    ], badge="Formulation", accent_color=TEAL)

    add_card(s6, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. Ultrafast Sinkhorn Scaling", [
        ("Iterative Dual Scaling", "Solved via alternating matrix scaling: P* = diag(u) exp(-M / epsilon) diag(v)."),
        ("GPU Acceleration", "Converges in under 5 matrix-vector multiplications, taking less than 0.4 ms per batch on modern GPU hardware."),
        ("Exact Marginals", "Unlike heuristic pseudo-labeling, Sinkhorn transport guarantees smooth, entropy-regularized assignments without label reinforcement.")
    ], badge="<0.4 ms GPU", accent_color=NAVY)

    add_card(s6, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Invariance to Class Imbalance", [
        ("The Histology Challenge", "A biopsy slide may contain 99% normal stroma or 90% tumor. Unconstrained models collapse to the majority class."),
        ("Marginal Safeguard", "Uniform marginal projection mathematically prohibits all probability mass from accumulating in a single prototype attractor."),
        ("Empirical Result", "Sinkhorn OT alone elevates multicenter AUROC from 0.8312 to 0.9124 while cutting collapse rate from 40% down to 8%.")
    ], badge="+8.1% AUROC", accent_color=ACCENT_GREEN)

    s6.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Mechanism 1 replaces entropy minimization with Sinkhorn Optimal Transport. We compute the Wasserstein distance between incoming "
        "patch embeddings and a prototype dictionary of tissue classes. By enforcing uniform column marginal constraints, the optimal transport "
        "plan must allocate features across classes in balanced proportions. Even if a slide has 99% normal tissue, the uniform marginal "
        "constraint makes it mathematically impossible for the network to collapse into a single class attractor. It converges in under 5 "
        "matrix-vector multiplications in less than 0.4 ms on GPU.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Fair Logistics Dispatcher: If a regional dispatcher has 100 trucks and 5 warehouses, an uncontrolled system might route all 100 trucks "
        "to one warehouse because it's slightly closer, causing massive gridlock. Sinkhorn Optimal Transport enforces strict capacity constraints: "
        "Warehouse 1 takes 20 trucks, Warehouse 2 takes 20, guaranteeing efficient matching while preventing systemic overload.\n\n"
        "KEY HIGHLIGHTS: <0.4 ms GPU execution; +8.1% AUROC gain from OT alignment alone."
    )

    # ==================== SLIDE 7: MECHANISM 2: CMR ====================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Mechanism 2: Cross-Attention Memory-Augmented Routing (CMR)", "Technical Deep Dive",
               "Dynamic memory routing identifies non-neoplastic tissue artifacts and isolates them from diagnostic prototypes.")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Cross-Attention Query-Key", [
        ("Episodic Memory Bank", "Maintains high-confidence source representations in a compact key-value memory bank M_mem = (K_mem, V_mem)."),
        ("Attention Alignment", "Attention weights A = Softmax(Q K_mem^T / sqrt(d)) compute contextual similarity between streaming patches and trusted source knowledge."),
        ("Feature Augmentation", "Streaming features are enriched with clean source context before classification, filtering out scanner-specific high-frequency noise.")
    ], badge="Memory Bank", accent_color=TEAL)

    add_card(s7, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. Out-of-Distribution Slack Channel", [
        ("Histological Realities", "Routine slides contain surgical marker ink, air bubbles, folded tissue, and cautery burn artifacts."),
        ("The Slack Prototype", "CMR routes anomalous tokens with low memory affinity into an explicit 'slack' prototype channel."),
        ("Distortion Shield", "By absorbing OOD feature mass into the slack channel, authentic diagnostic prototypes are shielded from distortive gradient pull.")
    ], badge="OOD Shielding", accent_color=NAVY)

    add_card(s7, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Quantitative Diagnostic Impact", [
        ("False Alarm Suppression", "Reduces false positive patch activations across necrotic and inflamed resection margins by 38.4%."),
        ("Performance Contribution", "Integrating CMR on top of Sinkhorn OT elevates AUROC from 0.9124 to 0.9348 and reduces remaining collapse risk to 4%."),
        ("Clinical Utility", "Ensures pathologists are not distracted by false-positive heatmaps on benign inflammatory mimics.")
    ], badge="-38.4% False Positives", accent_color=ACCENT_GREEN)

    s7.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Mechanism 2 is Cross-Attention Memory-Augmented Routing (CMR). Clinical slides contain pen marks (surgical ink), tissue folds, "
        "air bubbles under the coverslip, and cautery burns. A standard model confuses dark ink with aggressive tumor pigment, corrupting its updates. "
        "CMR maintains an episodic memory bank of validated source features. When streaming patches have near-zero cosine similarity with real tissue, "
        "CMR routes them into an explicit slack prototype channel, preventing distorted gradients from polluting diagnostic prototypes.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Security Bouncer with a VIP Coatroom: At an exclusive medical conference, only badge-holding oncologists are allowed to vote. "
        "If someone walks in off the street in a costume (surgical ink or an air bubble), the bouncer courteously steers them into the coatroom lounge "
        "(the slack channel). They can stay there safely, but they cannot vote on clinical decisions.\n\n"
        "KEY HIGHLIGHTS: -38.4% false-positive patch activations; +2.2% AUROC boost."
    )

    # ==================== SLIDE 8: MECHANISM 3: CONFORMAL RISK & ROLLBACK ====================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Mechanism 3: Finite-Sample Conformal Risk Control & Rollback", "Technical Deep Dive & Theorem 1",
               "A distribution-free fail-closed mechanism guarantees that clinical risk is strictly bounded at significance level alpha.")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Theorem 1: Conformal Bound", [
        ("Non-Conformity Scoring", "Evaluates non-conformity s_i = 1 - f_theta(x_i)_y on streaming calibration batches."),
        ("Finite-Sample Risk Bound", "P( R(theta_t) <= alpha ) >= 1 - delta, where empirical risk R is bounded with high probability."),
        ("Distribution-Free", "The guarantee holds strictly without making parametric assumptions on target scanner distributions."),
        ("Clinical Significance", "Provides a mathematical certificate that test-time adaptation cannot violate the hospital's safety envelope.")
    ], badge="Theorem 1", accent_color=NAVY)

    add_card(s8, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. Dynamic Drift Monitoring", [
        ("Online Risk Tracking", "Monitors cumulative non-conformity p-values across sliding temporal window of patch batches."),
        ("Interlock Trigger", "If empirical risk breaches threshold alpha = 0.05, the adaptation controller trips an emergency interlock."),
        ("Immediate Freeze", "Halts gradient backpropagation immediately, preventing corrupt batches from altering model normalization parameters.")
    ], badge="Safety Interlock", accent_color=ACCENT_RED)

    add_card(s8, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Sub-2ms Deterministic Rollback", [
        ("In-Memory Parameter Cache", "Maintains an atomic FIFO cache of pristine anchor parameters theta_0 in GPU VRAM."),
        ("1.8 ms State Restore", "Upon interlock trigger, restores theta_0 in exactly 1.8 ms—completely sub-frame and non-blocking."),
        ("Zero Collapse Guarantee", "Eliminates 100% of catastrophic collapse modes, dropping collapse rate from 40% to exactly 0.0% across all seeds.")
    ], badge="1.8 ms Rollback", accent_color=ACCENT_GREEN)

    s8.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Mechanism 3 provides statistical safety via Finite-Sample Conformal Risk Control (Theorem 1). We compute non-conformity scores on streaming "
        "batches to bound clinical risk: P(R(theta_t) <= alpha) >= 1 - delta. This holds strictly without parametric assumptions on target scanners. "
        "We track the moving variance of conformal p-values. If empirical risk breaches alpha = 0.05, the adaptation controller trips a safety "
        "interlock, restoring anchor parameters theta_0 in 1.8 milliseconds from GPU memory without stalling the WSI queue.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "High-Speed Factory Circuit Breaker: In a bottling plant, if an infrared sensor detects bottle caps deviating by 0.5 mm, an ultra-fast "
        "arm diverts the defective bottle in 2 ms while the main conveyor belt keeps running at full speed. Our conformal rollback detects "
        "parameter drift before diagnostic harm occurs and resets weights in 1.8 milliseconds.\n\n"
        "KEY HIGHLIGHTS: Theorem 1 distribution-free bound; 1.8 ms sub-frame rollback."
    )

    # ==================== SLIDE 9: EXPERIMENTAL SETUP ====================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Multicenter Benchmark Datasets & 8 Competing Baselines", "Experimental Rigor",
               "Evaluated across 5 clinical centers, 2 international challenges, and 8 state-of-the-art adaptation paradigms.")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Benchmark Datasets", [
        ("CAMELYON17 Consortium", "50 whole-slide images across 5 distinct university medical centers: Radboud, UMC Utrecht, Rijnstate, CWZ, and UMC Groningen."),
        ("Extreme Domain Shifts", "Features severe cross-scanner color drift, variable slice thickness, and diverse clinical preparation protocols."),
        ("MIDOG 2021 Challenge", "Mitotic figure detection across 4 scanner types (Hamamatsu XR, S360, Aperio ScanScope, 3DHISTECH) and multi-species tissue (human breast, canine mast cell).")
    ], badge="Multi-Center WSI", accent_color=TEAL)

    add_card(s9, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. 8 Competing Baselines", [
        ("Unadapted Baseline", "Frozen Source Model (no adaptation during deployment)."),
        ("Classical Vision TTA", "Tent (ICLR '21: entropy min), EATA (ICML '22: Fisher regularized), SAR (ICLR '23: sharpness-aware)."),
        ("Continual TTA", "CoTTA (CVPR '22: mean-teacher), RoTTA (CVPR '23: category memory bank)."),
        ("Test-Time Augmentation", "MEMO (NeurIPS '22: 16-view marginal entropy averaging)."),
        ("Proposed", "SafeAdapt (Ours: OT + CMR + Conformal Gating).")
    ], badge="8 SOTA Methods", accent_color=NAVY)

    add_card(s9, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Evaluation Protocol", [
        ("Multi-Seed Benchmarking", "5 random seeds per experiment to rigorously capture non-deterministic collapse behaviors."),
        ("Clinical Metrics", "Whole-slide Macro AUROC, Expected Calibration Error (ECE), and Catastrophic Collapse Rate (% slides with Delta AUROC < -0.10)."),
        ("Hardware Metrics", "Wall-clock inference latency (ms/patch), peak GPU memory (MB), and state rollback time (ms).")
    ], badge="Rigorous Protocol", accent_color=ACCENT_GREEN)

    s9.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "To rigorously evaluate SafeAdapt, we benchmarked across two premier multicenter datasets: CAMELYON17 (50 whole slides across 5 university "
        "medical centers in the Netherlands: Radboud, Utrecht, Rijnstate, CWZ, Groningen) and MIDOG 2021 (cross-scanner mitosis detection across "
        "4 digitizers and multi-species tissue). We compared against 8 baselines spanning classical TTA (Tent, EATA, SAR), continual TTA (CoTTA, RoTTA), "
        "and augmentation (MEMO), across 5 random seeds to capture non-deterministic collapse.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Multi-Center Drug Clinical Trials: If a pharmaceutical company tests a drug only on 20 healthy volunteers in their own lab, the FDA rejects it. "
        "A real clinical trial must test across 5 independent hospitals with diverse patient populations. CAMELYON17 represents a true multi-center "
        "trial across 5 hospital systems to ensure our AI works across any hospital scanner.\n\n"
        "KEY HIGHLIGHTS: 5 independent university medical centers; 8 competing adaptation baselines."
    )

    # ==================== SLIDE 10: MAIN RESULTS: CAMELYON17 ====================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Superior Diagnostic Performance Across Hospital Cohorts", "Empirical Benchmarks: CAMELYON17",
               "SafeAdapt achieves new state-of-the-art: 0.9495 AUROC, lowest ECE (0.0418), and 0.0% catastrophic collapse.")

    t_headers = ["Method", "Venue", "AUROC", "ECE", "Collapse Rate"]
    t_rows = [
        ["Source (Frozen)", "Baseline", "0.8312", "0.1642", "0.0%"],
        ["Tent", "ICLR '21", "0.7645", "0.2810", "40.0%"],
        ["EATA", "ICML '22", "0.8521", "0.1420", "20.0%"],
        ["SAR", "ICLR '23", "0.8640", "0.1285", "15.0%"],
        ["CoTTA", "CVPR '22", "0.8210", "0.1890", "25.0%"],
        ["RoTTA", "CVPR '23", "0.8715", "0.1150", "10.0%"],
        ["MEMO", "NeurIPS '22", "0.8840", "0.0980", "5.0%"],
        ["SafeAdapt (Ours)", "Proposed", "0.9495", "0.0418", "0.0%"]
    ]
    col_w = [Inches(1.8), Inches(1.1), Inches(1.0), Inches(1.0), Inches(1.2)]
    add_styled_table(s10, Inches(0.8), Inches(1.8), Inches(6.1), Inches(4.3),
                     t_headers, t_rows, col_w, highlight_row_idx=7, alert_row_idx=1)

    c_chip = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.25), Inches(6.1), Inches(0.65))
    c_chip.fill.solid(); c_chip.fill.fore_color.rgb = TEAL_BG
    c_chip.line.color.rgb = TEAL; c_chip.line.width = Pt(1.0)
    tf_chip = c_chip.text_frame
    tf_chip.margin_left = Inches(0.12); tf_chip.margin_top = Inches(0.08)
    p_ch = tf_chip.paragraphs[0]
    p_ch.text = "EMPIRICAL BREAKTHROUGH: +14.2% AUROC over Source, +9.8% over best baseline (MEMO), and 0.0% collapse."
    p_ch.font.size = Pt(9.5); p_ch.font.bold = True; p_ch.font.color.rgb = TEAL_DARK

    add_image_box(s10, "paper/ieee/figs/fig_v2_benchmark_auroc_ece.png", Inches(7.1), Inches(1.8), Inches(5.4), Inches(5.1),
                  "AUROC vs. Expected Calibration Error (ECE): SafeAdapt occupies the ideal top-left Pareto frontier.")

    s10.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Here are the headline results on CAMELYON17 (Table I). Unadapted ResNet achieves 0.8312 AUROC. Standard Tent collapses "
        "to 0.7645 AUROC with 40.0% collapse rate and 0.2810 calibration error. Baselines like EATA, SAR, and RoTTA reach 0.85-0.87 AUROC "
        "but still suffer 10-20% collapse. SafeAdapt establishes a new state-of-the-art: 0.9495 AUROC, the lowest Expected Calibration Error "
        "of 0.0418, and exactly 0.0% catastrophic collapse. In the Pareto plot, SafeAdapt dominates the top-left ideal frontier.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The 4 out of 10 Failure Rate: Imagine a car dealer saying, 'This car drives wonderfully on 6 out of 10 trips, but on 4 out of 10 "
        "trips, the steering locks up and you crash.' No hospital would allow that algorithm in their lab. SafeAdapt is the only framework "
        "that achieves a 0.0% failure rate across all 50 patient whole slides while delivering a +14.2% diagnostic boost.\n\n"
        "KEY HIGHLIGHTS: 0.9495 AUROC (SOTA); 0.0% Collapse Rate vs. 40.0% in Tent; 0.0418 ECE."
    )

    # ==================== SLIDE 11: SLIDE 29 CASE STUDY ====================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Resolving Slide 29: From Diagnostic Collapse to 0.9324 AUROC", "Clinical Case Study & Attractor Reversal",
               "SafeAdapt detects early parameter drift on Slide 29, triggering rollback and recovering 0.9324 AUROC.")

    add_image_box(s11, "paper/ieee/figs/fig_v2_slide29_attractor_reversal.png", Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.0),
                  "UMAP Latent Space on Patient Slide 29: (A) Source baseline; (B) Tent collapse into single point; (C) SafeAdapt recovery.")

    add_card(s11, Inches(0.8), Inches(5.0), Inches(3.75), Inches(1.9), "1. The Slide 29 Failure", [
        ("The Challenge", "Slide 29 features extreme hematoxylin stain drift and sparse micrometastasis (<2% tumor burden)."),
        ("Tent Degradation", "Standard entropy minimization collapsed AUROC from 0.73 down to 0.5120 (pure random chance).")
    ], badge="0.51 AUROC Collapse", accent_color=ACCENT_RED)

    add_card(s11, Inches(4.8), Inches(5.0), Inches(3.75), Inches(1.9), "2. Attractor Reversal Mechanism", [
        ("Conformal Detection", "Conformal risk monitor detected non-conformity drift at batch t=14."),
        ("Instant Recovery", "1.8 ms rollback restored anchor theta_0, allowing Sinkhorn OT to align marginals safely.")
    ], badge="Rollback at t=14", accent_color=TEAL)

    add_card(s11, Inches(8.8), Inches(5.0), Inches(3.75), Inches(1.9), "3. Quantitative Proof", [
        ("AUROC Recovery", "Recovers AUROC to 0.9324—a massive +0.4204 gain over collapsed Tent."),
        ("Margin Preservation", "UMAP shows crisp class separation between normal stroma and metastatic carcinoma.")
    ], badge="0.9324 AUROC", accent_color=ACCENT_GREEN)

    s11.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Let us examine Patient Slide 29—a slide with dark hematoxylin stain drift and sparse micrometastasis (<2% tumor). In Panel B, "
        "standard Tent collapsed into an attractor state where all features mapped to a single point, dropping AUROC to 0.5120 (coin-flip "
        "chance) and missing metastatic cancer. In Panel C, SafeAdapt's conformal monitor detected drift at batch 14, executed a 1.8 ms "
        "rollback, and used Optimal Transport to align features. Normal stroma and metastatic carcinoma clusters are cleanly separated, "
        "recovering 0.9324 AUROC.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Needle in a Haystack: A patient has tiny breast cancer spread in a massive lymph node. Because 98% of the slide is healthy, "
        "an uncontrolled AI takes the easy shortcut: it decides everything is healthy to maximize its confidence score, telling the patient "
        "they are cancer-free when they are not! SafeAdapt prevents the AI from taking that lazy, dangerous shortcut.\n\n"
        "KEY HIGHLIGHTS: 0.5120 AUROC (Tent collapse) vs. 0.9324 AUROC (SafeAdapt recovery); rollback at batch 14."
    )

    # ==================== SLIDE 12: MITOSIS DETECTION (MIDOG) ====================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Cross-Scanner & Cross-Species Mitosis Detection (MIDOG)", "Empirical Benchmarks: Detection",
               "SafeAdapt boosts mitotic figure F1-score to 0.5172, suppressing false positives across unseen scanners and species.")

    add_image_box(s12, "paper/ieee/figs/fig_v2_cmr_mitosis_detection.png", Inches(0.8), Inches(1.8), Inches(11.733), Inches(2.9),
                  "MIDOG 2021 Mitosis Detection: PR Curves and qualitative crops showing false-positive suppression on unseen scanners.")

    m_headers = ["Model", "Precision", "Recall", "F1-Score"]
    m_rows = [
        ["Source Baseline", "0.3812", "0.4485", "0.4120"],
        ["Tent (ICLR '21)", "0.2710", "0.4035", "0.3245"],
        ["EATA (ICML '22)", "0.4120", "0.4610", "0.4350"],
        ["SAR (ICLR '23)", "0.4280", "0.4700", "0.4480"],
        ["SafeAdapt (Ours)", "0.5284", "0.5065", "0.5172"]
    ]
    col_wm = [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2)]
    add_styled_table(s12, Inches(0.8), Inches(4.85), Inches(5.4), Inches(2.05),
                     m_headers, m_rows, col_wm, highlight_row_idx=4, alert_row_idx=1)

    add_card(s12, Inches(6.5), Inches(4.85), Inches(6.033), Inches(2.05), "CMR Slack Routing Eliminates Debris", [
        ("The Challenge", "Pyknotic apoptotic nuclei and hyperchromatic debris mimic mitotic figures on unseen scanners."),
        ("False-Alarm Explosion", "Tent experiences severe FP explosion (Precision drops from 0.38 down to 0.27)."),
        ("SafeAdapt Superiority", "CMR slack channel absorbs hyperchromatic debris, boosting Precision to 0.5284 (+38.7% over Source) and achieving SOTA F1-score of 0.5172.")
    ], badge="+25.5% F1 Gain", accent_color=ACCENT_GREEN)

    s12.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "In MIDOG 2021, models must detect mitotic figures across 4 scanner types and multiple species. The mitotic count dictates whether "
        "a patient needs chemotherapy. Under stain drift, dying apoptotic cells and debris look like dividing cells. Under Tent, false alarms "
        "exploded, driving precision down to 0.2710 and F1 down to 0.3245. SafeAdapt's CMR slack channel routed hyperchromatic debris away, "
        "boosting precision to 0.5284 (+38.7% over source) and achieving the top F1-score of 0.5172 (+25.5% relative gain).\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Distinguishing a Firefly from a Campfire Ember: A forest ranger wants to count fireflies at night. Under a full moon, glowing campfire "
        "sparks look just like fireflies. An untrained AI counts every spark, inflating the count tenfold. CMR acts like a spectral filter "
        "that filters out the sparks, ensuring the oncologist gets an honest, uncorrupted count of dividing cancer cells.\n\n"
        "KEY HIGHLIGHTS: 0.5172 F1-score; 0.5284 Precision (+38.7% over unadapted baseline)."
    )

    # ==================== SLIDE 13: CLINICAL BOUNDARY TESTING ====================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Stress-Testing Across Four Extreme Clinical Boundary Regimes", "Clinical Stress-Testing & Robustness",
               "SafeAdapt maintains rock-solid stability in non-ideal real-world hospital workflows where traditional TTA fails.")

    b_w = Inches(5.7)
    b_h = Inches(2.4)

    add_card(s13, Inches(0.8), Inches(1.8), b_w, b_h, "Regime 1: Single-Patch Streaming (Batch Size B = 1)", [
        ("Clinical Reality", "Real-time digital microscopes stream individual tissue patches sequentially as the objective moves."),
        ("Standard BN Failure", "Standard Batch Normalization cannot compute batch statistics when B=1, leading to NaN gradients."),
        ("SafeAdapt Solution", "Employs an asynchronous FIFO memory buffer (M=32) that preserves 0.9412 AUROC without latency penalty.")
    ], badge="B = 1 Streaming", accent_color=TEAL)

    add_card(s13, Inches(6.8), Inches(1.8), b_w, b_h, "Regime 2: Purely Benign Slides (0.0% Tumor Burden)", [
        ("Clinical Reality", "Over 80% of routine clinical lymph node dissections are purely benign (0.0% tumor cells)."),
        ("Entropy Min Danger", "Entropy minimization artificially forces positive predictions to balance class distributions, generating false tumors."),
        ("SafeAdapt Solution", "Exhibits zero false-positive drift on 10 purely benign validation slides, preserving 99.4% specificity.")
    ], badge="0.0% Tumor Slides", accent_color=ACCENT_GREEN)

    add_card(s13, Inches(0.8), Inches(4.45), b_w, b_h, "Regime 3: Weakly-Supervised Attention-MIL Integration", [
        ("Slide-Level Diagnosis", "Evaluated end-to-end with Whole-Slide Attention-MIL aggregation across 10,000+ patches per slide."),
        ("Diagnostic Triage", "SafeAdapt adapted patch representations improve slide-level patient triage accuracy by +6.8% over unadapted features."),
        ("Seamless Pipeline", "Functions as a drop-in patch feature adapter upstream of any clinical MIL aggregation architecture.")
    ], badge="MIL Integration", accent_color=NAVY)

    add_card(s13, Inches(6.8), Inches(4.45), b_w, b_h, "Regime 4: Continual Multi-Patient Drift Testing", [
        ("The HIPAA Concern", "Processing consecutive slides without resetting model state allows Patient A's features to alter Patient B's diagnosis."),
        ("SafeAdapt Policy", "Enforces strict deterministic parameter rollback to theta_0 at the conclusion of each whole-slide scan."),
        ("Zero Contamination", "Guarantees zero cross-patient statistical contamination and eliminates inter-patient error accumulation.")
    ], badge="Patient Isolation", accent_color=TEAL)

    s13.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "We stress-tested SafeAdapt across four extreme clinical boundary regimes: 1) Single-patch streaming (B=1): our FIFO buffer "
        "maintains 0.9412 AUROC without batch collapse; 2) Purely benign slides (0.0% tumor): in routine screening, >80% of lymph nodes are benign; "
        "SafeAdapt exhibited zero false-positive drift, preserving 99.4% specificity; 3) Whole-Slide Attention-MIL: boosted patient triage "
        "accuracy by +6.8%; 4) Continual Cross-Patient Testing: deterministic resets completely eliminate error accumulation across slides.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Routine Clean Checkup: 8 out of 10 people who get a screening biopsy are completely cancer-free. If an AI expects to see cancer, "
        "it will hallucinate cancer on clean tissue just to satisfy its mathematical expectation. SafeAdapt recognizes when a slide is 100% clean "
        "and stays quiet, preserving 99.4% specificity.\n\n"
        "KEY HIGHLIGHTS: 0.9412 AUROC at B=1 streaming; 99.4% specificity on 0.0% tumor burden slides."
    )

    # ==================== SLIDE 14: HARDWARE & EDGE LATENCY ====================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Hardware Benchmarks: Clinical PACS Integration & Real-Time Latency", "Hardware Feasibility",
               "SafeAdapt delivers 312 patches/sec at 1.8 ms rollback latency, easily fitting standard hospital PACS infrastructure.")

    h_headers = ["Method", "Latency (ms)", "Throughput (p/s)", "VRAM (MB)", "Rollback (ms)"]
    h_rows = [
        ["Source (Frozen)", "2.8 ms", "357 p/s", "1,420 MB", "N/A"],
        ["Tent (ICLR '21)", "3.1 ms", "322 p/s", "1,540 MB", "N/A"],
        ["CoTTA (CVPR '22)", "8.4 ms", "119 p/s", "3,110 MB", "N/A"],
        ["MEMO (16 views)", "48.6 ms", "20 p/s", "4,890 MB", "N/A (17x slower!)"],
        ["SafeAdapt (Ours)", "3.2 ms", "312 p/s", "1,680 MB", "1.8 ms"]
    ]
    col_wh = [Inches(1.7), Inches(1.0), Inches(1.2), Inches(1.0), Inches(1.1)]
    add_styled_table(s14, Inches(0.8), Inches(1.8), Inches(6.0), Inches(2.8),
                     h_headers, h_rows, col_wh, highlight_row_idx=4, alert_row_idx=3)

    add_card(s14, Inches(0.8), Inches(4.75), Inches(6.0), Inches(2.15), "Real-Time Clinical PACS Compatibility", [
        ("Negligible Overhead", "Only +0.4 ms overhead per patch over frozen baseline (3.2 ms vs 2.8 ms)."),
        ("Ultra-Low Memory Footprint", "Requires only 1,680 MB VRAM, fitting into standard 8GB hospital edge workstations."),
        ("Instant Rollback", "Rollback executes in 1.8 ms—recovering parameters in under 1/500th of a second without pipeline stall.")
    ], badge="312 patches/sec", accent_color=ACCENT_GREEN)

    add_image_box(s14, "paper/ieee/figs/fig_v2_gpu_training_convergence.png", Inches(7.0), Inches(1.8), Inches(5.5), Inches(5.1),
                  "GPU Inference Latency & Parameter Convergence: SafeAdapt converges in <= 5 steps with minimal latency overhead.")

    s14.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "A typical Whole-Slide Image contains over 10,000 patches. Table IV shows hardware efficiency. Multi-view augmentation (MEMO) "
        "requires 48.6 ms per patch and 4.8 GB VRAM—17 times slower than baseline, taking 10 minutes per slide. SafeAdapt requires only "
        "3.2 ms per patch (312 patches/sec), adding just 0.4 ms overhead over the frozen model. It processes a slide in under 32 seconds, "
        "executes rollback in 1.8 ms, and uses only 1,680 MB of VRAM, fitting into standard hospital workstations.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Doctor's Spinning Hourglass: A pathologist reviews 40 slides a day. If an AI takes 10 minutes per slide, the doctor stares at a "
        "spinning hourglass all morning and unplugs the software. SafeAdapt finishes analyzing the whole slide in 30 seconds while the pathologist "
        "reviews the patient's chart. It is fast, invisible, and seamless.\n\n"
        "KEY HIGHLIGHTS: 3.2 ms/patch (312 p/s); 1.8 ms rollback; 1,680 MB VRAM footprint."
    )

    # ==================== SLIDE 15: CLINICAL GOVERNANCE & DICOM ====================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15)
    add_header(s15, "Regulatory Compliance, Patient Privacy & DICOM Standards", "Clinical Governance",
               "Engineered for hospital deployment: HIPAA/GDPR compliance with zero cross-patient data leakage.")

    add_card(s15, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Deterministic Episodic Resets", [
        ("The Clinical Dilemma", "In a hospital queue, slide processing order is random. Continual learning algorithms (CoTTA, RoTTA) allow Patient A's tissue to alter Patient B's diagnosis."),
        ("Regulatory Hazard", "Violates FDA non-leakage standards and EU AI Act reproducibility requirements."),
        ("SafeAdapt Guarantee", "Enforces mathematical parameter resets to theta_0 upon slide completion, guaranteeing deterministic, order-independent diagnosis.")
    ], badge="HIPAA / GDPR", accent_color=TEAL)

    add_card(s15, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. Native DICOM Integration", [
        ("Standard Header Parsing", "Directly interfaces with DICOM Whole Slide Imaging metadata standards."),
        ("Automatic Triggering", "DICOM Accession Number (0008,0050) and Study Instance UID (0020,000D) automatically trigger FIFO buffer purges and parameter resets."),
        ("Audit Logging", "Outputs cryptographically hashed adaptation logs for every slide, fulfilling hospital QA requirements.")
    ], badge="DICOM WSI Tags", accent_color=NAVY)

    add_card(s15, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Fail-Closed Safety Philosophy", [
        ("Medical Device Principles", "Engineered in accordance with IEC 62304 medical device software lifecycle standards."),
        ("Safe Fallback State", "If an extreme out-of-distribution artifact occurs, the system defaults to the certified source model rather than risking erratic adapted outputs."),
        ("Pathologist Trust", "Provides pathologists with transparent safety indicators on every generated diagnostic report.")
    ], badge="IEC 62304 Aligned", accent_color=ACCENT_GREEN)

    s15.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "In Section X, we address clinical governance. In hospital queues, patient slide order is arbitrary. Continual methods like CoTTA or "
        "RoTTA keep memory banks active across slides, allowing Patient A's tissue statistics to alter Patient B's diagnosis! This violates "
        "HIPAA and GDPR patient non-leakage principles. SafeAdapt parses DICOM WSI tags—Accession Number (0008,0050) and Study UID (0020,000D)—triggering "
        "automatic parameter resets to theta_0. Following IEC 62304 standards, our architecture is fail-closed: it defaults to the certified source model upon anomaly.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "Sanitizing the Scalpel Between Patients: When a doctor examines Patient A, they don't use that same unwashed instrument on Patient B. "
        "They sanitize it thoroughly. Continual learning algorithms are like using an unsterilized tool across a patient queue—Patient A's tissue "
        "contaminates Patient B. SafeAdapt sterilizes the model between every patient.\n\n"
        "KEY HIGHLIGHTS: DICOM tags (0008,0050) and (0020,000D); IEC 62304 fail-closed safety compliance."
    )

    # ==================== SLIDE 16: COMPREHENSIVE ABLATION STUDY ====================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16)
    add_header(s16, "Dissecting SafeAdapt: Component-by-Component Impact", "Ablation & Sensitivity Analysis",
               "Every architectural safeguard provides measurable value: removing any module degrades accuracy or re-introduces collapse.")

    ab_headers = ["Configuration", "AUROC", "ECE", "Collapse Rate", "Contribution"]
    ab_rows = [
        ["Frozen Source Model", "0.8312", "0.1642", "0.0%", "Reference Baseline"],
        ["+ Standard Tent (Entropy Min)", "0.7645", "0.2810", "40.0%", "Collapse Failure (-0.0667)"],
        ["+ Sinkhorn Optimal Transport", "0.9124", "0.0890", "8.0%", "Primary Alignment (+0.0812)"],
        ["+ Cross-Attention Memory (CMR)", "0.9348", "0.0610", "4.0%", "Artifact Rejection (+0.0224)"],
        ["+ Full SafeAdapt (with Conformal)", "0.9495", "0.0418", "0.0%", "Complete Safety (+0.0147)"]
    ]
    col_wab = [Inches(2.2), Inches(0.9), Inches(0.9), Inches(1.1), Inches(1.8)]
    add_styled_table(s16, Inches(0.8), Inches(1.8), Inches(6.9), Inches(3.2),
                     ab_headers, ab_rows, col_wab, highlight_row_idx=4, alert_row_idx=1)

    ab_chip = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.15), Inches(6.9), Inches(1.75))
    ab_chip.fill.solid(); ab_chip.fill.fore_color.rgb = CARD_BG
    ab_chip.line.color.rgb = TEAL; ab_chip.line.width = Pt(1.5)
    tf_ab = ab_chip.text_frame
    tf_ab.margin_left = tf_ab.margin_right = Inches(0.15); tf_ab.margin_top = Inches(0.12)
    p_abh = tf_ab.paragraphs[0]
    p_abh.text = "KEY ABLATION INSIGHTS:"; p_abh.font.size = Pt(11); p_abh.font.bold = True; p_abh.font.color.rgb = TEAL_DARK
    for lead, txt in [
        ("Sinkhorn OT is the primary engine", "Drives a +0.0812 AUROC jump by preserving semantic category geometry."),
        ("CMR provides artifact shielding", "Adds +0.0224 AUROC and halves remaining collapse risk by absorbing debris."),
        ("Conformal Rollback provides total safety", "Provides the final +0.0147 gain and drives catastrophic collapse to exactly 0.0%.")
    ]:
        p_ab = tf_ab.add_paragraph()
        run1 = p_ab.add_run(); run1.text = f"• {lead}: "; run1.font.bold = True; run1.font.size = Pt(9.5); run1.font.color.rgb = NAVY
        run2 = p_ab.add_run(); run2.text = txt; run2.font.size = Pt(9.5); run2.font.color.rgb = SLATE_DARK

    add_card(s16, Inches(7.9), Inches(1.8), Inches(4.633), Inches(2.4), "Sensitivity to Epsilon & Tau", [
        ("Sinkhorn Regularizer epsilon", "Stable across 3 orders of magnitude (0.01 to 0.5); optimal at epsilon = 0.05."),
        ("Temperature tau", "Maintains high AUROC (>0.94) across tau in [0.05, 0.20], demonstrating hyperparameter robustness.")
    ], badge="Robust Hyperparameters", accent_color=NAVY)

    add_card(s16, Inches(7.9), Inches(4.4), Inches(4.633), Inches(2.5), "Sensitivity to Conformal Alpha", [
        ("Safety Bound alpha = 0.05", "Strikes optimal balance between positive adaptation and proactive rollback."),
        ("Overly Strict (alpha = 0.01)", "Triggers premature rollbacks, preserving safety but limiting adaptation gain to 0.8920."),
        ("Overly Permissive (alpha = 0.10)", "Allows 2% edge-case drift; confirms alpha = 0.05 as the ideal clinical setting.")
    ], badge="Calibrated Alpha", accent_color=TEAL)

    s16.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "Table V shows our comprehensive ablation study: 1) Frozen Source starts at 0.8312 AUROC; 2) Tent drops to 0.7645 AUROC with 40% collapse; "
        "3) Sinkhorn Optimal Transport provides the primary engine, boosting AUROC by +8.1% to 0.9124 and slashing collapse to 8%; 4) CMR adds +2.2% "
        "to reach 0.9348 AUROC by isolating artifacts; 5) Conformal Rollback provides total safety, achieving 0.9495 AUROC and driving collapse to "
        "exactly 0.0%. Sensitivity analysis confirms stability across broad hyperparameters.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Three-Legged Stool: SafeAdapt is like a three-legged stool: Optimal Transport is the strong seat holding the weight (+8.1%); "
        "CMR is the brace preventing wobbles (+2.2%); and Conformal Rollback is the non-slip rubber grip on the floor that guarantees zero falls "
        "(0.0%). Remove any leg, and the stool falls.\n\n"
        "KEY HIGHLIGHTS: Stepwise AUROC progression: 0.8312 -> 0.9124 -> 0.9348 -> 0.9495; collapse: 40% -> 0%."
    )

    # ==================== SLIDE 17: COMPARISON WITH CONTINUAL TTA ====================
    s17 = prs.slides.add_slide(blank_layout)
    set_slide_background(s17)
    add_header(s17, "SafeAdapt vs. Recent Continual Paradigms (SAR, CoTTA, RoTTA)", "Advanced Benchmarking",
               "Continual methods suffer from error accumulation across slides; SafeAdapt guarantees deterministic safety and superior accuracy.")

    add_card(s17, Inches(0.8), Inches(1.8), Inches(3.75), Inches(5.1), "1. Error Accumulation in Continual TTA", [
        ("The Continual Paradigm", "Methods like CoTTA (CVPR '22) and RoTTA (CVPR '23) maintain memory banks and EMA teachers across multiple patient slides."),
        ("The Pathology Failure", "When an extreme outlier slide enters the stream, corrupted features permanently poison the memory bank, contaminating subsequent patient slides."),
        ("SafeAdapt Resolution", "Deterministic episodic resets completely prevent cross-slide error propagation.")
    ], badge="Continual Poisoning", accent_color=ACCENT_RED)

    add_card(s17, Inches(4.8), Inches(1.8), Inches(3.75), Inches(5.1), "2. Gradient-Norm Pitfall in SAR", [
        ("SAR Mechanism", "Sharpness-Aware Minimization filters updates by checking for sudden gradient norm spikes."),
        ("The Pathology Reality", "Stain drift between scanners is smooth and continuous. Gradients remain low-norm but point in completely wrong directions."),
        ("Silent Drift", "SAR silently drifts into collapsed attractors on 15% of slides because its gradient threshold is never triggered.")
    ], badge="Threshold Blindness", accent_color=NAVY)

    add_card(s17, Inches(8.8), Inches(1.8), Inches(3.75), Inches(5.1), "3. Empirical Dominance", [
        ("AUROC Superiority", "SafeAdapt (0.9495) outperforms RoTTA (0.8715) by +0.078 and CoTTA (0.8210) by +0.128 AUROC."),
        ("Calibration Superiority", "ECE of 0.0418 vs 0.1150 (RoTTA) and 0.1890 (CoTTA)—a 64% reduction in miscalibration."),
        ("System Efficiency", "Requires 46% less GPU memory than RoTTA and 62% less than CoTTA.")
    ], badge="Clear Superiority", accent_color=ACCENT_GREEN)

    s17.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "We compared SafeAdapt against recent 2022-2024 continual paradigms. CoTTA and RoTTA suffer from error accumulation: when an outlier slide "
        "enters the stream, corrupted features permanently poison the memory bank, hurting subsequent slides. SAR assumes model failure causes gradient "
        "norm spikes, but stain drift produces smooth, low-norm gradients that silently drift into collapse without triggering SAR's detector. "
        "SafeAdapt outperforms RoTTA by +0.078 AUROC and CoTTA by +0.128 AUROC, with 46% less GPU memory.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Silent Gas Leak vs. The Smoke Detector: SAR is like a smoke detector that only alarms for thick black smoke (giant gradient spikes). "
        "But stain drift is like a colorless carbon monoxide leak: the gradients are small, smooth, and quiet. SAR sits silent while the model "
        "slowly suffocates. SafeAdapt's conformal monitor detects the leak instantly.\n\n"
        "KEY HIGHLIGHTS: +0.078 AUROC over RoTTA, +0.128 over CoTTA; 46% memory reduction."
    )

    # ==================== SLIDE 18: SUMMARY OF CONTRIBUTIONS ====================
    s18 = prs.slides.add_slide(blank_layout)
    set_slide_background(s18)
    add_header(s18, "Summary of Four Foundational Contributions", "Research Summary",
               "A complete theoretical, algorithmic, empirical, and clinical framework for safe test-time adaptation.")

    box_cw = Inches(5.7)
    box_ch = Inches(2.4)

    add_card(s18, Inches(0.8), Inches(1.8), box_cw, box_ch, "1. Theoretical Foundation", [
        ("Proposition 1 Proof", "Proved via Taylor expansion why entropy minimization causes representation collapse in pathology."),
        ("Rank Equivalence (rho = -0.9917)", "Demonstrated asymptotic inverse rank equivalence between epistemic variance and predictive confidence."),
        ("Mathematical Discovery", "Explained why scalar gating mechanisms in EATA and SAR fail to detect catastrophic collapse.")
    ], badge="Theoretical Rigor", accent_color=NAVY)

    add_card(s18, Inches(6.8), Inches(1.8), box_cw, box_ch, "2. Algorithmic Innovation", [
        ("Dual-Stream Sinkhorn OT", "Formulated exact marginal optimal transport to align streaming features without label-dependent collapse."),
        ("Cross-Attention Memory (CMR)", "Introduced dynamic episodic memory with an explicit slack channel to absorb staining artifacts."),
        ("Geometry Preservation", "Maintains category balance regardless of slide tumor burden.")
    ], badge="Algorithmic Novelty", accent_color=TEAL)

    add_card(s18, Inches(0.8), Inches(4.45), box_cw, box_ch, "3. Provable Safety Guarantees", [
        ("Theorem 1 Conformal Bounds", "Derived distribution-free finite-sample clinical risk bounds bounding error at significance level alpha."),
        ("Sub-2ms Deterministic Rollback", "Engineered an in-memory GPU parameter cache executing state recovery in exactly 1.8 ms."),
        ("Zero Catastrophic Collapse", "Completely eliminated catastrophic collapse across all 50 multicenter patient slides (0.0%).")
    ], badge="Provable Safety", accent_color=ACCENT_GREEN)

    add_card(s18, Inches(6.8), Inches(4.45), box_cw, box_ch, "4. SOTA Empirical Performance", [
        ("CAMELYON17 Benchmark", "0.9495 AUROC (+14.2% over source) and 0.0418 ECE across 5 university medical centers."),
        ("MIDOG 2021 Mitosis", "0.5172 F1-score (+25.5% over source) with 38.7% reduction in false-positive detections."),
        ("Clinical Edge Deployment", "312 patches/sec throughput with only 1,680 MB VRAM footprint.")
    ], badge="Empirical SOTA", accent_color=TEAL)

    s18.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "To summarize, this work delivers four foundational contributions: 1) Theoretical Rigor: proved Proposition 1 (rho = -0.9917 rank inversion) "
        "explaining why entropy minimization collapses in pathology; 2) Algorithmic Innovation: created SafeAdapt-OT and Cross-Attention Memory Routing "
        "(CMR) for geometry-preserving feature alignment; 3) Provable Safety Guarantees: formulated Theorem 1 for finite-sample Conformal Risk Control "
        "with 1.8 ms rollback; 4) SOTA Empirical Performance: demonstrated +14.2% AUROC on CAMELYON17, +25.5% F1 on MIDOG, and 0.0% collapse.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "The Complete Engineering Package: In medicine, an algorithm cannot just be a clever theory paper or a fast GPU script—it must be both. "
        "We delivered the theoretical proof of why previous methods failed, the algorithmic cure, the formal statistical safety certificate, "
        "and the validated real-time hospital deployment.\n\n"
        "KEY HIGHLIGHTS: Proposition 1 proof, Theorem 1 conformal bound, 0.9495 AUROC, 0.0% collapse."
    )

    # ==================== SLIDE 19: CONCLUSION & ROADMAP ====================
    s19 = prs.slides.add_slide(blank_layout)
    set_slide_background(s19)
    add_header(s19, "Clinical Impact, Open-Source Release & Future Directions", "Conclusion & Translation",
               "SafeAdapt bridges the gap between academic TTA research and real-world clinical whole-slide pathology deployment.")

    add_card(s19, Inches(0.8), Inches(1.8), Inches(3.75), Inches(4.0), "1. Open-Source Reproducibility", [
        ("Complete Codebase", "Fully documented PyTorch implementation with automated reproduction scripts."),
        ("Pretrained Checkpoints", "Source anchors and adaptation weights published under permissive academic open-source license."),
        ("Audit Verification", "100% reproducible across CAMELYON17 and MIDOG benchmarks with fixed seeds.")
    ], badge="Fully Reproducible", accent_color=TEAL)

    add_card(s19, Inches(4.8), Inches(1.8), Inches(3.75), Inches(4.0), "2. Prospective Clinical Deployment", [
        ("Digital Pathology Integration", "Active integration with open-source pathology viewers (QuPath) and hospital PACS (Orthanc)."),
        ("Live Multi-Scanner Trials", "Prospective validation planned across 3 hospital centers to evaluate real-world oncologist turnaround times."),
        ("Pathologist Trust", "Provides transparent confidence metrics alongside whole-slide heatmaps.")
    ], badge="Clinical Trials", accent_color=NAVY)

    add_card(s19, Inches(8.8), Inches(1.8), Inches(3.75), Inches(4.0), "3. Future Research Frontiers", [
        ("Pathology Foundation Models", "Extending SafeAdapt to self-supervised vision-language foundation models (CONCH, UNI, Virchow)."),
        ("Intraoperative Diagnostics", "Adapting SafeAdapt to rapid intraoperative frozen section analysis under extreme time constraints."),
        ("3D Volumetric Pathology", "Extending optimal transport alignment to multi-slice 3D spatial histopathology.")
    ], badge="Future Horizons", accent_color=ACCENT_GREEN)

    thank_banner = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.9))
    thank_banner.fill.solid(); thank_banner.fill.fore_color.rgb = NAVY
    thank_banner.line.color.rgb = TEAL; thank_banner.line.width = Pt(1.5)
    tf_th = thank_banner.text_frame
    tf_th.margin_left = tf_th.margin_right = Inches(0.2); tf_th.margin_top = Inches(0.12)
    p_th1 = tf_th.paragraphs[0]
    p_th1.text = "Thank you to the respected committee members and faculty reviewers."
    p_th1.font.size = Pt(14); p_th1.font.bold = True; p_th1.font.color.rgb = RGBColor(255, 255, 255)
    p_th1.alignment = PP_ALIGN.CENTER
    p_th2 = tf_th.add_paragraph()
    p_th2.text = "Open for questions, technical discussion, and viva voce examination."
    p_th2.font.size = Pt(12); p_th2.font.color.rgb = TEAL
    p_th2.alignment = PP_ALIGN.CENTER; p_th2.space_before = Pt(4)

    s19.notes_slide.notes_text_frame.text = (
        "SPOKEN SCRIPT:\n"
        "In conclusion, SafeAdapt transforms test-time adaptation into a robust, clinically certified safety framework. "
        "Our complete PyTorch codebase, model checkpoints, and reproduction scripts are released under an open-source license. "
        "We are actively integrating SafeAdapt into QuPath and Orthanc PACS for prospective multi-scanner clinical trials, and extending "
        "it to pathology foundation models (CONCH, UNI). Thank you very much for your time and guidance. I welcome your questions, "
        "technical discussion, and viva voce examination.\n\n"
        "REAL-WORLD ANALOGY:\n"
        "From the Lab Bench to the Patient's Bedside: A vaccine in a university freezer saves zero lives. It only saves lives when packaged "
        "into vials, shipped across hospitals, and injected safely into patients. SafeAdapt takes TTA out of the computer science lab and puts "
        "it directly onto the hospital workstation, ready for clinicians to trust tomorrow.\n\n"
        "KEY HIGHLIGHTS: Open-source PyTorch release; QuPath/PACS integration; ready for viva Q&A."
    )

    def safe_save(prs, path):
        try:
            prs.save(path)
            print(f"[SUCCESS] Saved to {path}")
        except PermissionError:
            alt_path = path.replace(".pptx", "_Updated.pptx")
            prs.save(alt_path)
            print(f"[INFO] {path} is currently open in PowerPoint. Saved updated version to {alt_path}")

    out_path_1 = "SafeAdapt_Presentation.pptx"
    out_path_2 = "paper/SafeAdapt_Google_Slides.pptx"
    safe_save(prs, out_path_1)
    safe_save(prs, out_path_2)
    print(f"Total slides generated: {len(prs.slides)}")

if __name__ == "__main__":
    build_presentation()
