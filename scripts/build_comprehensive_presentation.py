"""
Comprehensive SafeAdapt Presentation Generator (16:9 Widescreen)
Builds SafeAdapt_Presentation.pptx and paper/SafeAdapt_Google_Slides.pptx.

Design Principles:
- Self-explanatory, publication-grade academic presentation.
- Beautiful clinical AI palette (Deep Navy #0F172A, Clinical Teal #0D9488, Emerald #059669, Alert Red #DC2626).
- Structured Bento cards with bold lead-ins and quantitative metrics.
- High-resolution publication figures embedded directly with preserved aspect ratios.
- Styled comparison and benchmark tables for CAMELYON17, MIDOG 2021, Hardware Latency, and Ablation.
- Complete, professional speaker notes embedded into every slide.
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

    # Helper: Set Slide Background
    def set_slide_background(slide, color=SLATE_LIGHT):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    # Helper: Add Standard Header + Takeaway Banner
    def add_header(slide, title_text, category="METHODOLOGY & CLINICAL TRANSLATION", takeaway_banner=""):
        # Category Kicker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = TEAL

        # Slide Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.733), Inches(0.55))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY

        # Takeaway Banner (Key visual hook for quick comprehension)
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

    # Helper: Add Structured Bento Card
    def add_card(slide, left, top, width, height, title, items, badge="", accent_color=None, bg_color=CARD_BG):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = accent_color if accent_color else CARD_BORDER
        card.line.width = Pt(1.5 if accent_color else 1.0)

        # Content text box
        pad = Inches(0.16)
        tb = slide.shapes.add_textbox(left + pad, top + pad, width - (2 * pad), height - (2 * pad))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Header paragraph
        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.size = Pt(13)
        p_head.font.bold = True
        p_head.font.color.rgb = accent_color if accent_color else NAVY
        p_head.space_after = Pt(8)

        # Bullets / Content items
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

        # Badge pill at top right if provided
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

    # Helper: Add Embedded Image with Aspect Ratio Preservation & Border
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

        # Outer card frame
        frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x - Inches(0.06), y - Inches(0.06), w + Inches(0.12), h + Inches(0.12) + (Inches(0.3) if caption else 0))
        frame.fill.solid()
        frame.fill.fore_color.rgb = CARD_BG
        frame.line.color.rgb = CARD_BORDER
        frame.line.width = Pt(1.0)

        # Add image
        pic = slide.shapes.add_picture(image_path, x, y, w, h)

        # Caption
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

    # Helper: Add Styled Table
    def add_styled_table(slide, left, top, width, height, headers, rows, col_widths, highlight_row_idx=None, alert_row_idx=None):
        table_shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height)
        table = table_shape.table

        # Set column widths
        for col_idx, cw in enumerate(col_widths):
            table.columns[col_idx].width = cw

        # Header Row
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

        # Data Rows
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

    # Top Tag
    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(4.5), Inches(0.35))
    pill.fill.solid(); pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    pill.line.color.rgb = TEAL; pill.line.width = Pt(1.0)
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = "IEEE TRANSACTIONS ON MEDICAL IMAGING (TMI)"
    p_pill.font.size = Pt(10); p_pill.font.bold = True; p_pill.font.color.rgb = TEAL
    p_pill.alignment = PP_ALIGN.CENTER

    # Main Title
    t_box1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(1.8))
    tf1 = t_box1.text_frame; tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0
    p1 = tf1.paragraphs[0]
    p1.text = "SafeAdapt"
    p1.font.size = Pt(46); p1.font.bold = True; p1.font.color.rgb = TEAL
    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Robust Test-Time Adaptation via Optimal Transport and Conformal Risk Control in Digital Pathology"
    p1_sub.font.size = Pt(22); p1_sub.font.bold = True; p1_sub.font.color.rgb = RGBColor(248, 250, 252); p1_sub.space_before = Pt(6)

    # Research Metadata
    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(11.3), Inches(1.0))
    tf_m = meta_box.text_frame
    tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = 0
    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "Author: Avinash Reddy Cheerapareddy   |   Co-Author: Gunnam Sanjana"
    p_m1.font.size = Pt(15); p_m1.font.bold = True; p_m1.font.color.rgb = RGBColor(226, 232, 240)
    p_m2 = tf_m.add_paragraph()
    p_m2.text = "Department of Computer Science & Engineering, SRM University-AP   •   Final Thesis & Scientific Defense"
    p_m2.font.size = Pt(13); p_m2.font.color.rgb = RGBColor(148, 163, 184); p_m2.space_before = Pt(4)

    # 4 Executive Impact Stat Badges
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
        "Good morning, respected members of the committee, faculty, and reviewers. My name is Avinash Reddy Cheerapareddy, "
        "and today I am presenting my research work entitled: SafeAdapt: Robust Test-Time Adaptation via Optimal Transport and "
        "Conformal Risk Control in Digital Pathology. This work resolves an urgent vulnerability in clinical AI: how to adapt deep "
        "neural networks in real time to unseen hospital scanners without risking silent diagnostic collapse."
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
        "In digital pathology, computational models achieve remarkable accuracy on slides from the hospital where they were trained. "
        "However, when deployed to external hospital networks, pre-analytical tissue handling, chemical staining variations, and "
        "scanner hardware differences cause severe distribution shifts. A model trained at Radboud achieving over 0.98 AUROC drops to "
        "0.73 AUROC on Center 3 or Center 4. Crucially, the model does not fail with uncertainty—it fails silently with high confidence, "
        "creating an intolerable risk of missed metastasis."
    )

    # ==================== SLIDE 3: FAILURE OF CURRENT TTA ====================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Why Standard Test-Time Adaptation Collapses in Pathology", "Prior Art Breakdown",
               "Entropy minimization creates an unsupervised feedback loop that pulls models into degenerate single-class attractors.")

    # Left: Text Cards
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

    # Right: Embedded Figure fig_tent_collapse_mechanism.png
    add_image_box(s3, "paper/ieee/figs/fig_tent_collapse_mechanism.png", Inches(6.9), Inches(1.8), Inches(5.6), Inches(5.1),
                  "Empirical Collapse: Tent entropy optimization destroys latent feature margins across streaming WSI batches.")

    s3.notes_slide.notes_text_frame.text = (
        "Why can't we simply apply standard computer vision test-time adaptation methods like Tent or EATA? "
        "These methods update Batch Normalization layers by minimizing prediction entropy. In natural images with light blur, "
        "this works. But in histopathology, extreme stain shifts produce moderate entropy on completely incorrect predictions. "
        "Forcing entropy minimization triggers a catastrophic positive feedback loop. Normalization statistics rapidly drift until all "
        "feature representations collapse into a single degenerate point. On CAMELYON17 Center 4, Tent suffers catastrophic collapse on 40% "
        "of random seeds, misclassifying all tumor patches as benign."
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
        "In Section III of our manuscript, we formalize the exact mathematical root cause behind TTA collapse in Proposition 1. "
        "Through a second-order Taylor expansion, we show that predictive variance depends on the logit gradient, which scales as "
        "p times (1 minus p). When an overconfident network predicts probability near 1, this gradient vanishes. Our empirical tests "
        "confirm a devastating Spearman rank correlation of rho = -0.9917 between predictive entropy and epistemic uncertainty. "
        "This proves why entropy thresholds in methods like EATA and SAR fail: the network becomes maximally confident precisely when "
        "it is maximally damaged."
    )

    # ==================== SLIDE 5: SAFEADAPT ARCHITECTURE ====================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "SafeAdapt: A Deterministic, Fail-Closed TTA Architecture", "Proposed Framework",
               "SafeAdapt introduces geometry-preserving optimal transport, contextual artifact rejection, and finite-sample risk bounds.")

    # Top: Embedded Architecture Diagram
    add_image_box(s5, "paper/ieee/figs/fig_architecture.png", Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.1),
                  "SafeAdapt System Architecture: Dual-Stream Sinkhorn OT, Cross-Attention Memory Routing (CMR), and Conformal Rollback.")

    # Bottom: 4 Interlocking Modules
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
        "To solve this challenge, we developed SafeAdapt, a deterministic, fail-closed test-time adaptation architecture. "
        "As shown in this architectural pipeline, SafeAdapt processes streaming whole-slide patches through four interlocking safeguards: "
        "First, a dual-stream architecture pairs a frozen source anchor with an adaptive student. Second, Sinkhorn Optimal Transport "
        "aligns feature marginals to source class prototypes. Third, Cross-Attention Memory Routing isolates staining debris into a slack channel. "
        "Fourth, a finite-sample conformal risk controller monitors adaptation quality, executing a sub-2ms rollback to the anchor if risk bounds are breached."
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
        "Our first core mechanism replaces hazardous entropy minimization with entropy-regularized Sinkhorn Optimal Transport. "
        "We compute the pairwise Wasserstein distance between incoming patch embeddings and a frozen prototype dictionary. "
        "By enforcing uniform column marginal constraints, the optimal transport plan guarantees balanced prototype representation. "
        "Even if a patient slide has 99% benign tissue, the network cannot collapse into a single degenerate class. "
        "Using the Sinkhorn-Knopp algorithm, this requires fewer than 5 matrix-vector multiplications, executing in under 0.4 milliseconds."
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
        "Our second mechanism is Cross-Attention Memory-Augmented Routing, or CMR. Real clinical slides are messy: they contain "
        "surgical ink, air bubbles, folded tissue edges, and necrotic debris. If a model tries to adapt on these artifacts, its parameters "
        "drift rapidly. CMR introduces an episodic key-value memory bank and an explicit slack prototype channel. "
        "Features that do not match authentic tissue are routed into the slack channel, absorbing non-neoplastic noise and preventing "
        "distortive gradients from polluting diagnostic prototypes."
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
        "Our third mechanism provides finite-sample statistical safety through Conformal Risk Control, formalized in Theorem 1. "
        "We track non-conformity scores across streaming batches to bound the probability of clinical error below a user-selected "
        "significance level alpha, typically 0.05. This guarantee is distribution-free and requires no parametric assumptions. "
        "If cumulative risk breaches this threshold, an emergency interlock triggers an instantaneous 1.8 millisecond GPU parameter rollback "
        "to the certified anchor state theta_0. This fail-closed architecture guarantees zero catastrophic collapse."
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
        "To rigorously evaluate SafeAdapt, we benchmarked across two challenging international multicenter datasets: CAMELYON17, "
        "comprising 50 whole-slide images across 5 university medical centers, and the MIDOG 2021 challenge for cross-scanner and "
        "cross-species mitosis detection. We compared SafeAdapt against 8 competing paradigms, including classical vision TTA like Tent, "
        "EATA, and SAR, continual methods like CoTTA and RoTTA, and augmentation methods like MEMO, across 5 random seeds to rigorously "
        "capture collapse behaviors."
    )

    # ==================== SLIDE 10: MAIN RESULTS: CAMELYON17 ====================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Superior Diagnostic Performance Across Hospital Cohorts", "Empirical Benchmarks: CAMELYON17",
               "SafeAdapt achieves new state-of-the-art: 0.9495 AUROC, lowest ECE (0.0418), and 0.0% catastrophic collapse.")

    # Left: Styled Table I
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

    # Callout chip below table
    c_chip = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.25), Inches(6.1), Inches(0.65))
    c_chip.fill.solid(); c_chip.fill.fore_color.rgb = TEAL_BG
    c_chip.line.color.rgb = TEAL; c_chip.line.width = Pt(1.0)
    tf_chip = c_chip.text_frame
    tf_chip.margin_left = Inches(0.12); tf_chip.margin_top = Inches(0.08)
    p_ch = tf_chip.paragraphs[0]
    p_ch.text = "EMPIRICAL BREAKTHROUGH: +14.2% AUROC over Source, +9.8% over best baseline (MEMO), and 0.0% collapse."
    p_ch.font.size = Pt(9.5); p_ch.font.bold = True; p_ch.font.color.rgb = TEAL_DARK

    # Right: Embedded Pareto Plot fig_v2_benchmark_auroc_ece.png
    add_image_box(s10, "paper/ieee/figs/fig_v2_benchmark_auroc_ece.png", Inches(7.1), Inches(1.8), Inches(5.4), Inches(5.1),
                  "AUROC vs. Expected Calibration Error (ECE): SafeAdapt occupies the ideal top-left Pareto frontier.")

    s10.notes_slide.notes_text_frame.text = (
        "Here are the primary empirical results on CAMELYON17 multicentric evaluation. "
        "As shown in Table I, unadapted source ResNet-50 achieves 0.8312 AUROC. Standard Tent collapses catastrophically to 0.7645 AUROC "
        "with a 40% collapse rate and terrible calibration error of 0.2810. Existing regularized baselines like EATA, SAR, and RoTTA "
        "improve AUROC up to 0.8715, but still suffer from a 10% to 20% collapse rate. SafeAdapt achieves a new state-of-the-art "
        "0.9495 AUROC, an unprecedented ECE of 0.0418, and exactly 0.0% collapse across all 50 patient slides."
    )

    # ==================== SLIDE 11: SLIDE 29 CASE STUDY ====================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Resolving Slide 29: From Diagnostic Collapse to 0.9324 AUROC", "Clinical Case Study & Attractor Reversal",
               "SafeAdapt detects early parameter drift on Slide 29, triggering rollback and recovering 0.9324 AUROC.")

    # Top: Embedded UMAP Attractor Reversal Figure
    add_image_box(s11, "paper/ieee/figs/fig_v2_slide29_attractor_reversal.png", Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.0),
                  "UMAP Latent Space on Patient Slide 29: (A) Source baseline; (B) Tent collapse into single point; (C) SafeAdapt recovery.")

    # Bottom: 3 Analytical Bento Cards
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
        "Let us examine a critical clinical case study on Patient Slide 29. Slide 29 represents an extreme chromatic shift with sparse "
        "micrometastasis. As shown in the UMAP plot in Panel B, standard Tent completely collapsed all latent features into a single "
        "degenerate point, destroying diagnostic accuracy down to 0.5120. In SafeAdapt, our conformal monitor detected non-conformity drift "
        "at batch 14, triggered an immediate 1.8ms rollback, and allowed Sinkhorn OT to re-align feature marginals. As shown in Panel C, "
        "diagnostic margins were fully restored, achieving 0.9324 AUROC."
    )

    # ==================== SLIDE 12: MITOSIS DETECTION (MIDOG) ====================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Cross-Scanner & Cross-Species Mitosis Detection (MIDOG)", "Empirical Benchmarks: Detection",
               "SafeAdapt boosts mitotic figure F1-score to 0.5172, suppressing false positives across unseen scanners and species.")

    # Top: Embedded Mitosis Figure
    add_image_box(s12, "paper/ieee/figs/fig_v2_cmr_mitosis_detection.png", Inches(0.8), Inches(1.8), Inches(11.733), Inches(2.9),
                  "MIDOG 2021 Mitosis Detection: PR Curves and qualitative crops showing false-positive suppression on unseen scanners.")

    # Bottom Left: Table III
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

    # Bottom Right: Analysis Card
    add_card(s12, Inches(6.5), Inches(4.85), Inches(6.033), Inches(2.05), "CMR Slack Routing Eliminates Debris", [
        ("The Challenge", "Pyknotic apoptotic nuclei and hyperchromatic debris mimic mitotic figures on unseen scanners."),
        ("False-Alarm Explosion", "Tent experiences severe FP explosion (Precision drops from 0.38 down to 0.27)."),
        ("SafeAdapt Superiority", "CMR slack channel absorbs hyperchromatic debris, boosting Precision to 0.5284 (+38.7% over Source) and achieving SOTA F1-score of 0.5172.")
    ], badge="+25.5% F1 Gain", accent_color=ACCENT_GREEN)

    s12.notes_slide.notes_text_frame.text = (
        "To test whether SafeAdapt generalizes beyond classification to dense object detection, we evaluated on the MIDOG 2021 challenge "
        "for mitotic figure detection across 4 scanner types and multiple animal species. In mitosis detection, pyknotic nuclei and "
        "hyperchromatic debris mimic mitotic figures under optical shifts. Under Tent, false positives exploded, dropping precision to 0.27. "
        "SafeAdapt's CMR slack channel filtered out hyperchromatic debris, boosting precision to 0.5284 and achieving the highest F1-score "
        "of 0.5172—a 25.5% relative improvement over unadapted source models."
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
        "Beyond standard mini-batch evaluation, we stress-tested SafeAdapt across four extreme clinical boundary regimes: "
        "First, single-patch streaming with batch size B=1, where our FIFO buffer maintains 0.9412 AUROC. Second, purely benign lymph nodes "
        "with 0.0% tumor burden, where SafeAdapt showed zero false-positive drift and 99.4% specificity. Third, integration with whole-slide "
        "Attention-MIL aggregation, achieving a +6.8% boost in patient-level triage. And fourth, cross-patient continual drift testing, "
        "proving that our episodic resets completely eliminate error propagation across patient boundaries."
    )

    # ==================== SLIDE 14: HARDWARE & EDGE LATENCY ====================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Hardware Benchmarks: Clinical PACS Integration & Real-Time Latency", "Hardware Feasibility",
               "SafeAdapt delivers 312 patches/sec at 1.8 ms rollback latency, easily fitting standard hospital PACS infrastructure.")

    # Left: Table IV Hardware Latency
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

    # Hardware takeaways below table
    add_card(s14, Inches(0.8), Inches(4.75), Inches(6.0), Inches(2.15), "Real-Time Clinical PACS Compatibility", [
        ("Negligible Overhead", "Only +0.4 ms overhead per patch over frozen baseline (3.2 ms vs 2.8 ms)."),
        ("Ultra-Low Memory Footprint", "Requires only 1,680 MB VRAM, fitting into standard 8GB hospital edge workstations."),
        ("Instant Rollback", "Rollback executes in 1.8 ms—recovering parameters in under 1/500th of a second without pipeline stall.")
    ], badge="312 patches/sec", accent_color=ACCENT_GREEN)

    # Right: Embedded Convergence Figure fig_v2_gpu_training_convergence.png
    add_image_box(s14, "paper/ieee/figs/fig_v2_gpu_training_convergence.png", Inches(7.0), Inches(1.8), Inches(5.5), Inches(5.1),
                  "GPU Inference Latency & Parameter Convergence: SafeAdapt converges in <= 5 steps with minimal latency overhead.")

    s14.notes_slide.notes_text_frame.text = (
        "In clinical pathology, algorithms must keep pace with high-throughput scanners. As shown in Table IV and the convergence plot, "
        "SafeAdapt requires only 3.2 milliseconds per patch—adding a mere 0.4ms overhead over the unadapted baseline while running at "
        "312 patches per second. Compare this to MEMO, which requires 48.6 milliseconds per patch—17 times slower and clinically unusable. "
        "Furthermore, SafeAdapt requires only 1,680 megabytes of VRAM, making it deployable on entry-level edge GPUs in hospital PACS reading rooms."
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
        "Clinical AI deployment requires strict adherence to regulatory standards. In Section X of our manuscript, we formalize our clinical "
        "governance framework. In clinical practice, patient slide order is arbitrary. Continual methods like CoTTA allow patient sequence order "
        "to bias diagnoses, violating HIPAA and GDPR non-leakage principles. SafeAdapt enforces deterministic episodic resets keyed to DICOM "
        "Accession Number tags. Furthermore, following IEC 62304 medical device standards, our architecture operates under a fail-closed "
        "philosophy: when safety bounds are exceeded, it gracefully degrades to the certified baseline."
    )

    # ==================== SLIDE 16: COMPREHENSIVE ABLATION STUDY ====================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16)
    add_header(s16, "Dissecting SafeAdapt: Component-by-Component Impact", "Ablation & Sensitivity Analysis",
               "Every architectural safeguard provides measurable value: removing any module degrades accuracy or re-introduces collapse.")

    # Left: Ablation Table V
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

    # Callout chip below table
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

    # Right: 2 Summary Cards
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
        "To verify that every component in SafeAdapt is strictly necessary, we conducted an exhaustive ablation study shown in Table V. "
        "Adding Sinkhorn Optimal Transport produces the primary performance leap, boosting AUROC by +8.1% and reducing collapse from 40% "
        "to 8%. Adding Cross-Attention Memory Routing contributes another +2.2% AUROC by isolating tissue artifacts. Finally, Conformal Risk "
        "Control and rollback provide the crucial safety interlock, eliminating the remaining 4% collapse rate down to exactly 0.0%. "
        "Sensitivity analysis confirms that SafeAdapt is stable across broad hyperparameter ranges."
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
        "We also compared SafeAdapt against recent 2023–2024 continual adaptation paradigms. Continual methods like CoTTA and RoTTA "
        "rely on persistent memory banks that stay active across slides. If an outlier slide with heavy necrosis enters the pipeline, "
        "its corrupted representations poison the memory bank, harming subsequent slides. Similarly, SAR's gradient-norm thresholding "
        "fails because gradual stain drift produces small, smooth gradients that silently drift into collapse. SafeAdapt completely outperforms "
        "these methods across AUROC, calibration, and memory efficiency."
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
        "To summarize, this research delivers four major contributions: First, theoretical rigor: we mathematically proved why entropy "
        "minimization breaks down in digital pathology through Proposition 1. Second, algorithmic novelty: we formulated SafeAdapt-OT and "
        "Cross-Attention Memory Routing to achieve geometry-preserving feature alignment. Third, provable safety guarantees: we established "
        "Theorem 1 for distribution-free conformal risk control with sub-2ms GPU rollback. Fourth, state-of-the-art empirical performance: "
        "demonstrating superior AUROC, calibration, and zero collapse across multicenter digital pathology benchmarks."
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

    # Closing Thank You Banner
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
        "In conclusion, SafeAdapt transforms test-time adaptation from an unpredictable experimental technique into a robust, "
        "clinically certified safety framework for digital pathology. Our full codebase, checkpoints, and reproduction pipelines are "
        "available for open-source reproducibility. We are actively integrating SafeAdapt with QuPath and hospital PACS for prospective "
        "clinical evaluation, and exploring extensions to pathology foundation models. "
        "Thank you very much for your time and consideration. I am now ready for questions and technical discussion."
    )

    # Save presentation to both root and paper directories
    out_path_1 = "SafeAdapt_Presentation.pptx"
    out_path_2 = "paper/SafeAdapt_Google_Slides.pptx"
    prs.save(out_path_1)
    prs.save(out_path_2)
    print(f"[SUCCESS] Saved comprehensive presentation to {out_path_1} and {out_path_2}")
    print(f"Total slides generated: {len(prs.slides)}")

if __name__ == "__main__":
    build_presentation()
