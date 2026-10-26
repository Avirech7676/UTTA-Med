"""
Builds an elite, 16:9 widescreen PowerPoint presentation (SafeAdapt_Google_Slides.pptx)
designed specifically for native import into Google Slides.
Includes:
- Professional academic medical AI color palette (Navy, Medical Blue, Crimson Alert, Emerald Success)
- High-resolution embedded figures from paper/ieee/figs/
- Styled tables, bento card containers, and stat callouts
- Native speaker notes attached to every slide for conference/defense presentation
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen standard
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6] # Blank slide

    # Color Palette
    C_NAVY = RGBColor(14, 43, 92)       # #0E2B5C
    C_BLUE = RGBColor(28, 90, 160)      # #1C5AA0
    C_RED = RGBColor(180, 40, 40)       # #B42828 (Alert/Collapse)
    C_GREEN = RGBColor(34, 139, 34)     # #228B22 (Success)
    C_BG_CARD = RGBColor(248, 250, 252) # #F8FAFC
    C_TEXT = RGBColor(30, 41, 59)       # #1E293B
    C_MUTED = RGBColor(100, 116, 139)   # #64748B
    C_WHITE = RGBColor(255, 255, 255)
    C_BORDER = RGBColor(203, 213, 225)  # #CBD5E1

    def add_header(slide, title_text, category_text="SAFEADAPT RESEARCH DECK"):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_BLUE

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY

    def add_footer(slide, current_num, total_num=18):
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.3))
        tf = footer_box.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = f"SafeAdapt: Geometry-Preserving Optimal Transport in Digital Pathology  |  IEEE TMI 2026  |  DOI: 10.5281/zenodo.23658430          Slide {current_num} of {total_num}"
        p.font.size = Pt(9)
        p.font.color.rgb = C_MUTED

    def add_card(slide, left, top, width, height, bg_color=C_BG_CARD, border_color=C_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
        return shape

    def add_speaker_note(slide, note_text):
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = note_text

    # =========================================================
    # SLIDE 1: TITLE SLIDE (Dark Navy Hero Theme)
    # =========================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.fill.background()

    # Accent Stripe
    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.6), Inches(0.08))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = RGBColor(56, 189, 248) # Light blue accent
    stripe.line.fill.background()

    # Title box
    t_box = s1.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "SafeAdapt: Geometry-Preserving Optimal Transport and Fail-Closed Safeguards for Test-Time Adaptation in Digital Pathology"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = C_WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Reversing Negative Transfer and Precluding Attractor Collapse Under Multicenter Clinical Shift"
    p1_sub.font.size = Pt(16)
    p1_sub.font.color.rgb = RGBColor(148, 163, 184) # Slate 400
    p1_sub.space_before = Pt(14)

    # Authors Card
    c_auth = add_card(s1, Inches(0.8), Inches(4.2), Inches(6.8), Inches(2.3), bg_color=RGBColor(23, 58, 114), border_color=RGBColor(38, 78, 142))
    tf_auth = c_auth.text_frame
    tf_auth.word_wrap = True
    tf_auth.margin_left = Inches(0.3)
    tf_auth.margin_top = Inches(0.25)
    
    pa1 = tf_auth.paragraphs[0]
    pa1.text = "Avinash Reddy Cheerapareddy  &  Gunnam Sanjana"
    pa1.font.size = Pt(16)
    pa1.font.bold = True
    pa1.font.color.rgb = C_WHITE
    
    pa2 = tf_auth.add_paragraph()
    pa2.text = "Department of Computer Science and Engineering\nSRM University-AP, Amaravati, Andhra Pradesh, India\navinashreddy_cheerapareddy@srmap.edu.in | sanjana_gunnam@srmap.edu.in"
    pa2.font.size = Pt(12)
    pa2.font.color.rgb = RGBColor(203, 213, 225)
    pa2.space_before = Pt(8)

    # Repository & Archive Card
    c_repo = add_card(s1, Inches(7.8), Inches(4.2), Inches(4.7), Inches(2.3), bg_color=RGBColor(23, 58, 114), border_color=RGBColor(38, 78, 142))
    tf_repo = c_repo.text_frame
    tf_repo.word_wrap = True
    tf_repo.margin_left = Inches(0.3)
    tf_repo.margin_top = Inches(0.25)
    
    pr1 = tf_repo.paragraphs[0]
    pr1.text = "Open Science & Reproducibility"
    pr1.font.size = Pt(15)
    pr1.font.bold = True
    pr1.font.color.rgb = RGBColor(56, 189, 248)

    pr2 = tf_repo.add_paragraph()
    pr2.text = "• Official Zenodo Archive: 10.5281/zenodo.23658430\n• GitHub: github.com/Avirech7676/UTTA-Med\n• Audit: 57/57 Bit-Exact Matches | 148 Unit Tests\n• Target: IEEE Transactions on Medical Imaging (TMI)"
    pr2.font.size = Pt(11)
    pr2.font.color.rgb = RGBColor(226, 232, 240)
    pr2.space_before = Pt(8)

    add_speaker_note(s1, 
        "Good morning, esteemed colleagues and committee members. Today, I am proud to present SafeAdapt: "
        "a geometry-preserving optimal transport and fail-closed safeguarding framework that addresses one of "
        "the most critical vulnerabilities in computational pathology—the catastrophic collapse of test-time "
        "adaptation under multicenter clinical domain shift. All code, data, and models are fully open-sourced."
    )

    # =========================================================
    # SLIDE 2: MOTIVATION
    # =========================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Cross-Center Stain and Scanner Variability Degrades Deep Pathology Models", "1. Clinical Motivation")
    add_footer(s2, 2)

    # Left Column: Problem Cards
    c1 = add_card(s2, Inches(0.8), Inches(1.6), Inches(7.2), Inches(2.4))
    tf = c1.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "The Reality of Clinical Deployment"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Digital pathology models fail when transferred across clinical institutions.\n• Variations in tissue microtomy, histochemical staining batches (H&E), and scanner sensor optics induce massive, unpredictable domain shifts.\n• Retraining is legally prohibited: Patient privacy regulations (HIPAA, GDPR) strictly forbid sharing raw whole-slide images (WSIs) across hospitals."
    p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    c2 = add_card(s2, Inches(0.8), Inches(4.3), Inches(7.2), Inches(2.4))
    tf = c2.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "The Promise & Challenge of Test-Time Adaptation (TTA)"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Test-Time Adaptation adapts model weights on-the-fly using unlabeled target slides.\n• Requires ZERO access to source hospital training data.\n• Must operate continuously during clinical whole-slide scanning without human intervention."
    p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    # Right Column: Big Stat Callout
    c_stat = add_card(s2, Inches(8.3), Inches(1.6), Inches(4.2), Inches(5.1), bg_color=RGBColor(254, 242, 242), border_color=C_RED)
    tf = c_stat.text_frame
    tf.margin_left = tf.margin_top = Inches(0.35)
    p = tf.paragraphs[0]
    p.text = "THE CLINICAL TRANSFER GAP"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_RED
    
    p = tf.add_paragraph()
    p.text = "-28%"
    p.font.size = Pt(48); p.font.bold = True; p.font.color.rgb = C_RED
    p.space_before = Pt(10)

    p = tf.add_paragraph()
    p.text = "Sensitivity Drop on Unseen Hospital Slides\n\nTransferring a ResNet-18 trained on Camelyon17 Centers 0, 3, 4 to unseen Center 2 drops metastasis detection sensitivity by up to 28% due to histochemical stain disparity."
    p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    p = tf.add_paragraph()
    p.text = "Camelyon17 Benchmark: 455,954 patches across 5 clinical centers."
    p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_MUTED; p.space_before = Pt(16)

    add_speaker_note(s2,
        "In digital pathology, algorithms trained on one hospital's slides routinely fail when deployed at another. "
        "Staining reagents, tissue slicing, and scanner optics introduce massive domain shifts. Centralizing patient "
        "data for retraining is barred by HIPAA and GDPR. Thus, models must adapt on-the-fly during inference."
    )

    # =========================================================
    # SLIDE 3: RESEARCH GAP (TENT COLLAPSE)
    # =========================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Existing Test-Time Adaptation Methods Suffer Catastrophic Attractor Collapse", "2. Research Gap")
    add_footer(s3, 3)

    c_gap = add_card(s3, Inches(0.8), Inches(1.6), Inches(6.2), Inches(5.1))
    tf = c_gap.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Unconstrained Entropy Minimization (Tent, ICLR'21)"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    
    p = tf.add_paragraph()
    p.text = "• Adapts BatchNorm affine parameters (scale γ, shift β) by minimizing Shannon entropy: L = -Σ p log p.\n• Underlying Assumption: Confident predictions reflect correct target domain alignment."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    p = tf.add_paragraph()
    p.text = "The Discovery: 40% Collapse Rate on Real Pathology"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_RED; p.space_before = Pt(12)

    p = tf.add_paragraph()
    p.text = "• Across 5 random seeds on Camelyon17 (N = 85,054 target patches):\n   - Tent collapses on 2 out of 5 seeds (Seeds 7 and 99).\n   - Macro AUROC drops from 0.930 to 0.3429 (worse than random guessing!).\n   - Sensitivity collapses to < 3% (misses almost all cancer).\n   - Harmful adaptation rate reaches 18.7% ± 17.9%."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    # Right Column: Visual Figure
    fig_path = os.path.join('paper', 'ieee', 'figs', 'fig_tent_collapse_mechanism.png')
    if os.path.exists(fig_path):
        s3.shapes.add_picture(fig_path, Inches(7.3), Inches(1.6), width=Inches(5.2))
    else:
        c_alt = add_card(s3, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.1), bg_color=RGBColor(254, 242, 242), border_color=C_RED)

    add_speaker_note(s3,
        "Test-Time Adaptation, specifically entropy minimization like Tent, seemed promising. But when we conducted "
        "a rigorous 5-seed audit on Camelyon17, we uncovered a shocking truth: Tent collapses on 40% of random seeds. "
        "On Seeds 7 and 99, AUROC plunges to 0.34, sensitivity drops to near-zero, and the harm rate reaches nearly 19%. "
        "Erroneous pseudo-gradients drag the entire network into a single-class attractor."
    )

    # =========================================================
    # SLIDE 4: RESEARCH QUESTION & OBJECTIVES
    # =========================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Research Question: Achieving Guaranteed Positive Transfer and Clinical Safety", "3. Core Objective")
    add_footer(s4, 4)

    # Big Central Question Callout
    c_q = add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.4), bg_color=RGBColor(239, 246, 255), border_color=C_BLUE)
    tf = c_q.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "CENTRAL RESEARCH QUESTION"
    p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_BLUE
    p = tf.add_paragraph()
    p.text = "Can streaming test-time adaptation achieve guaranteed positive transfer in computational pathology without falling into single-class attractor collapse or violating clinical safety bounds?"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_before = Pt(4)

    # 3 Objective Columns
    objs = [
        ("1. Disprove Fallacies", "Analyze why scalar uncertainty and confidence gating fail to prevent representation collapse under realistic clinical class imbalance.", C_NAVY),
        ("2. Geometric Alignment", "Formulate an Entropic Optimal Transport objective with exact marginal constraints (Q^T 1_B = r) against certified source prototypes.", C_BLUE),
        ("3. Fail-Closed Safeguards", "Engineer tissue optical density monitors, PAC parameter bounds (||Δθ||_inf <= 0.05), and episodic WSI resets for patient isolation.", C_GREEN)
    ]
    for idx, (title, desc, color) in enumerate(objs):
        c_obj = add_card(s4, Inches(0.8 + idx * 4.0), Inches(3.3), Inches(3.7), Inches(3.4))
        tf = c_obj.text_frame
        tf.margin_left = tf.margin_top = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = color
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(10)

    add_speaker_note(s4,
        "This led directly to our central research question: Can we formulate streaming test-time adaptation "
        "such that positive transfer is mathematically guaranteed, while precluding attractor collapse and enforcing "
        "fail-closed runtime safety? We broke this into three concrete objectives."
    )

    # =========================================================
    # SLIDE 5: CONTRIBUTION OVERVIEW (4-Quadrant Bento Grid)
    # =========================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "SafeAdapt Introduces Four Foundational Contributions", "4. Contributions")
    add_footer(s5, 5)

    contribs = [
        ("THEORETICAL DISPROOF: THE TAYLOR TRAP", 
         "Proved via Taylor expansion that MC-Dropout predictive variance is asymptotically rank-equivalent to softmax confidence (ρ = -0.9917) in 1D sigmoid heads. Uncertainty gating wastes 3.55x latency with zero added discrimination.", 
         Inches(0.8), Inches(1.6), Inches(5.7), Inches(2.4), C_NAVY),
        
        ("GEOMETRY-PRESERVING OT-TTA", 
         "Formulated batch adaptation as an Entropic Optimal Transport problem with exact target marginal prior constraints (Q^T 1_B = r). Mathematically predefines class distribution, precluding attractor collapse.", 
         Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.4), C_BLUE),

        ("BIOLOGICAL SAFEGUARDS: CMR TISSUE GATING", 
         "Cellularity-Mitosis Ratio (CMR) leverages Beer-Lambert optical density deconvolution, eliminating all 14 false silence alarms on benign acellular stroma during MIDOG 2021 mitosis detection.", 
         Inches(0.8), Inches(4.3), Inches(5.7), Inches(2.4), C_GREEN),

        ("PROVABLE PAC TRUST REGION & RUNTIME INTERLOCKS", 
         "Proved PAC L_inf <= 0.05 affine drift bounds maximum output probability shift to <= 0.0460. Coupled with 1.8 ms dual-buffer atomic rollbacks and episodic WSI-boundary patient isolation resets.", 
         Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.4), C_NAVY)
    ]

    for title, desc, left, top, w, h, col in contribs:
        c = add_card(s5, left, top, w, h)
        tf = c.text_frame
        tf.margin_left = tf.margin_top = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    add_speaker_note(s5,
        "Our work delivers four core contributions spanning theory, algorithms, clinical biology, and systems engineering. "
        "We disprove the utility of MC-Dropout in 1D heads, solve attractor collapse with Optimal Transport, engineer biological "
        "tissue gating, and establish provable PAC parameter bounds."
    )

    # =========================================================
    # SLIDE 6: PROPOSED APPROACH OVERVIEW (Architecture)
    # =========================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "SafeAdapt Architecture: High-Level End-to-End Pipeline", "5. System Architecture")
    add_footer(s6, 6)

    fig_arch = os.path.join('paper', 'ieee', 'figs', 'fig_architecture.png')
    if os.path.exists(fig_arch):
        s6.shapes.add_picture(fig_arch, Inches(0.8), Inches(1.5), width=Inches(11.7))

    add_speaker_note(s6,
        "Here is the SafeAdapt pipeline. When target patches stream from the scanner, clinical diagnostics are emitted "
        "first, before any model updates occur. Next, the patch undergoes biological optical density gating. If admitted, "
        "Entropic Optimal Transport aligns features against certified prototypes. Before committing changes, runtime "
        "sentinels verify that parameters remain within our PAC displacement sphere. If anything diverges, an atomic rollback "
        "instantly restores the certified state."
    )

    # =========================================================
    # SLIDE 7: METHOD: KEY COMPONENT 1 (OPTIMAL TRANSPORT)
    # =========================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Entropic Optimal Transport with Marginal Constraints Precludes Collapse", "6. Method: OT-TTA")
    add_footer(s7, 7)

    c_ot = add_card(s7, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))
    tf = c_ot.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Mathematical Formulation"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    
    p = tf.add_paragraph()
    p.text = "min <Q, M> - ε H(Q)   s.t. Q 1_K = 1/B 1_B,  Q^T 1_B = r\n\n• Cost matrix: M_ik = -log p(y=k | x_i)\n• p = (1/B) 1_B : Uniform sample marginal\n• r = [r_0, r_1]^T : Certified target class marginal prior\n• Solved via Sinkhorn-Knopp iterations in < 2.5 ms per batch"
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    p = tf.add_paragraph()
    p.text = "Why Attractor Collapse is Impossible"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_GREEN; p.space_before = Pt(10)

    p = tf.add_paragraph()
    p.text = "• In Tent, all predictions can collapse to a single majority class.\n• In SafeAdapt-OT, the column constraint Q^T 1_B = r mathematically forces exactly r_1 * B mass to be transported to the tumor class.\n• Reverses collapse on severe outlier Slide 29: AUROC 0.0521 -> 0.9324!"
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    # Right: Slide 29 Reversal Figure
    fig_s29 = os.path.join('paper', 'ieee', 'figs', 'fig_v2_slide29_attractor_reversal.png')
    if os.path.exists(fig_s29):
        s7.shapes.add_picture(fig_s29, Inches(7.1), Inches(1.6), width=Inches(5.4))

    add_speaker_note(s7,
        "Why does Optimal Transport eliminate attractor collapse? In standard entropy minimization, nothing prevents "
        "the network from predicting 100% benign tissue. In SafeAdapt-OT, we enforce exact marginal prior constraints: "
        "Q transpose ones equals r. The Sinkhorn-Knopp algorithm forces the transportation matrix to assign exactly the "
        "expected biological mass to the tumor class. This single constraint rescues outlier Slide 29 from complete Tent failure."
    )

    # =========================================================
    # SLIDE 8: METHOD: KEY COMPONENT 2 (TISSUE GATING & PAC BOUNDS)
    # =========================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Biological Tissue Gating and Provable PAC Parameter Bounds", "7. Method: Safeguards")
    add_footer(s8, 8)

    # Left: CMR Tissue Gating
    c_cmr = add_card(s8, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tf = c_cmr.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Cellularity-Mitosis Ratio (CMR)"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    
    p = tf.add_paragraph()
    p.text = "• The Clinical Problem: Dense mitosis detectors (MIDOG 2021) misinterpret benign acellular stroma as model silence, tripping false alarms.\n\n• Beer-Lambert Optical Density Deconvolution:\n   OD = -log10(I / I_0),   H = v_hematoxylin^T OD\n\n• Tissue Gating Logic: If patch cellularity H < τ_cellularity, silence monitors are bypassed physiologically.\n\n• Result: Eliminates all 14 false silence alarms across 50 whole-slide cases!"
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    # Right: PAC Bound Theorem
    c_pac = add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tf = c_pac.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Provable PAC Parameter Drift Bound"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY

    p = tf.add_paragraph()
    p.text = "• Theorem (Output Probability Drift Bound):\n   Clamping affine parameter displacement on terminal normalization layers:\n   ||θ_t - θ_0||_inf <= ε = 0.05\n\n   Strictly bounds the maximum probability shift:\n   |σ(z_t) - σ(z_0)| <= (1/4) ||w_L||_1 [ ε |x - μ|/σ + ε ] = 0.0460\n\n• Guarantees fail-closed safety: Even under corrupted streams, predictions cannot deviate significantly from certified source bounds."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    add_speaker_note(s8,
        "In dense object detection like MIDOG, benign stroma naturally contains zero mitoses. Standard detectors mistake "
        "this physiological silence for model collapse. We introduce the Cellularity-Mitosis Ratio using Beer-Lambert optical "
        "density to silence monitors during stroma traversal, eliminating 14 false alarms. Furthermore, we prove that bounding "
        "affine drift to 0.05 guarantees that the maximum output probability shift can never exceed 0.0460."
    )

    # =========================================================
    # SLIDE 9: EXPERIMENTAL SETUP
    # =========================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Multicenter Evaluation Setup with Zero Target Leakage Protocol", "8. Experimental Setup")
    add_footer(s9, 9)

    c_d1 = add_card(s9, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tf = c_d1.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Camelyon17-WILDS (Classification)"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Task: Sentinel lymph node metastasis classification\n• Total Patches: 455,954 patches (96x96 px) across 5 centers\n• Source Training: Centers 0, 3, 4 (n = 302,436 patches)\n• In-Domain Validation: Centers 0, 3, 4 (n = 33,560 patches)\n• OOD Tuning: Center 1 (n = 34,904 patches) -> Strict Zero Leakage!\n• Target Test Cohort: Center 2 (n = 85,054 patches, 10 WSIs)\n• Backbone: ResNet-18, Dropout 0.5, Adam lr=1e-5"
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    c_d2 = add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tf = c_d2.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "MIDOG 2021 (Mitosis Detection)"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Task: Mitotic figure detection under scanner domain shift\n• Dataset: 50 whole-slide cases, 1,100 patches (512x512 px)\n• Target Domain: Held-out Scanner 3\n• Architecture: Faster R-CNN ResNet-50-FPN\n• Strict 5-Seed Protocol: Seeds {42, 123, 2024, 7, 99}\n• Baselines: Source-Only, Tent, EATA, SAR, Matched Random Controls\n• Hardware: Audited on Tesla T4 and RTX 4090 GPUs"
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(8)

    add_speaker_note(s9,
        "To ensure absolute credibility, we evaluated SafeAdapt across two major multicenter benchmarks: "
        "Camelyon17-WILDS with over 455,000 patches across 5 clinical centers, and MIDOG 2021 with 50 cases across optical "
        "scanners. Crucially, we enforce a zero-target-leakage protocol: all thresholds were frozen exclusively on Hospital 1 "
        "before evaluating on unseen Hospital 2."
    )

    # =========================================================
    # SLIDE 10: MAIN RESULTS (TABLE 2 BENCHMARKS)
    # =========================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "SafeAdapt Achieves Superior AUROC, Cuts ECE by 43%, and Slashes Harm >100x", "9. Primary Results")
    add_footer(s10, 10)

    # Add Table
    rows, cols = 7, 5
    table_shape = s10.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(6.8), Inches(3.6))
    table = table_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.3)
    table.columns[3].width = Inches(1.2)
    table.columns[4].width = Inches(1.2)

    data = [
        ["Method", "AUROC ↑", "F1 Score ↑", "ECE ↓", "Harm Rate ↓"],
        ["Source-Only", "0.930 ± 0.009", "0.807 ± 0.025", "0.128 ± 0.026", "—"],
        ["Tent (ICLR'21)", "0.698 ± 0.324", "0.506 ± 0.411", "0.284 ± 0.179", "0.187 ± 0.179"],
        ["EATA (ICML'22)", "0.921 ± 0.026", "0.809 ± 0.031", "0.152 ± 0.023", "0.025 ± 0.015"],
        ["Random-Gated", "0.697 ± 0.304", "0.487 ± 0.385", "0.298 ± 0.166", "0.194 ± 0.170"],
        ["Confidence-Gated", "0.936 ± 0.022", "0.831 ± 0.032", "0.130 ± 0.030", "0.007 ± 0.003"],
        ["SafeAdapt (Ours)", "0.9495 ± 0.013", "0.842 ± 0.031", "0.075 ± 0.005", "0.0018 ± 0.0003"]
    ]

    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = C_WHITE
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            elif r_idx == 6:
                p.font.bold = True
                p.font.color.rgb = C_GREEN
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(240, 253, 244)
            else:
                p.font.color.rgb = C_TEXT

    # Highlights box below table
    c_hl = add_card(s10, Inches(0.8), Inches(5.4), Inches(6.8), Inches(1.3), bg_color=RGBColor(240, 253, 244), border_color=C_GREEN)
    tf = c_hl.text_frame
    tf.margin_left = tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "STATISTICAL PROOF: 5/5 POSITIVE SEEDS"
    p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_GREEN
    p = tf.add_paragraph()
    p.text = "ΔF1 vs Source: +0.041, +0.035, +0.038, +0.028, +0.022 (Paired Wilcoxon p = 0.0312).\nHarmful adaptation slashed from 18.7% down to 0.18% (>100x safety improvement)."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT; p.space_before = Pt(3)

    # Right Figure: AUROC / ECE Benchmark
    fig_bm = os.path.join('paper', 'ieee', 'figs', 'fig_v2_benchmark_auroc_ece.png')
    if os.path.exists(fig_bm):
        s10.shapes.add_picture(fig_bm, Inches(7.9), Inches(1.6), width=Inches(4.6))

    add_speaker_note(s10,
        "Looking at our primary results in Table 2: SafeAdapt achieves 0.9495 AUROC, outperforming the source model, "
        "Tent, and EATA. Notice that our F1 improvements are positive across all 5 seeds, yielding a statistically "
        "significant Wilcoxon p-value of 0.0312. Crucially, harmful adaptation drops from 18.7% down to 0.18%—an over "
        "100-fold reduction in clinical harm."
    )

    # =========================================================
    # SLIDE 11: ABLATION STUDY (TAYLOR TRAP)
    # =========================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Ablation Study: Sample Identity and Mathematical Disproof of Taylor Trap", "10. Ablation Analysis")
    add_footer(s11, 11)

    c_ab1 = add_card(s11, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.1))
    tf = c_ab1.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "1. Random Gating Control"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Matched-coverage control (randomly dropping 30% of patches) collapses on Seeds 7 and 99 (AUROC ~ 0.35).\n• Proves that gains arise from sample identity, not just taking fewer gradient steps."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(4)

    p = tf.add_paragraph()
    p.text = "2. Adaptation Step Sensitivity (k)"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_before = Pt(8)
    p = tf.add_paragraph()
    p.text = "• k = 1: AUROC = 0.977, Harm = 0.020 (Optimal)\n• k = 2: AUROC = 0.956\n• k >= 5: AUROC collapses to 0.375!\n• Single-step adaptation is strictly necessary to prevent parameter drift."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(4)

    p = tf.add_paragraph()
    p.text = "3. The Binary Taylor Trap"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_RED; p.space_before = Pt(8)
    p = tf.add_paragraph()
    p.text = "• Sigmoid Taylor expansion: Var[p] ≈ p^2 (1 - p)^2 σ_z^2\n• Spearman correlation: ρ = -0.9917 between MC-Dropout variance and softmax confidence!\n• MC-Dropout incurs a 3.55x latency penalty with zero added discriminatory power."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(4)

    # Right Figure: Uncertainty / Error correlation
    fig_u = os.path.join('paper', 'ieee', 'figs', 'fig_u_error.png')
    if os.path.exists(fig_u):
        s11.shapes.add_picture(fig_u, Inches(7.0), Inches(1.6), width=Inches(5.5))

    add_speaker_note(s11,
        "Our ablations revealed two vital insights. First, when we ran a random-gated control with matched 70% coverage, "
        "it collapsed on the exact same seeds as Tent, proving that sample identity matters, not just taking fewer steps. "
        "Second, we addressed the common belief that MC-Dropout uncertainty provides orthogonal information. Through Taylor "
        "expansion and empirical evaluation, we proved a Spearman correlation of negative 0.9917 with softmax confidence."
    )

    # =========================================================
    # SLIDE 12: QUALITATIVE CASE STUDY
    # =========================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Qualitative Case Study: Slide 29 Restoration and MIDOG Stroma Gating", "11. Qualitative Evidence")
    add_footer(s12, 12)

    c_case1 = add_card(s12, Inches(0.8), Inches(1.6), Inches(5.8), Inches(2.4))
    tf = c_case1.text_frame
    tf.margin_left = tf.margin_top = Inches(0.25)
    p = tf.paragraphs[0]
    p.text = "Case 1: Severe Stain Outlier (Slide 29)"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Slide 29 exhibits dark purple hematoxylin precipitation from Center 2.\n• Unconstrained Tent collapses logits to -∞, missing 100% of micro-metastases.\n• SafeAdapt-OT restores feature margins, elevating AUROC from 0.0521 to 0.9324 ± 0.0078."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    c_case2 = add_card(s12, Inches(0.8), Inches(4.3), Inches(5.8), Inches(2.4))
    tf = c_case2.text_frame
    tf.margin_left = tf.margin_top = Inches(0.25)
    p = tf.paragraphs[0]
    p.text = "Case 2: MIDOG Mitosis Detection Across Scanners"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• On Target Scanner 3, tumor stroma contains zero mitoses.\n• Baseline silence monitors trigger 14 false alarm rollbacks.\n• CMR tissue gating measures optical density H, recognizes stroma physiologically, and lifts F1: 0.5172 ± 0.0012 (p < 0.001)."
    p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    # Right Figure: MIDOG Mitosis Detection
    fig_cmr = os.path.join('paper', 'ieee', 'figs', 'fig_v2_cmr_mitosis_detection.png')
    if os.path.exists(fig_cmr):
        s12.shapes.add_picture(fig_cmr, Inches(7.0), Inches(1.6), width=Inches(5.5))

    add_speaker_note(s12,
        "On Slide 29, which suffered severe dark purple over-staining, unconstrained Tent lost all diagnostic sensitivity. "
        "SafeAdapt-OT completely restored feature localization, maintaining an AUROC of 0.9324. Similarly, on MIDOG, CMR optical "
        "gating prevented false alarm rollbacks across benign stroma, lifting mitosis detection F1 to 0.517."
    )

    # =========================================================
    # SLIDE 13: DISCUSSION: WHY IT WORKS
    # =========================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Discussion: Why SafeAdapt Succeeds Where Prior TTA Methods Fail", "12. Discussion")
    add_footer(s13, 13)

    pillars = [
        ("1. Geometric Balance", "Entropic Optimal Transport replaces unconstrained gradient descent with exact marginal transportation. By constraining Q^T 1_B = r, the network is physically unable to minimize loss by seeking trivial single-class attractors.", C_NAVY),
        ("2. Biological Grounding", "Rather than treating histology images as abstract RGB pixels, Beer-Lambert optical density deconvolution anchors adaptation in true physical staining properties, discriminating physiological acellularity from sensor collapse.", C_BLUE),
        ("3. Fail-Closed Runtime Perimeter", "PAC L_inf parameter trust regions (Δp <= 0.0460) and 1.8 ms dual-buffer atomic rollbacks ensure that even adversarial target streams cannot degrade the model below its certified baseline source performance.", C_GREEN)
    ]

    for idx, (p_title, p_desc, col) in enumerate(pillars):
        c = add_card(s13, Inches(0.8 + idx * 4.0), Inches(1.6), Inches(3.7), Inches(5.1))
        tf = c.text_frame
        tf.margin_left = tf.margin_top = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = p_title
        p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph()
        p.text = p_desc
        p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(12)

    add_speaker_note(s13,
        "To synthesize: SafeAdapt succeeds because it replaces unconstrained gradient descent with geometric optimal transport, "
        "roots adaptation in physical tissue properties, and enforces a fail-closed runtime perimeter. It proves that test-time "
        "adaptation does not have to be an unstable gamble."
    )

    # =========================================================
    # SLIDE 14: LIMITATIONS
    # =========================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Honest Limitations and Methodological Boundary Conditions", "13. Limitations")
    add_footer(s14, 14)

    limits = [
        ("Multi-Step Drift Sensitivity", "SafeAdapt requires single-step adaptation (k = 1). Setting k >= 5 triggers representation collapse even with selective gating, as accumulated affine drift overpowers the margin."),
        ("Class Prior Estimation", "Optimal transport requires a coarse estimate of the target class marginal prior (r). While robust to ±15% estimation errors, radical unforeseen prior shifts require adaptive estimation."),
        ("Batch Size Lower Bound", "Optimal Transport Sinkhorn alignment operates over mini-batches (B >= 32) to compute empirical margins; single-sample (B = 1) streaming requires running buffer memory."),
        ("Architectural Focus", "Evaluated rigorously on ResNet-18 (Camelyon17) and Faster R-CNN R50-FPN (MIDOG 2021). Deeper Vision Transformers (e.g., UNI, Virchow) are verified in code but require further compute.")
    ]

    for idx, (title, desc) in enumerate(limits):
        col_pos = idx % 2
        row_pos = idx // 2
        c = add_card(s14, Inches(0.8 + col_pos * 6.0), Inches(1.6 + row_pos * 2.6), Inches(5.7), Inches(2.3), bg_color=RGBColor(254, 242, 242), border_color=C_RED)
        tf = c.text_frame
        tf.margin_left = tf.margin_top = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = f"• {title}"
        p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_RED
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(6)

    add_speaker_note(s14,
        "In the spirit of scientific transparency, we must note our limitations: SafeAdapt requires single-step updates; "
        "setting k greater than or equal to 5 induces drift. It also requires a coarse estimate of the target class marginal prior, "
        "and operates best with mini-batches of at least 32 patches."
    )

    # =========================================================
    # SLIDE 15: FUTURE WORK
    # =========================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Future Work: Scaling SafeAdapt to Next-Generation Pathology AI", "14. Future Roadmap")
    add_footer(s15, 15)

    c_fw1 = add_card(s15, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tf = c_fw1.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Technical Roadmap"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Pathology Foundation Models:\n   Extend SafeAdapt-OT to gigapixel vision transformers (UNI, Prov-GigaPath, Virchow) to adapt self-supervised patch tokens.\n\n• Online Prior Estimation:\n   Implement Bayesian Dirichlet-multinomial tracking to dynamically infer target marginals r_t during continuous streaming.\n\n• Multi-Organ Generalization:\n   Validate across colorectal, prostate, and breast biopsies."
    p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(10)

    c_fw2 = add_card(s15, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1))
    tf = c_fw2.text_frame
    tf.margin_left = tf.margin_top = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "Clinical Translation"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_NAVY
    p = tf.add_paragraph()
    p.text = "• Scanner-Side Firmware Integration:\n   Partnering to embed SafeAdapt directly into whole-slide scanner processing engines (e.g., Philips, Hamamatsu) for real-time inference during image acquisition.\n\n• Multiplex Staining Pipelines:\n   Evaluate adaptation on immunohistochemical (IHC) markers and multiplex fluorescence cohorts.\n\n• Regulatory Engagement:\n   Formulating certified safety dossiers for FDA SaMD clearance."
    p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(10)

    add_speaker_note(s15,
        "Looking ahead, we are extending SafeAdapt to gigapixel vision transformer foundation models such as UNI and Prov-GigaPath, "
        "formulating dynamic Bayesian prior updating, and partnering to embed SafeAdapt directly into scanner firmware for "
        "real-time inference during slide digitization."
    )

    # =========================================================
    # SLIDE 16: CONCLUSION
    # =========================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Conclusion: Transforming TTA into a Certified Clinical Reality", "15. Conclusion")
    add_footer(s16, 16)

    c_conc = add_card(s16, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.1), bg_color=RGBColor(240, 253, 244), border_color=C_GREEN)
    tf = c_conc.text_frame
    tf.margin_left = tf.margin_top = Inches(0.4)
    p = tf.paragraphs[0]
    p.text = "SUMMARY OF RESEARCH ACHIEVEMENTS"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_GREEN

    items = [
        ("The Discovery:", "Unconstrained entropy minimization collapses on 40% of random seeds in digital pathology, and MC-Dropout uncertainty is mathematically redundant with softmax confidence (ρ = -0.9917)."),
        ("The Solution:", "SafeAdapt couples Entropic Optimal Transport marginal constraints (Q^T 1_B = r) with Beer-Lambert tissue gating and PAC L_inf parameter trust regions."),
        ("The Clinical Impact:", "Elevates 5-seed cohort performance to 0.9495 AUROC (p = 0.0312), completely reverses collapse on outlier Slide 29 (0.0521 -> 0.9324), eliminates 14 false alarms on MIDOG, and slashes harmful adaptation by >100x (0.18%)."),
        ("The Paradigm Shift:", "SafeAdapt proves that test-time adaptation can transition from an unpredictable empirical gamble into a mathematically certified, clinically deployable reality.")
    ]

    for label, text in items:
        p = tf.add_paragraph()
        p.text = f"• {label} {text}"
        p.font.size = Pt(13); p.font.color.rgb = C_TEXT; p.space_before = Pt(12)

    add_speaker_note(s16,
        "In conclusion: We demonstrated that unconstrained TTA collapses on 40% of pathology seeds; we disproved the utility "
        "of MC-Dropout in 1D heads; and we delivered SafeAdapt, which achieves 0.9495 AUROC, rescues outlier slides, and slashes "
        "harmful adaptation by over 100-fold."
    )

    # =========================================================
    # SLIDE 17: KEY TAKEAWAYS
    # =========================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Three Key Takeaways for Machine Learning Researchers and Clinicians", "16. Key Takeaways")
    add_footer(s17, 17)

    takeaways = [
        ("1. Attractor Collapse is a Solvable Geometric Defect",
         "Entropy minimization collapses because it lacks marginal constraints; Entropic Optimal Transport guarantees balanced class allocation by solving Q^T 1_B = r.",
         C_NAVY),
        ("2. MC-Dropout Uncertainty is Unnecessary in 1D Heads",
         "First-order Taylor expansion proves predictive variance is rank-equivalent to softmax confidence (ρ = -0.9917). Eliminate it to save 3.55x computational inference penalty.",
         C_BLUE),
        ("3. Fail-Closed Safeguards are Mandatory for Patient Safety",
         "Parameter displacement bounding (||Δθ||_inf <= 0.05), Beer-Lambert cellularity gating, and episodic per-WSI resets are non-negotiable requirements for patient diagnostic isolation.",
         C_GREEN)
    ]

    for idx, (title, desc, col) in enumerate(takeaways):
        c = add_card(s17, Inches(0.8), Inches(1.6 + idx * 1.7), Inches(11.7), Inches(1.4))
        tf = c.text_frame
        tf.margin_left = tf.margin_top = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = col
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(12); p.font.color.rgb = C_TEXT; p.space_before = Pt(4)

    add_speaker_note(s17,
        "Please leave today with these three insights: Attractor collapse is an avoidable geometric defect solved by optimal transport; "
        "MC-Dropout is unnecessary in 1D classification; and fail-closed safeguards are mandatory for trustworthy clinical deployment."
    )

    # =========================================================
    # SLIDE 18: THANK YOU & Q&A
    # =========================================================
    s18 = prs.slides.add_slide(blank_layout)
    bg18 = s18.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg18.fill.solid()
    bg18.fill.fore_color.rgb = C_NAVY
    bg18.line.fill.background()

    # Title box
    t_box18 = s18.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.7), Inches(1.6))
    tf18 = t_box18.text_frame
    tf18.word_wrap = True
    p18 = tf18.paragraphs[0]
    p18.text = "Thank You! Questions & Discussion"
    p18.font.size = Pt(32); p18.font.bold = True; p18.font.color.rgb = C_WHITE

    p18_sub = tf18.add_paragraph()
    p18_sub.text = "SafeAdapt: Clinically Safe Test-Time Adaptation in Digital Pathology"
    p18_sub.font.size = Pt(16); p18_sub.font.color.rgb = RGBColor(56, 189, 248); p18_sub.space_before = Pt(8)

    # Left: Contacts
    c_ct = add_card(s18, Inches(0.8), Inches(3.0), Inches(5.7), Inches(3.5), bg_color=RGBColor(23, 58, 114), border_color=RGBColor(38, 78, 142))
    tf = c_ct.text_frame
    tf.margin_left = tf.margin_top = Inches(0.35)
    p = tf.paragraphs[0]
    p.text = "RESEARCH TEAM & CONTACTS"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = RGBColor(56, 189, 248)
    p = tf.add_paragraph()
    p.text = "Avinash Reddy Cheerapareddy\navinashreddy_cheerapareddy@srmap.edu.in\n\nGunnam Sanjana\nsanjana_gunnam@srmap.edu.in\n\nDepartment of Computer Science and Engineering\nSRM University-AP, Amaravati, Andhra Pradesh, India"
    p.font.size = Pt(12); p.font.color.rgb = RGBColor(226, 232, 240); p.space_before = Pt(10)

    # Right: Open Science
    c_os = add_card(s18, Inches(6.8), Inches(3.0), Inches(5.7), Inches(3.5), bg_color=RGBColor(23, 58, 114), border_color=RGBColor(38, 78, 142))
    tf = c_os.text_frame
    tf.margin_left = tf.margin_top = Inches(0.35)
    p = tf.paragraphs[0]
    p.text = "OPEN-SOURCE REPRODUCIBILITY"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = RGBColor(56, 189, 248)
    p = tf.add_paragraph()
    p.text = "• Official Zenodo Archive:\n   https://doi.org/10.5281/zenodo.23658430\n\n• GitHub Repository:\n   https://github.com/Avirech7676/UTTA-Med\n\n• Bit-Exact Verification: 57/57 Passed\n• Automated Pytest Suite: 148 Passed Unit Tests"
    p.font.size = Pt(12); p.font.color.rgb = RGBColor(226, 232, 240); p.space_before = Pt(10)

    add_speaker_note(s18,
        "Thank you all for your attention. Our full dataset, models, and code are archived on Zenodo and GitHub. "
        "My co-author Gunnam Sanjana and I now welcome your questions."
    )

    out_path = os.path.join('paper', 'SafeAdapt_Google_Slides.pptx')
    prs.save(out_path)
    print(f"SUCCESS: Created 16:9 widescreen presentation: {out_path} ({os.path.getsize(out_path)} bytes).")

if __name__ == '__main__':
    create_deck()
