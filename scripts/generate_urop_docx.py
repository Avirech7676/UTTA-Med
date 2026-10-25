"""Script to generate a publication-grade Word document (.docx) for the UROP Research Monograph."""

import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_h(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    if level == 1:
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(5)
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(27, 54, 93) # Navy
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="4" w:color="1B365D"/></w:pBdr>')
        p._p.get_or_add_pPr().append(pBdr)
    elif level == 2:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run.font.size = Pt(12.5)
        run.font.color.rgb = RGBColor(46, 97, 113) # Teal
    elif level == 3:
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(3)
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(50, 50, 50)
    return p

def add_p(doc, text="", bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.color.rgb = RGBColor(30, 30, 30)
    if text:
        r_text = p.add_run(text)
        r_text.font.color.rgb = RGBColor(40, 40, 40)
    return p

def add_callout(doc, text, title="NOTE", border_color="1B365D", fill_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, fill_color)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="20" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.size = Pt(9.5)
    r_title.font.color.rgb = RGBColor(27, 54, 93)
    
    r_txt = p.add_run(text)
    r_txt.font.size = Pt(9.5)
    r_txt.font.color.rgb = RGBColor(40, 40, 40)
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(4)

def add_code(doc, code_str):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="12" w:space="0" w:color="94A3B8"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(code_str)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(30, 41, 59)
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(4)

def format_custom_table(tbl, col_widths, headers, data, align_cols=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="CBD5E1", sz="4")
    
    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        cell = hdr_cells[i]
        cell.width = Inches(col_widths[i])
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        if align_cols and align_cols[i] == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align_cols and align_cols[i] == "right":
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    
    for r_idx, row_data in enumerate(data):
        row = tbl.add_row()
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        bg_color = "F8FAFC" if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            
            if align_cols and align_cols[c_idx] == "center":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif align_cols and align_cols[c_idx] == "right":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            run = p.add_run(val)
            run.font.size = Pt(8.0)
            if "SUPPORTED" in val or "Stable" in val or "PASSED" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(21, 128, 61)
            elif "CONTRADICTED" in val or "COLLAPSED" in val or "Collapsed" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(185, 28, 28)
            else:
                run.font.color.rgb = RGBColor(40, 40, 40)

def generate_urop_monograph():
    doc = docx.Document()
    
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
        p_hdr = s.header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_hdr = p_hdr.add_run("UROP Monograph | SRM University-AP | SafeAdapt-TTA")
        r_hdr.font.size = Pt(8.5)
        r_hdr.font.color.rgb = RGBColor(120, 120, 120)
        
        p_ftr = s.footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_ftr = p_ftr.add_run("SRM University-AP, Amaravathi, Andhra Pradesh, India — Department of Computer Science and Engineering")
        r_ftr.font.size = Pt(8.5)
        r_ftr.font.color.rgb = RGBColor(120, 120, 120)

    # Title Block
    p_t = doc.add_paragraph()
    p_t.paragraph_format.space_before = Pt(0)
    p_t.paragraph_format.space_after = Pt(2)
    r_title = p_t.add_run("Selective Test-Time Adaptation and Runtime Safeguarding for Histopathology under Cross-Hospital Domain Shift")
    r_title.bold = True
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("A Longitudinal Reliability Framework and Multi-Seed Investigation on Camelyon17-WILDS")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(46, 97, 113)

    # Author & Affiliation Table
    tbl_auth = doc.add_table(rows=4, cols=2)
    tbl_auth.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_auth, color="E2E8F0", sz="4")
    col_w = [1.8, 4.7]
    auth_data = [
        ("Author", "Avinash Reddy Cheerapareddy"),
        ("Affiliation", "Department of Computer Science and Engineering\nSRM University-AP, Amaravathi, Andhra Pradesh, India"),
        ("Program & Date", "Undergraduate Research Opportunities Program (UROP) | October 2026"),
        ("Correspondence", "avinashreddy.crp@gmail.com / cheerapareddy.avinash@alumni.ee")
    ]
    for idx, (lbl, val) in enumerate(auth_data):
        row = tbl_auth.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(col_w[0])
        c1.width = Inches(col_w[1])
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=50, bottom=50, left=100, right=100)
        set_cell_margins(c1, top=50, bottom=50, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(lbl)
        r0.bold = True
        r0.font.size = Pt(9)
        r0.font.color.rgb = RGBColor(27, 54, 93)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        p1.paragraph_format.line_spacing = 1.1
        r1 = p1.add_run(val)
        r1.font.size = Pt(9)
        r1.font.color.rgb = RGBColor(30, 30, 30)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(8)
    p_sp.paragraph_format.space_after = Pt(0)

    # Abstract Box
    add_callout(doc,
        "Deep learning models deployed in computational histopathology face severe out-of-distribution performance degradation when transferred across clinical healthcare centers. "
        "Unsupervised Test-Time Adaptation (TTA) via entropy minimization (Tent) updates parameters online using unlabeled target batches. However, in our five-seed study on Camelyon17-WILDS, "
        "unconstrained Tent exhibits catastrophic instability: collapsing completely on 2 of 5 seeds (AUROC dropping from 0.930 to 0.338, sensitivity < 3%). "
        "We introduce SafeAdapt-TTA, coupling hard selective uncertainty gating with strict parameter displacement bounding (||Delta theta||_inf <= 0.05), numerical gradient checks, atomic rollback, "
        "and whole-slide boundary resets. We discover that in single-logit binary classification, Monte Carlo dropout predictive variance and max-softmax confidence induce near-identical patch rankings (rho = -0.9917), "
        "enabling deterministic confidence gating to match MC-dropout accuracy while eliminating a 3.55x latency penalty. Selective gating reduces harmful adaptation ~25-fold (from 18.7% to 0.7%) and consistently improves F1 (+0.023, 95% CI [0.010, 0.036]). "
        "Crucially, under frozen pre-adaptation streaming evaluation on an external target domain (Hospital 2; 10 WSIs, 1,280 patches), adaptation yielded slight negative transfer (Macro AUROC 0.9308 -> 0.9300, F1 0.7840 -> 0.7822) "
        "due to confirmation bias on outlier slides. We establish a scientific firewall delineating what the findings establish, backed by a 120-test automated regression suite.",
        title="ABSTRACT", border_color="1B365D", fill_color="F8FAFC"
    )
    add_p(doc, "Keywords: Test-Time Adaptation, Digital Pathology, Camelyon17-WILDS, Domain Shift, Uncertainty Estimation, Model Reliability, SafeAdapt, Confirmation Bias, Longitudinal Evaluation.", bold_prefix="Keywords: ")

    # Section 1: Introduction
    add_h(doc, "1 Introduction", 1)
    add_h(doc, "1.1 Background", 2)
    add_p(doc, 
        "Digital pathology is undergoing rapid clinical transformation, driven by high-throughput whole-slide optical scanners and deep convolutional and transformer architectures for computer-aided cancer diagnosis, grading, and staging (Bandi et al., 2018). "
        "In clinical oncology, identifying metastatic breast adenocarcinoma foci within sentinel lymph node sections is essential for assigning pathological tumor-node-metastasis (pTNM) stages, which directly dictate whether patients undergo adjuvant chemotherapy, regional radiation, or axillary lymph node dissection. "
        "However, deep neural networks trained in a controlled host institution frequently suffer catastrophic performance degradation when deployed across external hospital networks (Koh et al., 2021). "
        "Variations in microtome sectioning thickness, slide mounting reagents, hematoxylin and eosin (H&E) dye chemical formulations, automated staining bath dwell times, and scanner optical properties introduce pervasive out-of-distribution (OOD) domain shifts."
    )
    
    add_h(doc, "1.2 Motivation", 2)
    add_p(doc, 
        "Test-Time Adaptation (TTA) has emerged as an attractive paradigm for autonomous domain generalization (Wang et al., 2021; Liang et al., 2023). "
        "By updating a subset of source model parameters (typically the scale and bias affine parameters of Batch Normalization layers) online using incoming unlabeled target batches, TTA aligns target feature statistics with source representations without accessing historical training data. "
        "However, medical imaging data violates key assumptions of stationary natural vision: extreme tissue heterogeneity, severe class imbalance (normal stroma vastly outnumbers micrometastases), and biological artifacts."
    )

    add_h(doc, "1.3 Problem Statement", 2)
    add_p(doc, 
        "Despite theoretical promises, applying unsupervised entropy minimization to clinical histopathology reveals two severe vulnerabilities:\n"
        "1. Representation Collapse & Attractor States: Under unconstrained entropy minimization, neural networks rapidly converge to trivial single-class degenerate solutions (predicting 100% of tissue as normal stroma) to achieve mathematical zero entropy.\n"
        "2. Confirmation Bias & Error Amplification: Under real clinical stain shifts, models frequently output high-confidence false-positive predictions. Minimizing entropy on erroneous confident predictions reinforces decision boundary errors.\n"
        "3. Computational Inefficiency: Evaluating 20 stochastic Monte Carlo dropout passes per patch imposes an unsustainable 3.5x to 5x latency penalty in high-throughput clinical scanning workflows."
    )

    add_h(doc, "1.4 Research Questions", 2)
    add_p(doc, "This research monograph addresses the following central scientific question:")
    add_callout(doc, 
        "Can selective uncertainty gating and runtime parameter safeguarding eliminate representation collapse and harmful adaptation in digital pathology under cross-hospital domain shifts, and does unsupervised adaptation translate into positive diagnostic transfer on external clinical target cohorts?",
        title="PRIMARY RESEARCH QUESTION", border_color="2E6171", fill_color="F0FDF4"
    )
    add_p(doc, 
        "RQ1 (Collapse Dynamics): Does unconstrained entropy minimization (Tent) exhibit reproducible representation collapse under multi-seed evaluation on Camelyon17-WILDS?\n"
        "RQ2 (Harm Reduction): Does selective gating based on uncertainty thresholds calibrated on a held-out institution significantly cut harmful adaptation and improve F1 across independent seeds?\n"
        "RQ3 (Gating Equivalence & Efficiency): Does epistemic uncertainty estimation via Monte Carlo dropout provide meaningful patch-selection discrimination beyond deterministic softmax confidence in single-logit binary histopathology classification?\n"
        "RQ4 (Selection vs. Step Budget): Are stability gains attributable to the selective identification of reliable adaptation signals, or merely a trivial consequence of taking fewer gradient steps (matched-coverage random control)?\n"
        "RQ5 (Runtime Parameter Safeguards): Can a runtime controller enforcing explicit parameter displacement bounds (||Delta theta||_inf <= 0.05) and slide-boundary resets guarantee numerical stability and eliminate inter-patient contamination?\n"
        "RQ6 (External Target Transfer): Does selective test-time adaptation improve diagnostic metrics relative to a frozen source model when evaluated on an unseen external hospital under a leak-free streaming inference protocol?"
    )

    add_h(doc, "1.5 Contributions", 2)
    add_p(doc, 
        "1. Multi-Seed Empirical Demonstration of TTA Instability: Documenting that unconstrained Tent collapses completely on 2 of 5 seeds (AUROC 0.930 -> 0.338, sensitivity < 3%).\n"
        "2. SafeAdapt-TTA Architectural Controller: Designing and verifying a runtime controller integrating selective gating, numerical gradient auditing, atomic rollback, and whole-slide resets.\n"
        "3. Resolution of Binary Gating Equivalence: Mathematically and empirically proving (rho = -0.9917) that MC-dropout predictive variance and softmax confidence induce near-identical rankings, enabling compute-frugal deployment.\n"
        "4. Matched-Coverage Causal Controls: Proving that random gating still collapses, demonstrating that stability depends strictly on sample selection quality.\n"
        "5. Transparent Reporting of External Target Negative Transfer: Demonstrating on Hospital 2 that adaptation yielded slight negative transfer relative to frozen source due to confirmation bias on outlier slides.\n"
        "6. Complete Cryptographic Reproducibility Package: Releasing a 120-test automated regression suite (100% passing rate) and bit-exact table verification scripts (54/54 passed checks)."
    )

    # Section 2: Related Work & Positioning
    add_h(doc, "2 Related Work & Conceptual Positioning", 1)
    add_p(doc, 
        "Table 1 positions SafeAdapt-TTA within the landscape of related test-time adaptation, uncertainty estimation, and medical image generalization literature."
    )
    t1_headers = ["Research Line", "Key Methods", "Update Scope", "Filtering Heuristic", "Runtime Safeguards", "Target Benchmark"]
    t1_widths = [1.1, 1.0, 0.9, 1.1, 1.2, 1.2]
    t1_align = ["left", "left", "left", "left", "left", "left"]
    t1_data = [
        ["General TTA", "Tent (Wang 2021)", "BN Affine", "None (100% update)", "None", "Natural Vision (ImageNet-C)"],
        ["Point-Filtered TTA", "EATA (Niu 2022)", "BN Affine", "Point Entropy", "Fisher Regularizer", "Natural Vision"],
        ["Sharpness-Aware TTA", "SAR (Niu 2023)", "BN Affine", "Gradient Norm Threshold", "State Reset", "Natural Vision"],
        ["Bayesian Medical TTA", "CertainTTA (2025)", "Full Network", "MC Predictive Entropy", "None", "Cardiac & Fundus Seg."],
        ["Foundation Model TTA", "UAD-FM (2025)", "LoRA Adapters", "Dirichlet Evidential", "None", "Colorectal Cancer (CRC)"],
        ["SafeAdapt-TTA (Ours)", "SafeAdapt (2026)", "BN Affine", "Selective Uncertainty Gate", "Hard L_inf, Rollback, Reset", "Camelyon17-WILDS Lymph Node"]
    ]
    tbl_pos = doc.add_table(rows=1, cols=6)
    format_custom_table(tbl_pos, t1_widths, t1_headers, t1_data, t1_align)

    # Section 3: SafeAdapt Framework
    add_h(doc, "3 Conceptual and Algorithmic Framework: SafeAdapt-TTA", 1)
    add_h(doc, "3.1 Problem Formulation & Optimization Scope", 2)
    add_p(doc, 
        "Let f_theta denote a ResNet-18 model trained on source hospitals. All convolutional kernels and linear head weights are frozen. Adaptation is restricted exclusively to the BatchNorm affine parameters theta_BN = {(gamma_l, beta_l)} with frozen running statistics (mu_l, sigma_l^2). "
        "Under unconstrained Tent, the gradient of binary entropy with respect to logit z is (2p - 1)z >= 0 for all z in R. When normal patches vastly outnumber tumor patches, negative gradients dominate, driving affine bias beta_l negative and triggering catastrophic attractor collapse."
    )

    add_h(doc, "3.2 Mathematical Analysis of Binary Gating Equivalence", 2)
    add_callout(doc, 
        "LEMMA 1 (Strict Monotonicity of Binary Entropy and Confidence):\n"
        "For binary probability p in [0, 1], let confidence C(p) = max(p, 1-p) in [0.5, 1.0]. The binary Shannon entropy H(p) = -p log p - (1-p) log (1-p) is a strictly decreasing bijective function of confidence C:\n"
        "dH/dC = log((1-C)/C) < 0  for all C in (0.5, 1.0).\n\n"
        "COROLLARY 1: On single-logit binary classification, filtering by Shannon entropy, deterministic confidence, or Monte Carlo dropout predictive variance yields identical patch sample rankings (Spearman rho = -0.9917).",
        title="MATHEMATICAL THEOREM & LEMMA", border_color="1B365D", fill_color="F8FAFC"
    )

    add_h(doc, "3.3 Operational Distinction: Unconstrained TTA vs SafeAdapt-TTA", 2)
    t2_headers = ["Operational Dimension", "Unconstrained TTA (Tent)", "SafeAdapt-TTA (Ours)"]
    t2_widths = [2.0, 2.2, 2.3]
    t2_align = ["left", "left", "left"]
    t2_data = [
        ["Patch Admission Policy", "100% of incoming target patches", "Selective Hard Gate (score <= tau)"],
        ["Empty-Batch Behavior", "Always updates", "Explicit Skip (Zero backward/step)"],
        ["Parameter Scope", "Unbounded drift in R^D", "Strict L_inf Sphere (||Delta|| <= 0.05)"],
        ["Numerical Integrity", "Unchecked (NaNs corrupt model)", "Pre-step gradient & loss auditing"],
        ["State Recovery Mechanism", "None (Irreversible corruption)", "Atomic State Rollback"],
        ["Inter-Slide Isolation", "Cross-slide continuous drift", "Whole-Slide Boundary Invariance Reset"],
        ["Target Timing", "Post-adaptation (biased)", "Pre-adaptation streaming (realistic)"],
        ["Epistemic Overhead", "1 forward pass", "1 forward pass (deterministic confidence)"],
        ["Target Label Access", "Zero (Compliant)", "Zero (Strictly enforced by API)"]
    ]
    tbl_op = doc.add_table(rows=1, cols=3)
    format_custom_table(tbl_op, t2_widths, t2_headers, t2_data, t2_align)

    # Section 4: Experimental Corpus
    add_h(doc, "4 Experimental Framework & Frozen Corpus Dimensions", 1)
    add_p(doc, 
        "Table 3 details the authoritative dimensions and cryptographic SHA-256 hashes of the frozen experimental corpus."
    )
    t3_headers = ["Dataset / Artifact Name", "Split / Cohort", "Dimensions / Denominators", "Cryptographic SHA-256 Hash"]
    t3_widths = [1.8, 1.3, 1.6, 1.8]
    t3_align = ["left", "left", "left", "left"]
    t3_data = [
        ["camelyon17_resnet18_source_s42_BEST", "Source (0, 3, 4)", "11.18M params (ResNet-18)", "f0b85ec1a89707a8ad1807a544280..."],
        ["headline_5seed.json", "v1 Multi-Seed", "5 seeds x 6 methods", "9f0f0753835c5971ae9ca9394def0..."],
        ["table_main_results_final.csv", "v1 Summary Table", "6 rows x 6 metrics", "320df9cd65e84be39a4b3bbb5f440..."],
        ["table_paired_statistics_final.csv", "v1 Hypothesis Tests", "8 contrasts x 8 stats", "b7510afe90e26e6e21c0f9ce87fcb..."],
        ["stage_d2_benchmark_results.json", "Dev (Hospital 1)", "320 patches (5 WSIs)", "143ca46038968393b160ed8a7bf5c..."],
        ["stage_e1_evaluation_results.json", "Dev Streaming (H1)", "4 arms x 320 patches", "ab060f1ec9e9d725e667272e80148..."],
        ["stage_e2_target_results.json", "Target (Hospital 2)", "4 arms x 1,280 patches (10 WSIs)", "f5c90277991999ecc6ae5a63fa353..."],
        ["safeadapt_e2_frozen.yaml", "Runtime Config", "Frozen execution config", "3042225de8be5bd013a5725898a78..."]
    ]
    tbl_corp = doc.add_table(rows=1, cols=4)
    format_custom_table(tbl_corp, t3_widths, t3_headers, t3_data, t3_align)

    # Section 5: Empirical Results & Hypotheses
    add_h(doc, "5 Empirical Results & Hypotheses Verification", 1)
    add_p(doc, 
        "Table 4 synthesizes the final inferential status across all preregistered hypotheses and exploratory probes."
    )
    t4_headers = ["Hypothesis ID", "Description", "Primary Endpoint", "Effect Size & p-value", "Formal Status"]
    t4_widths = [0.9, 1.4, 1.4, 1.6, 1.2]
    t4_align = ["center", "left", "left", "left", "center"]
    t4_data = [
        ["H1", "Tent Collapse Instability", "5-seed AUROC & Sensitivity", "Delta = -0.232, 2/5 Collapsed", "SUPPORTED"],
        ["H2", "Selective Harm Reduction", "Harm Rate & Paired F1", "Harm: 18.7% -> 0.7%, d=2.26", "SUPPORTED"],
        ["H3", "Binary Gating Equivalence", "Spearman rho & Delta AUROC", "rho = -0.9917, p = 0.8688", "SUPPORTED"],
        ["H4", "Matched-Coverage Failure", "Random Gate AUROC & Harm", "AUROC: 0.697, 2/5 Collapsed", "SUPPORTED"],
        ["H5", "SafeAdapt Runtime Bounding", "Max L_inf Displacement", "Max disp = 4.01e-5 << 0.05", "SUPPORTED"],
        ["H6", "Target Positive Transfer", "Hospital 2 AUROC & F1", "Delta AUROC = -0.0008", "CONTRADICTED"],
        ["Probe 1", "Evidential Decoupling", "NCT-CRC-HE 9-class Harm", "Harm Rate = 0.0% (4 net gain)", "SUPPORTED"],
        ["Probe 2", "Multi-Scanner Detection", "MIDOG 2021 Faster R-CNN F1", "Delta F1 = +0.0134 (p < 0.05)", "SUPPORTED"]
    ]
    tbl_hyp = doc.add_table(rows=1, cols=5)
    format_custom_table(tbl_hyp, t4_widths, t4_headers, t4_data, t4_align)

    # Section 6: Discussion & Firewall
    add_h(doc, "6 Discussion & Scientific Firewall", 1)
    add_callout(doc, 
        "SCIENTIFIC FIREWALL (WHAT THE RESULTS DO NOT CLAIM):\n\n"
        "1. NOT CLAIMED: Certified Clinical Safety — SafeAdapt is an engineering controller with bounded parameter projections; it does not provide clinical non-inferiority guarantees.\n"
        "2. NOT CLAIMED: Universal SOTA AUROC Gain — Selective gating preserves discriminative accuracy; it does not establish a statistically significant AUROC gain over a frozen source model (p = 0.4199).\n"
        "3. NOT CLAIMED: Universal Uncertainty Equivalence — Monotonicity holds strictly for single-logit binary classification; in multi-class settings (K >= 3), evidential vacuity decouples from confidence.\n"
        "4. NOT CLAIMED: Clinical Deployment Readiness — The software is an investigational prototype and is strictly contra-indicated for primary clinical diagnostic use.",
        title="SCIENTIFIC FIREWALL", border_color="8A1538", fill_color="FFF1F2"
    )

    # Section 7: Reproducibility & Telemetry
    add_h(doc, "7 Reproducibility, Toolchain & Data Integrity", 1)
    add_p(doc, 
        "Environment Specifications:\n"
        "- Python Interpreter: Python 3.14.2 (Verified in conda environment 'avinash')\n"
        "- PyTorch Core: 2.14.0+cpu | Pytest: 9.1.1\n"
        "- Automated Regression Suite: 120 passed, 2 skipped, 0 failed in 11.16s\n"
        "- Table Consistency Audit: 54 of 54 checks passed (100% bit-exact consistency)"
    )

    # Section 8: Conclusion
    add_h(doc, "8 Conclusion", 1)
    add_p(doc, 
        "Unsupervised test-time adaptation via unconstrained entropy minimization is unstable in computational pathology, suffering catastrophic representation collapse under multi-seed evaluation. "
        "SafeAdapt-TTA resolves gross instability through selective uncertainty gating, cutting harmful adaptations ~25-fold and consistently improving paired F1. "
        "In single-logit binary classification, deterministic confidence gating matches MC-dropout predictive variance while saving 3.55x computational overhead. "
        "However, rigorous evaluation on an external clinical target domain reveals that parameter-bounding safeguards cannot convert negative adaptation dynamics into positive diagnostic transfer on outlier slides. "
        "These findings underscore the critical necessity of multi-seed reporting, slide-level inferential units, and rigorous safeguarding before considering test-time adaptation for clinical deployment."
    )

    out_1 = "./UROP_Research_Draft_UTTA_Med.docx"
    doc.save(out_1)
    print(f"Successfully generated: {out_1}")

    out_2 = "str(Path.home() / "Downloads")/UROP_Research_Draft_UTTA_Med.docx"
    try:
        shutil.copyfile(out_1, out_2)
        print(f"Successfully copied to: {out_2}")
    except Exception as e:
        print(f"Note: Could not copy to Downloads: {e}")

if __name__ == "__main__":
    generate_urop_monograph()
