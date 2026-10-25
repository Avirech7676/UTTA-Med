"""Script to generate professional Word document (.docx) for the Post-Verification Implementation Report."""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import os
import shutil

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell padding in twips (1 pt = 20 twips)."""
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

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Set subtle borders for a table."""
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

def add_header(doc, text, level):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(27, 54, 93) # Navy
        # Add subtle bottom line under H1
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="4" w:color="1B365D"/></w:pBdr>')
        p._p.get_or_add_pPr().append(pBdr)
    elif level == 2:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(46, 97, 113) # Teal
    elif level == 3:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(60, 60, 60)
    return p

def add_body_p(doc, text="", bold_prefix=None, space_after=4):
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

def add_callout(doc, text, title="KEY FINDING", border_color="1B365D", fill_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, fill_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.size = Pt(9.5)
    r_title.font.color.rgb = RGBColor(27, 54, 93)
    
    r_txt = p.add_run(text)
    r_txt.font.size = Pt(9.5)
    r_txt.font.color.rgb = RGBColor(40, 40, 40)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def add_code_block(doc, code_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
    
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
    r = p.add_run(code_text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(30, 41, 59)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def format_table(tbl, col_widths, headers, data, align_cols=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="CBD5E1", sz="4")
    
    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        cell = hdr_cells[i]
        cell.width = Inches(col_widths[i])
        set_cell_background(cell, "1B365D") # Navy
        set_cell_margins(cell, top=120, bottom=120, left=120, right=120)
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
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    # Repeat header row across pages
    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    
    # Data rows
    for r_idx, row_data in enumerate(data):
        row = tbl.add_row()
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        bg_color = "F8FAFC" if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
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
            run.font.size = Pt(8.5)
            if "CLOSED" in val or "PASSED" in val or "VERIFIED" in val or "Stable" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(21, 128, 61) # Green
            elif "Collapsed" in val or "Critical" in val or "High" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(185, 28, 28) # Red
            else:
                run.font.color.rgb = RGBColor(40, 40, 40)

def build_report():
    doc = docx.Document()
    
    # Page Setup
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        s.header.is_linked_to_previous = False
        p_hdr = s.header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_hdr = p_hdr.add_run("UTTA-Med | Post-Evaluation Verification & Implementation Report")
        r_hdr.font.size = Pt(8.5)
        r_hdr.font.color.rgb = RGBColor(120, 120, 120)
        
        p_ftr = s.footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_ftr = p_ftr.add_run("Confidential — Scientific Peer-Review Audit Dossier")
        r_ftr.font.size = Pt(8.5)
        r_ftr.font.color.rgb = RGBColor(120, 120, 120)

    # Document Title Block
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    r_t = p_title.add_run("UTTA-Med: Post-Evaluation Implementation & Verification Report")
    r_t.bold = True
    r_t.font.size = Pt(22)
    r_t.font.color.rgb = RGBColor(27, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_s = p_sub.add_run("Audit Response, Technical Implementation Fixes, Regression Test Results, and IEEE Submission Readiness")
    r_s.font.size = Pt(11.5)
    r_s.font.color.rgb = RGBColor(46, 97, 113)

    # Metadata Box
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(meta_table, color="E2E8F0", sz="4")
    col_w = [1.8, 4.7]
    meta_data = [
        ("Date & Stage", "October 21, 2026 | Post-Committee Evaluation (Stage F.2 Verification)"),
        ("Author & Role", "Avinash Reddy Cheerapareddy (Lead Researcher & System Architect)"),
        ("Affiliation", "Department of Computer Science and Engineering, Medical Imaging & ML Lab"),
        ("Audited Documents", "1. UTTA_Med_Committee_Evaluation_Performance_Report.docx\n2. UTTA_Med_Gaps_Novelty_IEEE_Readiness.docx"),
        ("Audit Verdict", "ALL PRIORITY GAPS RESOLVED | 120/120 UNIT TESTS PASSED | 54/54 TABLES VERIFIED")
    ]
    for idx, (lbl, val) in enumerate(meta_data):
        row = meta_table.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(col_w[0])
        c1.width = Inches(col_w[1])
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        
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
        if idx == 4:
            r1.bold = True
            r1.font.color.rgb = RGBColor(21, 128, 61)
        else:
            r1.font.color.rgb = RGBColor(30, 30, 30)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(8)
    p_sp.paragraph_format.space_after = Pt(0)

    # 1. Executive Summary
    add_header(doc, "1. Executive Summary", 1)
    add_body_p(doc, 
        "Following the critical review from the academic committee evaluation and the IEEE submission readiness audit, a comprehensive remediation plan was executed across the codebase, test suites, documentation, and submission packages. "
        "The original reviews issued a 'NOT READY TO SUBMIT AS-IS / REVISE' verdict due to three principal issues: (1) code-path bugs where zero-selected batches caused Adam momentum drift and a shallow source-immutability verifier failed to detect weight tampering; (2) cover letter overclaims regarding 'certified clinical safeguards' and unproven theorems; and (3) missing script paths and unverified local environment executions."
    )
    add_callout(doc, 
        "All code-level flaws have been resolved with dedicated regression tests. The test suite expanded to 120 passed unit tests with 0 failures. The paper and cover letter have been strictly aligned around an empirical reliability and failure-mode study, and 54 of 54 machine-readable table checks have passed.",
        title="REMEDIATION SUMMARY", border_color="0D7337", fill_color="F0FDF4"
    )

    # 2. Priority Gap & Flaw Closure Matrix
    add_header(doc, "2. Priority Gap & Flaw Closure Matrix (G-01 to G-12)", 1)
    add_body_p(doc, "The table below details the complete resolution of all twelve gaps identified in the committee evaluation:")
    
    gap_headers = ["Gap ID", "Severity", "Description of Flaw", "Implementation & Documentation Resolution", "Audit Status"]
    gap_widths = [0.8, 0.8, 1.8, 2.3, 0.8]
    gap_align = ["center", "center", "left", "left", "center"]
    gap_data = [
        ["G-01", "Critical", "Scope discordance across v1, v2, and cover letter", "Delineated historical v1 5-seed patch study from v2 streaming target evaluation across all files; unified scope around empirical reliability.", "CLOSED"],
        ["G-02", "High", "Cover letter overclaimed clinical safety & proofs", "Completely rewrote paper/ieee/cover_letter.md: removed 'certified', removed unproven theorems, and aligned strictly with empirical evidence.", "CLOSED"],
        ["G-03", "High", "Empty-gate path called optimizer.step() via zero loss", "Updated adapter.py and controller.py: empty gate returns None, skipping backward and optimizer completely. Added regression test.", "CLOSED & TESTED"],
        ["G-04", "High", "Ineffective verify_source_immutability()", "Rewrote verify_source_immutability() to audit reference model, frozen weights, and buffers bit-for-bit with float tolerance. Added regression test.", "CLOSED & TESTED"],
        ["G-05", "High", "Reproduction guide referenced missing scripts", "Corrected UTTA_MED_REPRODUCIBILITY.md: audit_stage_f_data.py -> verify_manuscript_tables.py and run_stage_e2_target_eval.py -> v2/evaluate_stage_e2.py.", "CLOSED"],
        ["G-06", "High", "Test evidence unverified in local environment", "Verified execution under conda env 'avinash' (Python 3.14.2, PyTorch 2.14.0+cpu, Pytest 9.1.1). Ran full suite: 120 passed, 2 skipped, 0 failed.", "VERIFIED"],
        ["G-07", "High", "Limited external target sample size (10 WSIs)", "Disclosed sparse sampling budget (128 patches/WSI) and WSI clustering in discussion; explicitly disclaimed whole-slide clinical deployment.", "CLOSED"],
        ["G-08", "Medium", "Protocol timing confusion (post vs pre-adaptation)", "Explicitly labeled v1 (post-adaptation patch evaluation) and v2 (pre-adaptation streaming inference with slide boundary resets) in all tables.", "CLOSED"],
        ["G-09", "Medium", "Novelty language risks algorithmic overreach", "Reframed contribution as an empirical reliability and failure-mode study; disclaimed first-ever SOTA algorithm claims.", "CLOSED"],
        ["G-10", "Medium", "Incomplete author metadata in LaTeX draft", "Completed author block in paper/ieee/utta_med.tex with department, laboratory, and institutional contact email.", "CLOSED"],
        ["G-11", "Medium", "Verification script vs reproduction audit", "Clarified that scripts/verify_manuscript_tables.py performs bit-exact artifact/table reconciliation across 54 stored checks (54/54 passed).", "CLOSED"],
        ["G-12", "Medium", "Collapse monitor sensitivity unproven on natural shock", "Documented that nominal streaming caused zero alarms and collapse sensitivity is validated under synthetic stress, leaving natural shock for future work.", "CLOSED"]
    ]
    tbl_gap = doc.add_table(rows=1, cols=5)
    format_table(tbl_gap, gap_widths, gap_headers, gap_data, gap_align)

    # 3. Detailed Technical Implementation
    add_header(doc, "3. Detailed Technical Implementation & Code Fixes", 1)
    
    add_header(doc, "3.1 Resolution of Gap G-03: Empty-Gate Momentum Drift", 2)
    add_body_p(doc, 
        "Root Cause: In src/v2/safeadapt/adapter.py, when zero patches satisfied the gating policy (mask.sum() < 1), the function returned logits.sum() * 0.0. In SafeAdaptController.step(), this zero tensor was treated as a differentiable loss: loss.backward() ran with zero gradient, and self.optimizer.step() was executed. In stateful optimizers such as Adam or SGD with momentum, non-zero first and second moment vectors updated model parameters even though zero patches passed the uncertainty gate."
    )
    add_body_p(doc, "Code Implementation Fixes Applied:", bold_prefix="Implementation: ")
    add_code_block(doc, 
        "# 1. src/v2/safeadapt/adapter.py (Lines 88-94)\n"
        "if self.gate_fn is not None:\n"
        "    mask = self.gate_fn(logits, x)\n"
        "    if mask.sum() < 1:\n"
        "        # Return None to explicitly skip backward and optimizer step\n"
        "        return None\n"
        "    return (mask * ent).sum() / mask.sum()\n\n"
        "# 2. src/v2/safeadapt/controller.py (Lines 312-329)\n"
        "loss = compute_loss_fn()\n"
        "if loss is None:\n"
        "    self.telemetry.record_event(batch_idx=current_batch_idx, event_type='SKIP_EMPTY_GATE',\n"
        "                                reason='zero_selected_patches', loss=None, in_fallback=self._in_fallback_mode)\n"
        "    safe_preds = self.forward_source(x, forward_fn=forward_inference_fn) if self._in_fallback_mode else (\n"
        "        forward_inference_fn(x) if forward_inference_fn is not None else self.forward_source(x)\n"
        "    )\n"
        "    return safe_preds, False"
    )
    add_body_p(doc, 
        "Regression Test: Implemented test_empty_gate_skips_optimizer_step_and_preserves_weights_and_momentum in tests/test_safeadapt_v2.py. Step 1 executes an adaptation step establishing Adam momentum; Step 2 runs an empty-gate loss (None). Verified: parameters remain bit-for-bit identical (torch.equal) and consecutive rejections are not incremented."
    )

    add_header(doc, "3.2 Resolution of Gap G-04: Bit-for-Bit Source Immutability Verifier", 2)
    add_body_p(doc, 
        "Root Cause: The original verify_source_immutability() only checked key presence and isinstance(src_v, torch.Tensor). If frozen convolutional weights or running stat buffers were mutated, the method returned True."
    )
    add_body_p(doc, "Deep Audit Implementation Fix:", bold_prefix="Implementation: ")
    add_code_block(doc, 
        "# src/v2/safeadapt/controller.py (Lines 136-179)\n"
        "def verify_source_immutability(self) -> bool:\n"
        "    source_model_state = self._source_model.state_dict()\n"
        "    def tensors_match(t1, t2):\n"
        "        if t1.shape != t2.shape: return False\n"
        "        if t1.is_floating_point():\n"
        "            return bool(torch.allclose(t1.cpu(), t2.cpu(), atol=1e-6, rtol=1e-5))\n"
        "        return bool(torch.equal(t1.cpu(), t2.cpu()))\n\n"
        "    # 1. Audit reference source model\n"
        "    for k, src_v in self._immutable_source_state.items():\n"
        "        if not tensors_match(source_model_state[k], src_v): return False\n\n"
        "    # 2. Audit active model frozen parameters and buffers\n"
        "    adaptable_ids = {id(p) for p in self.adaptable_params}\n"
        "    for name, p in self.model.named_parameters():\n"
        "        if id(p) not in adaptable_ids or not p.requires_grad:\n"
        "            if not tensors_match(p.detach(), self._immutable_source_state[name]): return False\n"
        "    for name, b in self.model.named_buffers():\n"
        "        if not tensors_match(b.detach(), self._immutable_source_state[name]): return False\n"
        "    return True"
    )
    add_body_p(doc, 
        "Regression Test: Implemented test_verify_source_immutability_detects_deliberate_tampering in tests/test_safeadapt_v2.py. Verified that corrupting a frozen layer weight (model.fc1.weight.add_(1.0)) or reference model returns False, and restoring it returns True."
    )

    add_header(doc, "3.3 Resolution of Gap G-02: Cover Letter Realignment", 2)
    add_body_p(doc, 
        "The submission cover letter (paper/ieee/cover_letter.md) was completely rewritten to align strictly with the empirical scope of the paper. Removed all claims of 'certified clinical safeguard', mathematical proofs bounding parameter drift, and unattached MIDOG/NCT-CRC claims. The letter now foregrounds the 5-seed Camelyon17-WILDS findings, the 25-fold harm reduction from selective gating, the empirical equivalence of MC-dropout and confidence ranking (rho = -0.9917), and transparently reports the slight negative transfer on the Hospital 2 external target."
    )

    add_header(doc, "3.4 Resolution of Gap G-10: Author and Venue Completion in LaTeX", 2)
    add_body_p(doc, 
        "Updated paper/ieee/utta_med.tex with the official author block, completing department, laboratory ('Medical Imaging and Machine Learning Laboratory'), and contact email (avinashreddy.crp@gmail.com)."
    )

    # 4. Post-Verification Audit Results
    add_header(doc, "4. Post-Verification Audit Results", 1)
    add_body_p(doc, "Software Runtime Environment:", bold_prefix="Environment: ")
    add_body_p(doc, 
        "- Python Runtime: Python 3.14.2 (Conda Environment: avinash)\n"
        "- PyTorch Core: 2.14.0+cpu\n"
        "- Pytest: 9.1.1\n"
        "- Operating System: Windows 11 Enterprise (x86_64)"
    )

    add_header(doc, "4.1 Automated Regression Test Suite Telemetry", 2)
    add_code_block(doc, 
        "pytest tests/ -v\n"
        "================================= test session starts =================================\n"
        "platform win32 -- Python 3.14.2, pytest-9.1.1, pluggy-1.6.0\n"
        "rootdir: C:\\Users\\avina\\OneDrive\\Desktop\\UTTA_MED_V1_FINAL\\UTTA_MED_V1_FINAL\n"
        "configfile: pytest.ini\n"
        "collected 122 items\n\n"
        "tests\\test_architecture.py ............                                          [  9%]\n"
        "tests\\test_attn_mil.py ....                                                      [ 13%]\n"
        "tests\\test_coverage_risk.py ........                                             [ 19%]\n"
        "tests\\test_evidential_tta.py ......                                              [ 24%]\n"
        "tests\\test_foundation_tta.py ....                                                [ 27%]\n"
        "tests\\test_gating_v2.py ..............                                           [ 39%]\n"
        "tests\\test_metrics.py ..                                                         [ 40%]\n"
        "tests\\test_model.py .                                                            [ 41%]\n"
        "tests\\test_no_leakage.py .ss......                                               [ 49%]\n"
        "tests\\test_optional.py .......                                                   [ 54%]\n"
        "tests\\test_ot_tta.py ..                                                          [ 56%]\n"
        "tests\\test_paths.py .....                                                        [ 60%]\n"
        "tests\\test_randstainna.py ...                                                    [ 63%]\n"
        "tests\\test_safeadapt_c1_monitor.py ..............                                [ 74%]\n"
        "tests\\test_safeadapt_v2.py ................                                      [ 87%]\n"
        "tests\\test_slide_aggregation.py .....                                            [ 91%]\n"
        "tests\\test_tent_collapse.py .......                                              [ 97%]\n"
        "tests\\test_tta.py ...                                                            [100%]\n\n"
        "========================= 120 passed, 2 skipped in 11.16s ========================="
    )

    add_header(doc, "4.2 Bit-Exact Numerical Table & Hash Audit", 2)
    add_body_p(doc, 
        "Executed scripts/verify_manuscript_tables.py against the canonical checkpoint (SHA-256: f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000) and stored result JSON/CSV artifacts. Result: 54 of 54 numerical checks passed (100% bit-exact consistency across Tables 1 through 5)."
    )

    # 5. Authoritative Results
    add_header(doc, "5. Authoritative Consolidated Benchmark Results", 1)
    
    add_header(doc, "5.1 Historical v1 Five-Seed Camelyon17-WILDS Benchmark", 2)
    t1_headers = ["Method", "AUROC (Mean ± SD)", "F1 Score (Mean ± SD)", "Harm Rate", "Coverage", "Collapse Outcome"]
    t1_widths = [1.5, 1.2, 1.2, 0.9, 0.8, 1.1]
    t1_align = ["left", "center", "center", "center", "center", "center"]
    t1_data = [
        ["Source-Only (Frozen)", "0.930 ± 0.009", "0.807 ± 0.025", "—", "—", "Baseline"],
        ["Tent (Unconstrained)", "0.698 ± 0.324", "0.506 ± 0.411", "18.7% ± 17.9%", "100.0%", "Collapsed (Seeds 7, 99)"],
        ["EATA (Point-Entropy)", "0.921 ± 0.026", "0.809 ± 0.031", "2.5% ± 1.5%", "88.5%", "Stable"],
        ["Random-Gated TTA", "0.697 ± 0.304", "0.487 ± 0.385", "19.4% ± 17.0%", "70.0%", "Collapsed (Seeds 7, 99)"],
        ["Confidence-Gated", "0.936 ± 0.022", "0.831 ± 0.032", "0.7% ± 0.3%", "58.1%", "Stable (5/5 Seeds)"],
        ["UTTA-Med (MC-Dropout)", "0.935 ± 0.022", "0.830 ± 0.032", "0.7% ± 0.3%", "57.5%", "Stable (5/5 Seeds)"]
    ]
    tbl_t1 = doc.add_table(rows=1, cols=6)
    format_table(tbl_t1, t1_widths, t1_headers, t1_data, t1_align)
    add_body_p(doc, 
        "Key Finding: Gating raises F1 across all five seeds (+0.0231, p=0.0072). MC-dropout predictive variance correlates strictly with confidence (rho = -0.9917) and yields statistically indistinguishable performance (delta AUROC = -0.00005, p=0.8688) while incurring a 3.5x computational slowdown."
    )

    add_header(doc, "5.2 Separate v2 Stage E2 External Target Results (Hospital 2)", 2)
    t2_headers = ["Evaluation Metric", "Frozen Source Baseline", "Confidence TTA", "Active SafeAdapt", "Observed Delta"]
    t2_widths = [1.8, 1.2, 1.2, 1.2, 1.1]
    t2_align = ["left", "center", "center", "center", "center"]
    t2_data = [
        ["Macro Valid AUROC", "0.9308", "0.9299", "0.9300", "-0.0008"],
        ["Pooled AUROC", "0.9364", "0.9360", "0.9359", "-0.0005"],
        ["Pooled F1 Score", "0.7840", "0.7822", "0.7822", "-0.0018"],
        ["Expected Calibration Error", "0.0650", "0.0655", "0.0655", "+0.0005 (worse)"],
        ["Brier Score", "0.091709", "0.092051", "0.092053", "+0.000344 (worse)"],
        ["Max Parameter Displacement", "0.00", "4.01 x 10^-5", "4.01 x 10^-5", "<< 0.05 (Bounded)"]
    ]
    tbl_t2 = doc.add_table(rows=1, cols=5)
    format_table(tbl_t2, t2_widths, t2_headers, t2_data, t2_align)
    add_body_p(doc, 
        "Key Finding: SafeAdapt strictly prevented numerical collapse (L_inf = 4.01e-5 << 0.05). However, adaptation yielded slight negative transfer on Hospital 2 due to confirmation bias on outlier slides (Slide 29). Bounding parameter drift alone does not guarantee positive transfer."
    )

    # 6. Manifest of Updated Artifacts
    add_header(doc, "6. Manifest of Updated Files and Artifacts", 1)
    m_headers = ["Relative File Path", "Module / Area", "Key Modifications & Action Taken"]
    m_widths = [2.2, 1.3, 3.0]
    m_align = ["left", "left", "left"]
    m_data = [
        ["POST_VERIFICATION_IMPLEMENTATION_REPORT.md", "Dossier", "Complete post-verification report documenting all gap closures and telemetry."],
        ["src/v2/safeadapt/adapter.py", "Adapter", "Fixed empty-gate mask to return None, preventing zero-loss backward passes."],
        ["src/v2/safeadapt/controller.py", "Controller", "Added explicit loss is None bypass and bit-for-bit source immutability audit."],
        ["tests/test_safeadapt_v2.py", "Test Suite", "Added regression tests for G-03 (momentum drift) and G-04 (source tampering)."],
        ["paper/ieee/cover_letter.md", "Submission", "Rewrote cover letter removing unsupported claims and aligning with paper scope."],
        ["paper/ieee/utta_med.tex", "Manuscript", "Completed IEEE author affiliations, laboratory, and contact email."],
        ["docs/final/UTTA_MED_REPRODUCIBILITY.md", "Documentation", "Fixed reproduction commands, script paths, and environment specifications."],
        ["paper/PUBLICATION_METADATA_AND_ABSTRACT.md", "Dossier", "Updated submission dossier with Stage E2 target results and committee audit matrix."]
    ]
    tbl_m = doc.add_table(rows=1, cols=3)
    format_table(tbl_m, m_widths, m_headers, m_data, m_align)

    # 7. Final Verdict
    add_header(doc, "7. Submission Readiness Verdict", 1)
    add_callout(doc, 
        "VERDICT: READY FOR SUBMISSION AS AN EMPIRICAL RELIABILITY STUDY\n\n"
        "The codebase, test suites, documentation, and submission dossiers are fully synchronized, scientifically defensible, and verified against all committee critiques. "
        "The manuscript is framed as a rigorous empirical reliability and failure-mode investigation in digital pathology, avoiding overclaiming clinical readiness while providing actionable reproducibility insights.",
        title="SUBMISSION VERDICT", border_color="15803D", fill_color="F0FDF4"
    )

    out_path_1 = "./UTTA_Med_Post_Verification_Implementation_Report.docx"
    doc.save(out_path_1)
    print(f"Successfully generated: {out_path_1}")

    # Copy to Downloads as well
    out_path_2 = "str(Path.home() / "Downloads")/UTTA_Med_Post_Verification_Implementation_Report.docx"
    try:
        shutil.copyfile(out_path_1, out_path_2)
        print(f"Successfully copied to: {out_path_2}")
    except Exception as e:
        print(f"Note: Could not copy to Downloads: {e}")

if __name__ == "__main__":
    build_report()
