/**
 * Google Apps Script for SafeAdapt Slide Deck Generation
 *
 * HOW TO USE THIS IN GOOGLE SLIDES:
 * 1. Open your browser and go to: https://slides.new (creates a new Google Slide deck)
 * 2. In the top menu, click: Extensions -> Apps Script
 * 3. Delete any default code in the editor, and paste this entire file.
 * 4. Click the "Save" (disk) icon, then click "Run".
 * 5. Grant permissions if prompted (Review permissions -> Choose your Google account -> Advanced -> Go to Untitled project -> Allow).
 * 6. Return to your Google Slides tab: All 18 formatted slides with titles, cards, tables, and speaker notes are instantly created!
 */

function createSafeAdaptDeck() {
  var presentation = SlidesApp.getActivePresentation();
  
  // Color Palette Constants (RGB Hex)
  var C_NAVY = '#0E2B5C';
  var C_BLUE = '#1C5AA0';
  var C_RED = '#B42828';
  var C_GREEN = '#228B22';
  var C_CARD_BG = '#F8FAFC';
  var C_BORDER = '#CBD5E1';
  var C_TEXT = '#1E293B';
  var C_WHITE = '#FFFFFF';
  var C_MUTED = '#64748B';

  // Helper function to set slide background
  function setBg(slide, color) {
    slide.getBackground().setSolidFill(color);
  }

  // Helper to add header & category
  function addHeader(slide, title, category) {
    var catBox = slide.insertTextBox(category.toUpperCase(), 50, 20, 860, 25);
    var pCat = catBox.getText().getParagraphs()[0];
    pCat.getTextStyle().setFontSize(10).setBold(true).setForegroundColor(C_BLUE);

    var titleBox = slide.insertTextBox(title, 50, 45, 860, 55);
    var pTitle = titleBox.getText().getParagraphs()[0];
    pTitle.getTextStyle().setFontSize(20).setBold(true).setForegroundColor(C_NAVY);
  }

  // Helper to add footer
  function addFooter(slide, num) {
    var foot = slide.insertTextBox(
      "SafeAdapt: Geometry-Preserving Optimal Transport in Digital Pathology | IEEE TMI 2026 | DOI: 10.5281/zenodo.23658430    Slide " + num + " of 18",
      50, 500, 860, 25
    );
    foot.getText().getTextStyle().setFontSize(9).setForegroundColor(C_MUTED);
  }

  // --------------------------------------------------------------------------
  // SLIDE 1: Title Slide
  // --------------------------------------------------------------------------
  var s1 = presentation.getSlides()[0] || presentation.appendSlide();
  setBg(s1, C_NAVY);

  var t1 = s1.insertTextBox(
    "SafeAdapt: Geometry-Preserving Optimal Transport and Fail-Closed Safeguards for Test-Time Adaptation in Digital Pathology",
    60, 80, 840, 150
  );
  var p1 = t1.getText().getParagraphs()[0];
  p1.getTextStyle().setFontSize(26).setBold(true).setForegroundColor(C_WHITE);

  var sub1 = s1.insertTextBox(
    "Reversing Negative Transfer and Precluding Attractor Collapse Under Multicenter Clinical Shift",
    60, 210, 840, 40
  );
  sub1.getText().getTextStyle().setFontSize(14).setForegroundColor('#94A3B8');

  var authBox = s1.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 60, 270, 430, 180);
  authBox.getFill().setSolidFill('#173A72');
  authBox.getBorder().getLineFill().setSolidFill('#264E8E');
  var authText = authBox.getText();
  authText.setText("Avinash Reddy Cheerapareddy & Gunnam Sanjana\n\nDepartment of Computer Science and Engineering\nSRM University-AP, Amaravati, Andhra Pradesh, India\navinashreddy_cheerapareddy@srmap.edu.in | sanjana_gunnam@srmap.edu.in");
  authText.getTextStyle().setFontSize(11).setForegroundColor(C_WHITE);

  var repBox = s1.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 510, 270, 390, 180);
  repBox.getFill().setSolidFill('#173A72');
  repBox.getBorder().getLineFill().setSolidFill('#264E8E');
  var repText = repBox.getText();
  repText.setText("Open Science & Provenance\n\n• Zenodo Dataset Archive: 10.5281/zenodo.23658430\n• GitHub: github.com/Avirech7676/UTTA-Med\n• Audit: 57/57 Bit-Exact Matches | 148 Unit Tests\n• Target: IEEE Transactions on Medical Imaging (TMI)");
  repText.getTextStyle().setFontSize(11).setForegroundColor(C_WHITE);

  s1.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Good morning. Today I present SafeAdapt: a geometry-preserving optimal transport and fail-closed safeguarding framework that addresses catastrophic test-time adaptation collapse in digital pathology."
  );

  // --------------------------------------------------------------------------
  // SLIDE 2: Clinical Motivation
  // --------------------------------------------------------------------------
  var s2 = presentation.appendSlide();
  setBg(s2, '#FFFFFF');
  addHeader(s2, "Cross-Center Stain and Scanner Variability Degrades Deep Pathology Models", "1. Clinical Motivation");
  addFooter(s2, 2);

  var card2a = s2.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 520, 170);
  card2a.getFill().setSolidFill(C_CARD_BG);
  card2a.getBorder().getLineFill().setSolidFill(C_BORDER);
  card2a.getText().setText("The Clinical Setting\n• Digital pathology models fail when transferred across hospitals.\n• Variations in microtomy, H&E staining reagents, and scanner sensor optics induce massive domain shifts.\n• Retraining is legally prohibited: HIPAA and GDPR ban pooling patient slides on central servers.");
  card2a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card2b = s2.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 300, 520, 170);
  card2b.getFill().setSolidFill(C_CARD_BG);
  card2b.getBorder().getLineFill().setSolidFill(C_BORDER);
  card2b.getText().setText("The Promise & Danger of Test-Time Adaptation (TTA)\n• TTA updates model parameters on-the-fly using unlabeled target data.\n• Requires zero access to source training datasets.\n• Must operate continuously during clinical whole-slide scanning.");
  card2b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card2c = s2.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 590, 110, 320, 360);
  card2c.getFill().setSolidFill('#FEF2F2');
  card2c.getBorder().getLineFill().setSolidFill(C_RED);
  card2c.getText().setText("THE CLINICAL TRANSFER GAP\n\n-28%\nSensitivity Drop on Unseen Slides\n\nTransferring a model trained on Camelyon17 Centers 0, 3, 4 to unseen Center 2 drops metastasis sensitivity by up to 28% due to histochemical stain disparity.\n\nDataset: 455,954 patches across 5 clinical centers.");
  card2c.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s2.getNotesPage().getSpeakerNotesShape().getText().setText(
    "In digital pathology, algorithms trained on one hospital routinely fail at another due to tissue prep and scanner variations. Centralizing data is prohibited by privacy laws."
  );

  // --------------------------------------------------------------------------
  // SLIDE 3: Research Gap (Tent Collapse)
  // --------------------------------------------------------------------------
  var s3 = presentation.appendSlide();
  setBg(s3, '#FFFFFF');
  addHeader(s3, "Existing Test-Time Adaptation Methods Suffer Catastrophic Attractor Collapse", "2. Research Gap");
  addFooter(s3, 3);

  var card3a = s3.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 860, 360);
  card3a.getFill().setSolidFill(C_CARD_BG);
  card3a.getBorder().getLineFill().setSolidFill(C_BORDER);
  card3a.getText().setText(
    "Unconstrained Entropy Minimization (Tent, ICLR 2021)\n" +
    "• Minimizes Shannon entropy L = -Σ p log p over BatchNorm affine parameters (γ, β).\n" +
    "• Erroneous high-confidence predictions inject destructive pseudo-gradients, collapsing all outputs toward a single majority class.\n\n" +
    "Our Multi-Seed Audit Discovery: 40% Collapse Rate on Real Pathology\n" +
    "• Across 5 random seeds on Camelyon17 (N = 85,054 target patches):\n" +
    "   - Tent collapses completely on 2 out of 5 seeds (Seeds 7 and 99).\n" +
    "   - Macro AUROC drops from 0.930 to 0.3429 (worse than random guessing!).\n" +
    "   - Sensitivity collapses to < 3% (misses almost all micro-metastases).\n" +
    "   - Harmful adaptation rate reaches 18.7% ± 17.9%."
  );
  card3a.getText().getTextStyle().setFontSize(12).setForegroundColor(C_TEXT);

  s3.getNotesPage().getSpeakerNotesShape().getText().setText(
    "When we conducted a 5-seed audit on Camelyon17, Tent collapsed on 40% of seeds. On seeds 7 and 99, AUROC plunged to 0.34 and sensitivity fell below 3%."
  );

  // --------------------------------------------------------------------------
  // SLIDE 4: Research Question
  // --------------------------------------------------------------------------
  var s4 = presentation.appendSlide();
  setBg(s4, '#FFFFFF');
  addHeader(s4, "Research Question: Achieving Guaranteed Positive Transfer and Safety", "3. Core Objective");
  addFooter(s4, 4);

  var card4q = s4.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 860, 90);
  card4q.getFill().setSolidFill('#EFF6FF');
  card4q.getBorder().getLineFill().setSolidFill(C_BLUE);
  card4q.getText().setText("CENTRAL RESEARCH QUESTION\nCan streaming test-time adaptation achieve guaranteed positive transfer in computational pathology without falling into single-class attractor collapse or violating clinical safety bounds?");
  card4q.getText().getTextStyle().setFontSize(13).setForegroundColor(C_NAVY);

  var card4a = s4.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 220, 270, 250);
  card4a.getFill().setSolidFill(C_CARD_BG);
  card4a.getBorder().getLineFill().setSolidFill(C_BORDER);
  card4a.getText().setText("1. Disprove Fallacies\n\nAnalyze why scalar uncertainty and confidence gating fail to prevent representation collapse under realistic clinical class imbalance.");
  card4a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card4b = s4.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 345, 220, 270, 250);
  card4b.getFill().setSolidFill(C_CARD_BG);
  card4b.getBorder().getLineFill().setSolidFill(C_BORDER);
  card4b.getText().setText("2. Geometric Alignment\n\nFormulate an Entropic Optimal Transport objective with exact marginal constraints (Q^T 1_B = r) against certified source prototypes.");
  card4b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card4c = s4.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 640, 220, 270, 250);
  card4c.getFill().setSolidFill(C_CARD_BG);
  card4c.getBorder().getLineFill().setSolidFill(C_BORDER);
  card4c.getText().setText("3. Clinical Safeguards\n\nEngineer tissue optical density monitors, PAC parameter bounds (||Δθ||_inf <= 0.05), and episodic WSI resets for patient isolation.");
  card4c.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s4.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Can we adapt models such that positive transfer is mathematically guaranteed, while precluding attractor collapse? We broke this into 3 objectives."
  );

  // --------------------------------------------------------------------------
  // SLIDE 5: Contributions (4 Bento Cards)
  // --------------------------------------------------------------------------
  var s5 = presentation.appendSlide();
  setBg(s5, '#FFFFFF');
  addHeader(s5, "SafeAdapt Introduces Four Foundational Contributions", "4. Contributions");
  addFooter(s5, 5);

  var c5a = s5.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 170);
  c5a.getFill().setSolidFill(C_CARD_BG);
  c5a.getBorder().getLineFill().setSolidFill(C_NAVY);
  c5a.getText().setText("1. THEORETICAL DISPROOF: THE TAYLOR TRAP\nProved via Taylor expansion that MC-Dropout variance is rank-equivalent to softmax confidence (ρ = -0.9917) in 1D sigmoid heads. Uncertainty gating merely wastes 3.55x latency.");
  c5a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var c5b = s5.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 170);
  c5b.getFill().setSolidFill(C_CARD_BG);
  c5b.getBorder().getLineFill().setSolidFill(C_BLUE);
  c5b.getText().setText("2. GEOMETRY-PRESERVING OT-TTA\nFormulated batch adaptation as an Entropic Optimal Transport problem with exact target marginal prior constraints (Q^T 1_B = r), mathematically precluding attractor collapse.");
  c5b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var c5c = s5.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 300, 420, 170);
  c5c.getFill().setSolidFill(C_CARD_BG);
  c5c.getBorder().getLineFill().setSolidFill(C_GREEN);
  c5c.getText().setText("3. BIOLOGICAL SAFEGUARDS: CMR TISSUE GATING\nCellularity-Mitosis Ratio (CMR) leverages Beer-Lambert optical density deconvolution, eliminating all 14 false silence alarms on benign acellular stroma during MIDOG 2021 mitosis detection.");
  c5c.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var c5d = s5.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 300, 420, 170);
  c5d.getFill().setSolidFill(C_CARD_BG);
  c5d.getBorder().getLineFill().setSolidFill(C_NAVY);
  c5d.getText().setText("4. PROVABLE PAC BOUNDS & RUNTIME INTERLOCKS\nProved PAC L_inf <= 0.05 affine drift bounds maximum output probability shift to <= 0.0460. Coupled with 1.8 ms dual-buffer atomic rollbacks and episodic WSI-boundary resets.");
  c5d.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s5.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Our four contributions span theory, algorithms, biology, and systems engineering. We disprove MC-Dropout in 1D heads, solve collapse with OT, and prove PAC bounds."
  );

  // --------------------------------------------------------------------------
  // SLIDE 6: Architecture & Pipeline
  // --------------------------------------------------------------------------
  var s6 = presentation.appendSlide();
  setBg(s6, '#FFFFFF');
  addHeader(s6, "SafeAdapt Architecture: High-Level End-to-End Pipeline", "5. System Architecture");
  addFooter(s6, 6);

  var pipeBox = s6.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 860, 240);
  pipeBox.getFill().setSolidFill(C_CARD_BG);
  pipeBox.getBorder().getLineFill().setSolidFill(C_BLUE);
  pipeBox.getText().setText(
    "END-TO-END STREAMING PIPELINE WORKFLOW:\n\n" +
    "[1. Target Stream] -> Streaming WSI Patches (x_t) arrive during scanning.\n" +
    "[2. Diagnostic Logging] -> Pre-adaptation prediction emitted FIRST from uncorrupted state.\n" +
    "[3. Biological Gating] -> Beer-Lambert optical density deconvolution drops acellular stroma.\n" +
    "[4. Entropic OT Alignment] -> Sinkhorn-Knopp computes transportation Q subject to Q^T 1_B = r.\n" +
    "[5. Runtime Sentinel] -> Verifies gradient finiteness and displacement ||θ_t - θ_0||_inf <= 0.05.\n" +
    "[6. Patient Isolation] -> Model unconditionally re-anchors to source state θ_0 at slide transitions."
  );
  pipeBox.getText().getTextStyle().setFontSize(12).setForegroundColor(C_TEXT);

  var phiBox = s6.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 370, 860, 100);
  phiBox.getFill().setSolidFill('#EFF6FF');
  phiBox.getBorder().getLineFill().setSolidFill(C_NAVY);
  phiBox.getText().setText("Execution Philosophy: Diagnostics Precede Adaptation\nClinical diagnostic predictions are never generated from in-flux adapted weights. Slide-boundary resets ensure complete inter-patient isolation.");
  phiBox.getText().getTextStyle().setFontSize(12).setForegroundColor(C_NAVY);

  s6.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Notice the execution philosophy: clinical diagnostics are emitted first, before any model updates occur. And parameters reset at every slide boundary."
  );

  // --------------------------------------------------------------------------
  // SLIDE 7: Method: OT-TTA
  // --------------------------------------------------------------------------
  var s7 = presentation.appendSlide();
  setBg(s7, '#FFFFFF');
  addHeader(s7, "Entropic Optimal Transport with Marginal Constraints Precludes Collapse", "6. Method: OT-TTA");
  addFooter(s7, 7);

  var card7a = s7.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 860, 360);
  card7a.getFill().setSolidFill(C_CARD_BG);
  card7a.getBorder().getLineFill().setSolidFill(C_BLUE);
  card7a.getText().setText(
    "Mathematical Formulation of Entropic Optimal Transport (OT-TTA):\n\n" +
    "   min_{Q in U(p, r)}  <Q, M> - ε H(Q)\n" +
    "   where U(p, r) = { Q in R_+^{B x K} | Q 1_K = (1/B) 1_B,  Q^T 1_B = r }\n\n" +
    "• Cost Matrix: M_ik = -log p(y = k | x_i) against certified source prototypes.\n" +
    "• Marginal Prior Constraint: r = [r_0, r_1]^T guarantees that exactly r_1 * B mass is transported to tumor.\n" +
    "• Solved via rapid Sinkhorn-Knopp matrix scaling in < 2.5 ms per batch on GPU.\n\n" +
    "Clinical Proof on Severe Stain Outlier Slide 29:\n" +
    "• Slide 29 features extreme dark purple hematoxylin precipitation.\n" +
    "• Standard Tent collapses to 0.0521 AUROC (predicts 0% tumor across the entire slide).\n" +
    "• SafeAdapt-OT restores feature margin: elevates AUROC to 0.9324 ± 0.0078 (0.9360 on Seed 42)!"
  );
  card7a.getText().getTextStyle().setFontSize(12).setForegroundColor(C_TEXT);

  s7.getNotesPage().getSpeakerNotesShape().getText().setText(
    "In SafeAdapt-OT, the column constraint Q^T 1_B = r mathematically forces the model to allocate mass to both classes, rescuing outlier Slide 29 to 0.932 AUROC."
  );

  // --------------------------------------------------------------------------
  // SLIDE 8: Method: Safeguards & PAC Bounds
  // --------------------------------------------------------------------------
  var s8 = presentation.appendSlide();
  setBg(s8, '#FFFFFF');
  addHeader(s8, "Biological Tissue Gating and Provable PAC Parameter Bounds", "7. Method: Safeguards");
  addFooter(s8, 8);

  var card8a = s8.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 360);
  card8a.getFill().setSolidFill(C_CARD_BG);
  card8a.getBorder().getLineFill().setSolidFill(C_NAVY);
  card8a.getText().setText(
    "Cellularity-Mitosis Ratio (CMR)\n\n" +
    "• The Problem: In dense mitosis detection (MIDOG 2021), normal tumor stroma contains zero mitoses.\n" +
    "• Conventional silence monitors mistake physiological absence for detector failure, tripping false alarms.\n\n" +
    "• Beer-Lambert Optical Density Deconvolution:\n" +
    "   OD = -log10(I / I_0)\n" +
    "   H = v_hematoxylin^T OD\n\n" +
    "• Result: If tissue cellularity H < τ_cellularity, silence triggers are bypassed physiologically.\n" +
    "• Eliminates all 14 false silence alarms across 50 whole-slide cases!"
  );
  card8a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card8b = s8.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 360);
  card8b.getFill().setSolidFill(C_CARD_BG);
  card8b.getBorder().getLineFill().setSolidFill(C_GREEN);
  card8b.getText().setText(
    "Provable PAC Parameter Drift Bound\n\n" +
    "• Theorem: For terminal normalization affine parameters θ = {γ, β}, clamping parameter displacement:\n" +
    "   ||θ_t - θ_0||_inf <= ε = 0.05\n\n" +
    "• Strictly bounds maximum output probability shift:\n" +
    "   |σ(z_t) - σ(z_0)| <= (1/4) ||w_L||_1 [ ε |x - μ|/σ + ε ]\n" +
    "   Δp_max <= 0.0460\n\n" +
    "• Sub-Frame Atomic Rollback:\n" +
    "   Dual-buffer memory restores preceding certified state in 1.8 ms upon non-finite gradient interception."
  );
  card8b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s8.getNotesPage().getSpeakerNotesShape().getText().setText(
    "CMR optical density gating eliminates 14 false silence alarms on stroma, while our PAC theorem bounds maximum probability change to 0.0460."
  );

  // --------------------------------------------------------------------------
  // SLIDE 9: Experimental Setup
  // --------------------------------------------------------------------------
  var s9 = presentation.appendSlide();
  setBg(s9, '#FFFFFF');
  addHeader(s9, "Multicenter Evaluation Setup with Zero Target Leakage Protocol", "8. Experimental Setup");
  addFooter(s9, 9);

  var card9a = s9.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 360);
  card9a.getFill().setSolidFill(C_CARD_BG);
  card9a.getBorder().getLineFill().setSolidFill(C_NAVY);
  card9a.getText().setText(
    "Camelyon17-WILDS (Classification)\n\n" +
    "• Task: Lymph node metastasis patch classification\n" +
    "• Total Dataset: 455,954 patches (96x96 px), 5 centers\n" +
    "• Source Training: Centers 0, 3, 4 (n = 302,436)\n" +
    "• In-Domain Val: Centers 0, 3, 4 (n = 33,560)\n" +
    "• OOD Calibration: Center 1 (n = 34,904) -> Zero Leakage!\n" +
    "• Target Test Cohort: Center 2 (n = 85,054, 10 WSIs)\n" +
    "• Backbone: ResNet-18, 1-logit BCE, Adam lr=1e-5"
  );
  card9a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card9b = s9.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 360);
  card9b.getFill().setSolidFill(C_CARD_BG);
  card9b.getBorder().getLineFill().setSolidFill(C_BLUE);
  card9b.getText().setText(
    "MIDOG 2021 (Mitosis Detection)\n\n" +
    "• Task: Dense mitotic figure detection under scanner shift\n" +
    "• Dataset: 50 whole-slide cases, 1,100 patches (512x512 px)\n" +
    "• Target Scanner: Held-out Scanner 3\n" +
    "• Architecture: Faster R-CNN ResNet-50-FPN\n\n" +
    "Rigorous Scientific Protocols:\n" +
    "• 5 Random Seeds: {42, 123, 2024, 7, 99}\n" +
    "• Baselines: Source-Only, Tent, EATA, SAR, Random Controls\n" +
    "• Certified hardware reproducibility on Tesla T4 GPUs."
  );
  card9b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s9.getNotesPage().getSpeakerNotesShape().getText().setText(
    "We evaluate on two multicenter benchmarks across 5 random seeds. Crucially, all hyperparameters were frozen on Hospital 1 before testing on Hospital 2."
  );

  // --------------------------------------------------------------------------
  // SLIDE 10: Primary Results Table
  // --------------------------------------------------------------------------
  var s10 = presentation.appendSlide();
  setBg(s10, '#FFFFFF');
  addHeader(s10, "SafeAdapt Achieves Superior AUROC, Cuts ECE by 43%, and Slashes Harm >100x", "9. Primary Results");
  addFooter(s10, 10);

  var table10 = s10.insertTable(7, 5, 50, 110, 860, 240);
  var tableData = [
    ["Method", "AUROC ↑", "F1 Score ↑", "ECE ↓", "Harm Rate ↓"],
    ["Source-Only", "0.930 ± 0.009", "0.807 ± 0.025", "0.128 ± 0.026", "—"],
    ["Tent (ICLR'21)", "0.698 ± 0.324", "0.506 ± 0.411", "0.284 ± 0.179", "0.187 ± 0.179"],
    ["EATA (ICML'22)", "0.921 ± 0.026", "0.809 ± 0.031", "0.152 ± 0.023", "0.025 ± 0.015"],
    ["Random-Gated (70%)", "0.697 ± 0.304", "0.487 ± 0.385", "0.298 ± 0.166", "0.194 ± 0.170"],
    ["Confidence-Gated", "0.936 ± 0.022", "0.831 ± 0.032", "0.130 ± 0.030", "0.007 ± 0.003"],
    ["SafeAdapt (Ours)", "0.9495 ± 0.013", "0.842 ± 0.031", "0.075 ± 0.005", "0.0018 ± 0.0003"]
  ];

  for (var r = 0; r < 7; r++) {
    for (var c = 0; c < 5; c++) {
      var cell = table10.getCell(r, c);
      cell.getText().setText(tableData[r][c]);
      var cellP = cell.getText().getParagraphs()[0];
      cellP.getTextStyle().setFontSize(10);
      if (r === 0) {
        cellP.getTextStyle().setBold(true).setForegroundColor(C_WHITE);
        cell.getFill().setSolidFill(C_NAVY);
      } else if (r === 6) {
        cellP.getTextStyle().setBold(true).setForegroundColor(C_GREEN);
        cell.getFill().setSolidFill('#F0FDF4');
      } else {
        cellP.getTextStyle().setForegroundColor(C_TEXT);
      }
    }
  }

  var statBox10 = s10.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 370, 860, 100);
  statBox10.getFill().setSolidFill('#F0FDF4');
  statBox10.getBorder().getLineFill().setSolidFill(C_GREEN);
  statBox10.getText().setText(
    "STATISTICAL PROOF ACROSS ALL 5 SEEDS (HOSPITAL 2):\n" +
    "• Paired Wilcoxon Test: p = 0.0312 (ΔF1 > 0 on 5/5 random initialization seeds).\n" +
    "• Expected Calibration Error (ECE) cut by 43% (0.128 -> 0.075).\n" +
    "• Harmful adaptation slashed from 18.7% down to 0.18% (>100x safety improvement)."
  );
  statBox10.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s10.getNotesPage().getSpeakerNotesShape().getText().setText(
    "SafeAdapt achieves 0.9495 AUROC and cuts harm by over 100-fold. The paired Wilcoxon test yields p = 0.0312 across all 5 seeds."
  );

  // --------------------------------------------------------------------------
  // SLIDE 11: Ablations & Taylor Trap
  // --------------------------------------------------------------------------
  var s11 = presentation.appendSlide();
  setBg(s11, '#FFFFFF');
  addHeader(s11, "Ablation Study: Sample Identity and Disproof of Taylor Trap", "10. Ablation Analysis");
  addFooter(s11, 11);

  var card11a = s11.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 360);
  card11a.getFill().setSolidFill(C_CARD_BG);
  card11a.getBorder().getLineFill().setSolidFill(C_NAVY);
  card11a.getText().setText(
    "1. Matched-Coverage Random Gating Control\n" +
    "• Dropping 30% of patches at random (Random-Gated, 70% coverage) collapses on the exact same seeds as Tent (Seeds 7 and 99: AUROC ~ 0.35).\n" +
    "• Proves that adaptation gains arise from sample identity, not just taking fewer gradient steps.\n\n" +
    "2. Adaptation Step Sensitivity (k)\n" +
    "• k = 1: AUROC = 0.977, Harm = 0.020 (Optimal)\n" +
    "• k = 2: AUROC = 0.956\n" +
    "• k >= 5: AUROC collapses to 0.375!\n" +
    "• Single-step updates are strictly necessary to avoid drift."
  );
  card11a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card11b = s11.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 360);
  card11b.getFill().setSolidFill(C_CARD_BG);
  card11b.getBorder().getLineFill().setSolidFill(C_RED);
  card11b.getText().setText(
    "3. The Binary Uncertainty Taylor Trap\n\n" +
    "• Widespread Belief: Monte Carlo Dropout provides orthogonal epistemic uncertainty.\n\n" +
    "• Our Mathematical Disproof:\n" +
    "   Via first-order Taylor expansion around μ_z:\n" +
    "   Var_w[p] ≈ p_0^2 (1 - p_0)^2 σ_z^2\n\n" +
    "• Empirical Spearman Rank Correlation:\n" +
    "   ρ = -0.9917 between MC-Dropout variance and max-softmax confidence!\n\n" +
    "• Conclusion: MC-Dropout incurs a 3.55x latency penalty with zero added ranking discrimination."
  );
  card11b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s11.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Random gating proves sample identity matters. And our Taylor expansion proves MC-Dropout is rank-equivalent to confidence, wasting 3.55x compute."
  );

  // --------------------------------------------------------------------------
  // SLIDE 12: Qualitative Case Studies
  // --------------------------------------------------------------------------
  var s12 = presentation.appendSlide();
  setBg(s12, '#FFFFFF');
  addHeader(s12, "Qualitative Case Study: Slide 29 Restoration & MIDOG Tissue Detection", "11. Qualitative Evidence");
  addFooter(s12, 12);

  var card12a = s12.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 360);
  card12a.getFill().setSolidFill(C_CARD_BG);
  card12a.getBorder().getLineFill().setSolidFill(C_NAVY);
  card12a.getText().setText(
    "Case 1: Severe Stain Outlier (Slide 29)\n\n" +
    "• Slide 29 from Hospital 2 has extreme dark purple hematoxylin precipitation.\n" +
    "• Unconstrained Tent collapses logits to -∞, predicting 0% tumor across the entire slide.\n\n" +
    "• SafeAdapt-OT restores feature margins:\n" +
    "   - Source-Only AUROC: 0.8592\n" +
    "   - Tent Collapse AUROC: 0.0521 (Seed 7)\n" +
    "   - SafeAdapt-OT AUROC: 0.9324 ± 0.0078\n\n" +
    "• Diagnostic sensitivity fully restored on real tissue."
  );
  card12a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var card12b = s12.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 360);
  card12b.getFill().setSolidFill(C_CARD_BG);
  card12b.getBorder().getLineFill().setSolidFill(C_GREEN);
  card12b.getText().setText(
    "Case 2: MIDOG Mitosis Detection on Stroma\n\n" +
    "• Target Scanner 3 contains extensive benign stroma with zero mitotic figures.\n" +
    "• Baseline silence monitors trigger 14 false alarms, prematurely halting adaptation.\n\n" +
    "• CMR Tissue Gating:\n" +
    "   - Measures optical density H\n" +
    "   - Detects stroma physiologically\n" +
    "   - Permits continuous safe adaptation\n\n" +
    "• Result: Detection F1 = 0.5172 ± 0.0012 (p < 0.001)."
  );
  card12b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s12.getNotesPage().getSpeakerNotesShape().getText().setText(
    "On Slide 29, Tent lost all sensitivity; SafeAdapt restored AUROC to 0.932. On MIDOG, CMR eliminated all 14 false silence alarms."
  );

  // --------------------------------------------------------------------------
  // SLIDE 13: Discussion
  // --------------------------------------------------------------------------
  var s13 = presentation.appendSlide();
  setBg(s13, '#FFFFFF');
  addHeader(s13, "Discussion: Why SafeAdapt Succeeds Where Prior TTA Methods Fail", "12. Discussion");
  addFooter(s13, 13);

  var dCards = [
    ["1. Geometric Balance", "Entropic Optimal Transport replaces unconstrained gradient descent with exact marginal transportation. By constraining Q^T 1_B = r, the model cannot seek trivial single-class attractors.", 50, C_NAVY],
    ["2. Biological Grounding", "Beer-Lambert optical density deconvolution anchors adaptation in true physical staining properties, discriminating physiological acellularity from sensor collapse.", 345, C_BLUE],
    ["3. Fail-Closed Runtime Perimeter", "PAC L_inf parameter trust regions (Δp <= 0.0460) and 1.8 ms dual-buffer atomic rollbacks ensure target artifacts cannot degrade certified baseline performance.", 640, C_GREEN]
  ];

  for (var i = 0; i < 3; i++) {
    var cD = s13.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, dCards[i][2], 110, 270, 360);
    cD.getFill().setSolidFill(C_CARD_BG);
    cD.getBorder().getLineFill().setSolidFill(dCards[i][3]);
    cD.getText().setText(dCards[i][0] + "\n\n" + dCards[i][1]);
    cD.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);
  }

  s13.getNotesPage().getSpeakerNotesShape().getText().setText(
    "SafeAdapt succeeds through geometric transport constraints, biological tissue grounding, and a fail-closed systems perimeter."
  );

  // --------------------------------------------------------------------------
  // SLIDE 14: Limitations
  // --------------------------------------------------------------------------
  var s14 = presentation.appendSlide();
  setBg(s14, '#FFFFFF');
  addHeader(s14, "Honest Limitations and Methodological Boundary Conditions", "13. Limitations");
  addFooter(s14, 14);

  var l1 = s14.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 170);
  l1.getFill().setSolidFill('#FEF2F2'); l1.getBorder().getLineFill().setSolidFill(C_RED);
  l1.getText().setText("• Multi-Step Drift Sensitivity\nSafeAdapt requires single-step adaptation (k = 1). Setting k >= 5 triggers representation collapse even with selective gating, as accumulated affine drift overpowers the margin.");
  l1.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var l2 = s14.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 170);
  l2.getFill().setSolidFill('#FEF2F2'); l2.getBorder().getLineFill().setSolidFill(C_RED);
  l2.getText().setText("• Class Prior Estimation\nOptimal transport requires a coarse estimate of the target class marginal prior (r). While robust to ±15% estimation errors, radical unforeseen prior shifts require adaptive estimation.");
  l2.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var l3 = s14.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 300, 420, 170);
  l3.getFill().setSolidFill('#FEF2F2'); l3.getBorder().getLineFill().setSolidFill(C_RED);
  l3.getText().setText("• Batch Size Lower Bound\nOptimal Transport Sinkhorn alignment operates over mini-batches (B >= 32) to compute empirical margins; single-sample (B = 1) streaming requires running buffer memory.");
  l3.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var l4 = s14.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 300, 420, 170);
  l4.getFill().setSolidFill('#FEF2F2'); l4.getBorder().getLineFill().setSolidFill(C_RED);
  l4.getText().setText("• Architectural Scope\nEvaluated rigorously on ResNet-18 (Camelyon17) and Faster R-CNN R50-FPN (MIDOG 2021). Deeper Vision Transformers (e.g., UNI, Virchow) are verified in code but require further compute.");
  l4.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s14.getNotesPage().getSpeakerNotesShape().getText().setText(
    "In scientific transparency: SafeAdapt requires single-step updates (k = 1), requires a coarse class prior estimate, and operates on mini-batches of at least 32 patches."
  );

  // --------------------------------------------------------------------------
  // SLIDE 15: Future Work
  // --------------------------------------------------------------------------
  var s15 = presentation.appendSlide();
  setBg(s15, '#FFFFFF');
  addHeader(s15, "Future Work: Scaling SafeAdapt to Next-Generation Pathology AI", "14. Future Roadmap");
  addFooter(s15, 15);

  var f1 = s15.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 420, 360);
  f1.getFill().setSolidFill(C_CARD_BG); f1.getBorder().getLineFill().setSolidFill(C_NAVY);
  f1.getText().setText(
    "Technical Roadmap\n\n" +
    "• Pathology Foundation Models:\n" +
    "   Extend SafeAdapt-OT to gigapixel vision transformers (UNI, Prov-GigaPath, Virchow) to adapt self-supervised patch tokens.\n\n" +
    "• Online Prior Estimation:\n" +
    "   Implement Bayesian Dirichlet-multinomial tracking to dynamically infer target marginals r_t during continuous streaming.\n\n" +
    "• Multi-Organ Generalization:\n" +
    "   Validate across colorectal, prostate, and breast biopsies."
  );
  f1.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var f2 = s15.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 490, 110, 420, 360);
  f2.getFill().setSolidFill(C_CARD_BG); f2.getBorder().getLineFill().setSolidFill(C_BLUE);
  f2.getText().setText(
    "Clinical Translation\n\n" +
    "• Scanner-Side Firmware Integration:\n" +
    "   Partnering to embed SafeAdapt directly into whole-slide scanner processing engines (e.g., Philips, Hamamatsu) for real-time inference during image acquisition.\n\n" +
    "• Multiplex Staining Pipelines:\n" +
    "   Evaluate adaptation on immunohistochemical (IHC) markers and multiplex fluorescence cohorts.\n\n" +
    "• Regulatory SaMD Engagement:\n" +
    "   Formulating certified safety dossiers for clinical software clearance."
  );
  f2.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s15.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Our future roadmap extends SafeAdapt to gigapixel foundation models like UNI and embeds the algorithm directly into scanner firmware."
  );

  // --------------------------------------------------------------------------
  // SLIDE 16: Conclusion
  // --------------------------------------------------------------------------
  var s16 = presentation.appendSlide();
  setBg(s16, '#FFFFFF');
  addHeader(s16, "Conclusion: Transforming TTA into a Certified Clinical Reality", "15. Conclusion");
  addFooter(s16, 16);

  var cConc = s16.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 860, 360);
  cConc.getFill().setSolidFill('#F0FDF4');
  cConc.getBorder().getLineFill().setSolidFill(C_GREEN);
  cConc.getText().setText(
    "SUMMARY OF RESEARCH ACHIEVEMENTS:\n\n" +
    "• The Vulnerability: Unconstrained entropy minimization collapses on 40% of random seeds in digital pathology, and MC-Dropout uncertainty is mathematically redundant with softmax confidence (ρ = -0.9917).\n\n" +
    "• The Solution: SafeAdapt couples Entropic Optimal Transport marginal constraints (Q^T 1_B = r) with Beer-Lambert tissue gating and PAC L_inf parameter trust regions.\n\n" +
    "• The Clinical Impact: Elevates 5-seed cohort performance to 0.9495 AUROC (p = 0.0312), completely reverses collapse on outlier Slide 29 (0.0521 -> 0.9324), eliminates 14 false alarms on MIDOG, and slashes harmful adaptation by >100x (0.18%).\n\n" +
    "• Closing Verdict: SafeAdapt proves that test-time adaptation can transition from an unpredictable empirical gamble into a mathematically certified, clinically deployable reality."
  );
  cConc.getText().getTextStyle().setFontSize(12).setForegroundColor(C_TEXT);

  s16.getNotesPage().getSpeakerNotesShape().getText().setText(
    "SafeAdapt transforms TTA from an unpredictable empirical gamble into a mathematically certified, clinically deployable reality."
  );

  // --------------------------------------------------------------------------
  // SLIDE 17: Key Takeaways
  // --------------------------------------------------------------------------
  var s17 = presentation.appendSlide();
  setBg(s17, '#FFFFFF');
  addHeader(s17, "Three Key Takeaways for Machine Learning Researchers and Clinicians", "16. Key Takeaways");
  addFooter(s17, 17);

  var t17a = s17.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 110, 860, 100);
  t17a.getFill().setSolidFill(C_CARD_BG); t17a.getBorder().getLineFill().setSolidFill(C_NAVY);
  t17a.getText().setText("1. Attractor Collapse is a Solvable Geometric Defect\nEntropy minimization collapses because it lacks marginal constraints; Entropic Optimal Transport guarantees balanced class allocation by solving Q^T 1_B = r.");
  t17a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var t17b = s17.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 230, 860, 100);
  t17b.getFill().setSolidFill(C_CARD_BG); t17b.getBorder().getLineFill().setSolidFill(C_BLUE);
  t17b.getText().setText("2. MC-Dropout Uncertainty is Unnecessary in 1D Heads\nFirst-order Taylor expansion proves predictive variance is rank-equivalent to softmax confidence (ρ = -0.9917). Eliminate it to save 3.55x computational inference penalty.");
  t17b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  var t17c = s17.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 50, 350, 860, 100);
  t17c.getFill().setSolidFill(C_CARD_BG); t17c.getBorder().getLineFill().setSolidFill(C_GREEN);
  t17c.getText().setText("3. Fail-Closed Safeguards are Mandatory for Patient Safety\nParameter displacement bounding (||Δθ||_inf <= 0.05), Beer-Lambert cellularity gating, and episodic per-WSI resets are non-negotiable requirements for patient diagnostic isolation.");
  t17c.getText().getTextStyle().setFontSize(11).setForegroundColor(C_TEXT);

  s17.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Please leave today with these 3 insights: geometry fixes collapse, MC-Dropout is redundant, and fail-closed safeguards are mandatory."
  );

  // --------------------------------------------------------------------------
  // SLIDE 18: Thank You
  // --------------------------------------------------------------------------
  var s18 = presentation.appendSlide();
  setBg(s18, C_NAVY);

  var t18 = s18.insertTextBox("Thank You! Questions & Discussion", 60, 60, 840, 60);
  t18.getText().getTextStyle().setFontSize(28).setBold(true).setForegroundColor(C_WHITE);

  var sub18 = s18.insertTextBox("SafeAdapt: Clinically Safe Test-Time Adaptation in Digital Pathology", 60, 125, 840, 40);
  sub18.getText().getTextStyle().setFontSize(14).setForegroundColor('#38BDF8');

  var c18a = s18.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 60, 180, 420, 260);
  c18a.getFill().setSolidFill('#173A72'); c18a.getBorder().getLineFill().setSolidFill('#264E8E');
  c18a.getText().setText("RESEARCH TEAM & CONTACTS\n\nAvinash Reddy Cheerapareddy\navinashreddy_cheerapareddy@srmap.edu.in\n\nGunnam Sanjana\nsanjana_gunnam@srmap.edu.in\n\nDepartment of Computer Science and Engineering\nSRM University-AP, Amaravati, Andhra Pradesh, India");
  c18a.getText().getTextStyle().setFontSize(11).setForegroundColor(C_WHITE);

  var c18b = s18.insertShape(SlidesApp.ShapeType.ROUNDED_RECTANGLE, 500, 180, 400, 260);
  c18b.getFill().setSolidFill('#173A72'); c18b.getBorder().getLineFill().setSolidFill('#264E8E');
  c18b.getText().setText("OPEN-SOURCE REPRODUCIBILITY\n\n• Official Zenodo Archive:\n  https://doi.org/10.5281/zenodo.23658430\n\n• GitHub Repository:\n  https://github.com/Avirech7676/UTTA-Med\n\n• Bit-Exact Verification: 57/57 Passed\n• Automated Pytest Suite: 148 Passed Tests\n• Target: IEEE Transactions on Medical Imaging");
  c18b.getText().getTextStyle().setFontSize(11).setForegroundColor(C_WHITE);

  s18.getNotesPage().getSpeakerNotesShape().getText().setText(
    "Thank you all for your attention. My co-author Gunnam Sanjana and I now welcome your questions."
  );
}
