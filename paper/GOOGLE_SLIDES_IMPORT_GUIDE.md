# Google Slides Design & Import Guide for SafeAdapt

This guide explains how to open and present the **SafeAdapt** presentation directly inside **Google Slides**.

You have **two instantaneous, zero-effort methods**:
- **Method 1 (Fastest & Recommended):** Open the pre-built, high-resolution widescreen presentation file [`paper/SafeAdapt_Google_Slides.pptx`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/paper/SafeAdapt_Google_Slides.pptx) directly in Google Slides.
- **Method 2 (1-Click Native Script):** Run the Google Apps Script [`paper/google_slides_script.js`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/paper/google_slides_script.js) inside Google Slides to automatically generate all 18 slides with cards, styles, and speaker notes.

---

## Method 1: Direct Import into Google Slides (10 Seconds)

1. Open your browser and go to: [https://drive.google.com](https://drive.google.com)
2. Drag and drop the file:
   `c:\Users\avina\OneDrive\Desktop\UTTA_MED_V1_FINAL\UTTA_MED_V1_FINAL\paper\SafeAdapt_Google_Slides.pptx`
   into your Google Drive window.
3. Once uploaded, double-click the file and select:
   **"Open with Google Slides"** (or in Google Slides, click **File $\to$ Open $\to$ Upload**).
4. **Done!** Google Slides will automatically preserve:
   - Full 16:9 widescreen layout.
   - High-resolution embedded histology and benchmark figures from the paper.
   - Professional Navy/Blue/Red/Green card containers and benchmark tables.
   - **All 18 slide-by-slide speaker notes** in the bottom Speaker Notes pane.

---

## Method 2: Generate Natively via Google Apps Script

If you want Google Slides to build the slides natively in your Google account using code:

1. Open a new Google Slide deck by typing: [https://slides.new](https://slides.new) into your browser.
2. In the top navigation bar, click: **Extensions $\longrightarrow$ Apps Script**.
3. Delete any boilerplate text in the editor (`function myFunction() { ... }`).
4. Open [`paper/google_slides_script.js`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/paper/google_slides_script.js), copy the entire contents, and paste it into the editor.
5. Click the **Save** (disk) icon, then click the **Run** button at the top.
6. When prompted for authorization:
   - Click **Review permissions**.
   - Choose your Google account.
   - Click **Advanced** (bottom left), then click **Go to Untitled project (unsafe)**.
   - Click **Allow**.
7. Return to your Google Slides tab: All 18 slides with colored cards, benchmark tables, and speaker notes are generated instantly!

---

## Slide Deck Overview (18 Slides)

```
┌──────┬──────────────────────────────────────────┬─────────────────────────────┬───────────────────────────┐
│ #    │ Slide Title                              │ Visual Style / Card Type    │ Graphic / Figure Included │
├──────┼──────────────────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ 1    │ Title & Open-Science Provenance          │ Dark Navy Hero Card         │ Institutional Emblem / QR │
│ 2    │ Clinical Motivation & Transfer Gap       │ Problem Cards + Red Callout │ Camelyon17 Metric Callout │
│ 3    │ Research Gap: Attractor Collapse         │ Split Card + Red Alert Box  │ fig_tent_collapse_mechan. │
│ 4    │ Research Question & 3 Objectives         │ Blue Central Callout + 3 Col│ 3 Objective Pillars       │
│ 5    │ 4 Core Contributions Overview            │ 4-Quadrant Bento Grid Cards │ Theory, Algo, Bio, System │
│ 6    │ SafeAdapt Architecture & Pipeline        │ Horizontal Pipeline Cards   │ fig_architecture.png      │
│ 7    │ Method: Entropic Optimal Transport       │ Math Card + Green Success   │ fig_v2_slide29_attractor  │
│ 8    │ Method: Biological Tissue Safeguards     │ Dual Card: CMR & PAC Drift  │ Beer-Lambert & Trust Sph. │
│ 9    │ Experimental Setup (2 Benchmarks)        │ Dual Cards (Cam17 & MIDOG)  │ Dataset Topology Split    │
│ 10   │ Primary Results: 5-Seed Benchmark        │ Styled Table + Green Proof  │ fig_v2_benchmark_auroc    │
│ 11   │ Ablation: Gating & Taylor Disproof       │ 3-Point Card + Red Alert    │ fig_u_error.png           │
│ 12   │ Qualitative Case Studies (Slide 29)      │ Dual Case Bento Cards       │ fig_v2_cmr_mitosis_detect │
│ 13   │ Discussion: Why SafeAdapt Succeeds       │ 3 Pillars: Geometry, Bio    │ Architectural Schematic   │
│ 14   │ Honest Operational Limitations           │ 4 Crimson Border Warning Box│ k=1, Prior, B>=32 bounds  │
│ 15   │ Future Work: Foundation Models & Firmware│ Dual Future Roadmap Cards   │ 2026-2028 Strategic Plan  │
│ 16   │ Conclusion & Synthesis                   │ Emerald Green Summary Card  │ Before-After Verdict Box  │
│ 17   │ 3 Key Takeaways for Clinicians           │ 3 Numbered Takeaway Cards   │ Big Number Badges 1, 2, 3 │
│ 18   │ Thank You & Discussion (Q&A)             │ Dark Navy Hero Card         │ Team Contacts & Zenodo DOI│
└──────┴──────────────────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```
