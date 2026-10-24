# CLEANED UTTA-MED REPOSITORY STRUCTURE

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Date:** October 2026 | **Audit Base:** Git Commit `0fe5147` | **Branch:** `research-project-2`

---

## 1. EXECUTIVE SUMMARY

The active UTTA-Med Project 2 repository has been deeply reorganized and purged of all unauthorized extensions, obsolete intermediate files, duplicate reports, and internal archives. 
All recoverable unwanted development has been preserved in the external archive located strictly outside the repository at:
`C:\Users\avina\Documents\UTTA_MED_RESEARCH_ARCHIVE`

---

## 2. ACTIVE REPOSITORY DIRECTORY LAYOUT

```
UTTA_MED_V1_FINAL/
│
├── README.md                                  # Primary repository overview & execution instructions
├── LICENSE                                    # Official project license
├── requirements.txt                           # Core runtime Python dependencies
├── requirements-optional.txt                  # Optional benchmark dependencies
├── environment.yml                            # Conda environment definition
├── pytest.ini                                 # Test configuration
├── SETUP.ps1                                  # Local Windows execution setup script
│
├── UTTA_MED_FINAL_COMPACT_PAPER.pdf           # Authoritative compact manuscript build (SHA-256 verified)
├── FINAL_MANUSCRIPT.pdf                       # Authoritative full manuscript build (SHA-256 verified)
├── UTTA_MED_FINAL_COMPACT_PAPER.tex           # Primary LaTeX manuscript source
├── 20_FINAL_MANUSCRIPT.tex                    # Comprehensive LaTeX manuscript source
│
├── COMPLETE_REPOSITORY_FILE_AUDIT.csv         # Master audit of all active repository files
├── UNWANTED_FILES_CLASSIFICATION.csv          # Register of all classified unwanted files
├── EXTERNAL_ARCHIVE_MANIFEST.csv              # Cryptographic manifest of all externally archived files
├── DELETED_FILES_REGISTER.csv                 # Audit log of confirmed purged disposable files
├── PRESERVED_CORE_FILES.csv                   # Register of all certified core research files
├── CLEANED_REPOSITORY_STRUCTURE.md            # Structural layout and directory documentation
├── DEPENDENCY_VERIFICATION_REPORT.md          # Dependency integrity and import verification report
├── REGRESSION_TEST_RESULTS.md                 # Full regression test pass-through audit
├── FINAL_REPOSITORY_CLEANUP_REPORT.md         # Final comprehensive repository cleanup report
│
├── checkpoints/                               # Certified source models and centroids
│   ├── camelyon17_resnet18_source_s*.pt       # 5 certified Camelyon17 ResNet-18 source checkpoints
│   ├── midog_resnet18_s42_BEST.pt             # MIDOG 2021 primary detection checkpoint
│   ├── source_centroids_resnet18_s*.pt        # Source latent class prototypes
│   └── randstainna/                           # Validated optical density stain augmentation model
│
├── src/                                       # Core implementation modules
│   ├── models/resnet18.py                     # ResNet-18 feature extractor & 1-logit BCE head
│   ├── tta/                                   # Test-time adaptation controllers
│   │   ├── uncertainty_gated.py               # Feature-cached MC-Dropout selective adaptation
│   │   ├── confidence_gated.py                # Deterministic confidence threshold gating
│   │   ├── tent.py                            # Standard Tent entropy minimization baseline
│   │   ├── eata.py                            # Energy-aware / entropy-filtered adaptation
│   │   └── safety.py                          # Fail-closed drift alarms & episodic rollbacks
│   ├── data/                                  # Pathology dataset loaders & stain transforms
│   ├── calibration/                           # Temperature scaling & threshold calibration
│   └── uncertainty/                           # Epistemic & mutual information estimators
│
├── results/                                   # Authentic experimental results & predictions
│   ├── data/wsi_bootstrap/                    # 35 raw prediction arrays (N=85,054 patches each)
│   ├── generalization/h3/                     # MIDOG Scanner 1, 2, 3 detection predictions & overlays
│   ├── statistical/                           # Five-seed bootstrap distributions & paired statistics
│   ├── ablations/                             # MC-Dropout pass sweeps & matched budget curves
│   ├── optional/                              # Optional MedMNIST-C, Camelyon17-C, PCam benchmarks
│   ├── upgrade_p0/                            # Phase 0 slide-level aggregation and tumor burden data
│   └── tables/                                # Canonical manuscript tables & evaluation summaries
│
├── reproducibility/                           # Certified research reproducibility suite
│   ├── requirements.txt                       # Locked reproducibility dependencies
│   ├── requirements-lock.txt                  # Full dependency hash lock
│   ├── environment.yml                        # Exact Conda reproduction environment
│   ├── reproduction_commands.md               # End-to-end command-line reproduction instructions
│   ├── hardware.txt                           # Execution environment hardware audit
│   ├── checksums.sha256                       # Cryptographic checksums of all core checkpoints
│   ├── GIT_COMMIT.txt                         # Authoritative commit tag
│   ├── ARTIFACT_MANIFEST.csv                  # Manifest of authentic research artifacts
│   └── REPRODUCIBILITY_REPORT.md              # Certified reproducibility audit
│
├── configs/                                   # Clean model and evaluation configuration files
├── scripts/                                   # Verification, profiling, and benchmarking runners
├── tests/                                     # Certified pytest test suite (73 passed, 0 failed)
├── paper/                                     # Manuscript assets, figures, and bibtex references
├── figures/                                   # Canonical publication vector figures
├── figs/                                      # Auxiliary figure plots
└── docs/                                      # Research documentation
    ├── audit/                                 # Certified performance reports & discrepancy resolutions
    └── final/                                 # Final evidence inventory & R&D manual
```

---

## 3. ROOT DIRECTORY INTEGRITY
The root directory has been reduced from 155 scattered files to essential project infrastructure, authoritative papers, and master cleanup audit deliverables.
