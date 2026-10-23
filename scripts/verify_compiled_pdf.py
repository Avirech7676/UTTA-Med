import pypdf
import sys

import unicodedata

def verify_pdf():
    pdf_path = 'FINAL_MANUSCRIPT.pdf'
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"=== FINAL_MANUSCRIPT.pdf VERIFICATION ===")
    print(f"Total Pages: {num_pages}")
    
    full_text = ""
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        text = unicodedata.normalize('NFKD', raw_text)
        full_text += f"\n--- PAGE {i+1} ---\n" + text
        print(f"Page {i+1}: Length = {len(text)} characters")
        
    # Check for crucial elements
    checks = [
        ("Title", "Biologically Informed Fail-Closed Test-Time"),
        ("Abstract", "Abstract"),
        ("Track A Classification", "Track A (Classification)"),
        ("Confidence Gating (C1)", "Confidence Gating"),
        ("Table I Header", "TABLE I"),
        ("Table I Content", "Full Target Cohort"),
        ("Table II Header", "TABLE II"),
        ("Table II Content", "Canonical 131-Mitosis Unfiltered Manifest"),
        ("Bio-SafeAdapt", "Bio-SafeAdapt"),
        ("Parsimonious D8", "Parsimonious Champion"),
        ("D8 vs D9", "Parsimonious Equivalence of D8 and D9"),
        ("Corrected p-value 0.0087", "0087"),
        ("Corrected p-value 0.0043", "0043"),
        ("Corrected CI [+0.0061, +0.0143]", "0061"),
        ("Effect Size 6.16", "6.16"),
        ("References Section", "REFERENCES"),
        ("Citations 1-12", "[12]")
    ]
    
    print("\n=== CONTENT INTEGRITY CHECKS ===")
    all_passed = True
    for name, pattern in checks:
        if pattern in full_text:
            print(f"[PASS] {name}: found '{pattern}'")
        else:
            print(f"[FAIL] {name}: '{pattern}' NOT found in PDF text")
            all_passed = False
            
    print(f"\nFinal PDF Verification Result: {'ALL CHECKS PASSED' if all_passed else 'FAILURES DETECTED'}")
    return all_passed

if __name__ == '__main__':
    ok = verify_pdf()
    if not ok:
        sys.exit(1)
