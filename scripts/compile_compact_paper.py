import requests
import tarfile
import io
import pypdf
import os
import sys

def compile_compact_paper():
    tex_path = 'UTTA_MED_FINAL_COMPACT_PAPER.tex'
    pdf_out = 'UTTA_MED_FINAL_COMPACT_PAPER.pdf'
    
    print(f"Reading {tex_path}...")
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()
        
    print(f"Document size: {len(tex_content)} characters, {len(tex_content.splitlines())} lines.")
    
    # Check for accidental markdown syntax
    if '**' in tex_content:
        raise ValueError("Found accidental markdown syntax '**' in LaTeX document!")
        
    print("Building tarball archive with figures...")
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode='w') as tar:
        tar.add(tex_path, arcname='main.tex')
        for f in ['fig_d5_coverage_risk.png', 'fig_d6_slide29_seed7_hist.png', 'fig_d2_tumor_burden.png']:
            img_path = os.path.join('figs', f)
            if os.path.exists(img_path):
                tar.add(img_path, arcname=f'figs/{f}')
                print(f"  Added figs/{f} ({os.path.getsize(img_path)} bytes)")
            else:
                print(f"  Warning: {img_path} not found!")
    tar_buf.seek(0)
    
    print(f"Total tarball size: {len(tar_buf.getvalue())} bytes.")
    
    print("Sending compilation request to LaTeX.Online...")
    files = {'file': ('project.tar', tar_buf.getvalue(), 'application/x-tar')}
    url = 'https://latexonline.cc/data?target=main.tex'
    
    try:
        r = requests.post(url, files=files, timeout=120)
    except Exception as e:
        print(f"Compilation request failed: {e}")
        return False
        
    print(f"HTTP Status: {r.status_code}")
    if r.status_code == 200 and r.content.startswith(b'%PDF'):
        with open(pdf_out, 'wb') as f:
            f.write(r.content)
        print(f"SUCCESS: Generated {pdf_out} ({len(r.content)} bytes).")
        
        reader = pypdf.PdfReader(pdf_out)
        print(f"Verified PDF: {len(reader.pages)} pages successfully rendered.")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"  Page {i+1}: ~{len(text)} characters extracted.")
        return True
    else:
        print("Compilation failed or returned error log:")
        print(r.text[-3000:])
        return False

if __name__ == '__main__':
    ok = compile_compact_paper()
    if not ok:
        sys.exit(1)
