"""
Production LaTeX Manuscript Compilation Pipeline for UTTA-Med.
Compiles paper/ieee/utta_med_h5_manuscript_draft.tex and bundles publication figures into paper/ieee/utta_med_h5_manuscript_draft.pdf.
"""

import io
import tarfile
import requests
import os
import pypdf

def compile_manuscript():
    tex_path = os.path.join('paper', 'ieee', 'utta_med_h5_manuscript_draft.tex')
    pdf_out_path = os.path.join('paper', 'ieee', 'utta_med_h5_manuscript_draft.pdf')
    
    print(f"Reading LaTeX file: {tex_path}")
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()

    # Package files into bzip2 tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:bz2') as tar:
        ti = tarfile.TarInfo(name='utta_med_h5_manuscript_draft.tex')
        encoded = tex_content.encode('utf-8')
        ti.size = len(encoded)
        tar.addfile(ti, io.BytesIO(encoded))
        
        figs = [
            'fig_v2_slide29_attractor_reversal.jpg',
            'fig_v2_benchmark_auroc_ece.jpg',
            'fig_v2_cmr_mitosis_detection.jpg',
            'fig_v2_gpu_training_convergence.jpg'
        ]
        for f in figs:
            p = os.path.join('paper', 'ieee', 'figs', f)
            if os.path.exists(p):
                tar.add(p, arcname=f'figs/{f}')
            else:
                print(f"Warning: figure {p} not found.")

    buf.seek(0)
    data = buf.getvalue()
    print(f"Tarball created: {len(data)} bytes. Submitting to compiler...", flush=True)

    url = 'https://latexonline.cc/data?target=utta_med_h5_manuscript_draft.tex&command=pdflatex&force=true'
    r = requests.post(url, files={'file': ('archive.tar.bz2', data, 'application/x-bzip2')}, timeout=180)
    
    if r.status_code == 200 and r.content.startswith(b'%PDF'):
        with open(pdf_out_path, 'wb') as f:
            f.write(r.content)
        
        cameraready_out = os.path.join('paper', 'ieee', 'utta_med_camera_ready.pdf')
        with open(cameraready_out, 'wb') as f:
            f.write(r.content)
            
        print(f"SUCCESS: Compiled PDF saved to {pdf_out_path} ({len(r.content)} bytes).", flush=True)
        print(f"SUCCESS: Camera-ready PDF saved to {cameraready_out} ({len(r.content)} bytes).", flush=True)
        
        reader = pypdf.PdfReader(pdf_out_path)
        print(f"Verified PDF with pypdf: {len(reader.pages)} pages rendered successfully.", flush=True)
        return True
    else:
        print(f"Compilation failed with status {r.status_code}:", flush=True)
        print(r.text[-1000:], flush=True)
        return False

def compile_supplement():
    tex_path = os.path.join('paper', 'ieee', 'supplement.tex')
    pdf_out_path = os.path.join('paper', 'ieee', 'supplement.pdf')
    
    print(f"\nReading Supplement LaTeX file: {tex_path}")
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:bz2') as tar:
        ti = tarfile.TarInfo(name='supplement.tex')
        encoded = tex_content.encode('utf-8')
        ti.size = len(encoded)
        tar.addfile(ti, io.BytesIO(encoded))
        
    buf.seek(0)
    data = buf.getvalue()
    print(f"Supplement tarball created: {len(data)} bytes. Submitting to compiler...", flush=True)

    url = 'https://latexonline.cc/data?target=supplement.tex&command=pdflatex&force=true'
    r = requests.post(url, files={'file': ('archive.tar.bz2', data, 'application/x-bzip2')}, timeout=180)
    
    if r.status_code == 200 and r.content.startswith(b'%PDF'):
        with open(pdf_out_path, 'wb') as f:
            f.write(r.content)
            
        print(f"SUCCESS: Compiled Supplement saved to {pdf_out_path} ({len(r.content)} bytes).", flush=True)
        reader = pypdf.PdfReader(pdf_out_path)
        print(f"Verified Supplement PDF with pypdf: {len(reader.pages)} pages rendered successfully.", flush=True)
        return True
    else:
        print(f"Supplement compilation failed with status {r.status_code}:", flush=True)
        print(r.text[-1000:], flush=True)
        return False

if __name__ == '__main__':
    compile_manuscript()
    compile_supplement()

