import requests
import pypdf
import os
import sys

def compile_manuscript():
    tex_path = '20_FINAL_MANUSCRIPT.tex'
    pdf_output_path = 'FINAL_MANUSCRIPT.pdf'
    
    print(f"Reading LaTeX file: {tex_path}")
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()
        
    print(f"LaTeX document size: {len(tex_content)} characters, {len(tex_content.splitlines())} lines.")
    
    # Verify no accidental markdown bold or unescaped tabs
    if '**' in tex_content:
        raise ValueError("Found accidental markdown bold '**' in LaTeX document!")
    
    print("Sending compilation request to LaTeX.Online compiler...")
    url = 'https://latexonline.cc/compile'
    params = {'text': tex_content}
    
    try:
        response = requests.get(url, params=params, timeout=120)
    except Exception as e:
        print(f"Compilation request failed: {e}")
        return False
        
    print(f"HTTP Status: {response.status_code}")
    if response.status_code == 200 and response.content.startswith(b'%PDF'):
        with open(pdf_output_path, 'wb') as f:
            f.write(response.content)
        print(f"SUCCESS: Written compiled PDF to {pdf_output_path} ({len(response.content)} bytes).")
        
        # Verify with pypdf
        reader = pypdf.PdfReader(pdf_output_path)
        num_pages = len(reader.pages)
        print(f"Verified PDF with PyPDF: {num_pages} pages successfully rendered.")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"  Page {i+1}: ~{len(text)} characters extracted.")
            
        # Copy to paper/ieee/ if desired
        ieee_pdf_path = os.path.join('paper', 'ieee', 'FINAL_MANUSCRIPT.pdf')
        with open(ieee_pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"Copied to {ieee_pdf_path}.")
        return True
    else:
        print("Compilation failed or returned error log:")
        print(response.text[:2000])
        return False

if __name__ == '__main__':
    success = compile_manuscript()
    if not success:
        sys.exit(1)
