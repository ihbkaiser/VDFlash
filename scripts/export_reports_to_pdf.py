#!/usr/bin/env python3
"""Export all markdown reports in results/6_10_26/ to beautiful, full-featured PDFs with images and math."""

import os
import glob
import re
import subprocess
from pathlib import Path
import markdown

REPORTS_DIR = Path("results/6_10_26")
MATHJAX_PATH = (Path("scripts/assets/tex-chtml-full.js")).resolve()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<style>
  @page {
    size: A4;
    margin: 1.8cm 1.5cm 2.0cm 1.5cm;
    @bottom-right {
      content: counter(page);
    }
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #24292e;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }
  h1 {
    font-size: 18pt;
    font-weight: 700;
    color: #1a365d;
    border-bottom: 2.5px solid #2b6cb0;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 16px;
    page-break-after: avoid;
    break-after: avoid;
  }
  h2 {
    font-size: 13.5pt;
    font-weight: 600;
    color: #2b6cb0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 22px;
    margin-bottom: 12px;
    page-break-after: avoid;
    break-after: avoid;
  }
  h3 {
    font-size: 11.5pt;
    font-weight: 600;
    color: #2d3748;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
    break-after: avoid;
  }
  h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #4a5568;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
    break-after: avoid;
  }
  p {
    margin-top: 0;
    margin-bottom: 10px;
    text-align: justify;
  }
  ul, ol {
    margin-top: 0;
    margin-bottom: 10px;
    padding-left: 24px;
  }
  li {
    margin-bottom: 4px;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 14px;
    margin-bottom: 16px;
    font-size: 8.5pt;
    line-height: 1.35;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  tr {
    page-break-inside: avoid;
    break-inside: avoid;
  }
  th, td {
    border: 1px solid #cbd5e0;
    padding: 5px 7px;
    text-align: left;
    word-break: break-word;
    overflow-wrap: break-word;
  }
  th {
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: 600;
  }
  tr:nth-child(even) td {
    background-color: #f7fafc;
  }
  code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    font-size: 9pt;
    background-color: #edf2f7;
    color: #c53030;
    padding: 2px 4px;
    border-radius: 3px;
  }
  pre {
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 10px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.4;
    page-break-inside: avoid;
    break-inside: avoid;
    white-space: pre-wrap;
    word-break: break-word;
  }
  pre code {
    background-color: transparent;
    color: #2d3748;
    padding: 0;
    font-size: inherit;
  }
  blockquote {
    border-left: 4px solid #3182ce;
    margin: 12px 0;
    padding: 8px 16px;
    background-color: #ebf8ff;
    color: #2c5282;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 20px 0;
  }
  .image-container {
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  img {
    max-width: 95%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .caption {
    font-size: 8.5pt;
    color: #718096;
    font-style: italic;
    margin-top: 6px;
    text-align: center;
  }
  mjx-container {
    page-break-inside: avoid;
    break-inside: avoid;
  }
</style>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async src="file://__MATHJAX_PATH__"></script>
</head>
<body>
__CONTENT__
</body>
</html>
"""

def convert_md_to_html(md_path: Path) -> Path:
    raw_md = md_path.read_text(encoding="utf-8")
    
    # Process image links so they use absolute file:/// URIs
    def replace_img(match):
        alt = match.group(1)
        src = match.group(2)
        if not src.startswith("http") and not src.startswith("file://") and not src.startswith("/"):
            abs_img = (md_path.parent / src).resolve()
            src = f"file://{abs_img}"
        return f'<div class="image-container"><img src="{src}" alt="{alt}"/></div>'
    
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img, raw_md)
    
    # Protect LaTeX expressions from Python-Markdown mangling
    math_placeholders = []
    def save_math(match):
        idx = len(math_placeholders)
        math_placeholders.append(match.group(0))
        return f"MATHPH{idx}ENDPH"
    
    content = re.sub(r'\$\$(.*?)\$\$', save_math, content, flags=re.DOTALL)
    content = re.sub(r'\$([^\$\n]+?)\$', save_math, content)
    
    # Render with Python-Markdown
    html_body = markdown.markdown(
        content,
        extensions=[
            'tables',
            'fenced_code',
            'def_list',
            'sane_lists',
            'nl2br'
        ]
    )
    
    # Restore LaTeX expressions for MathJax
    for idx, math_str in enumerate(math_placeholders):
        html_body = html_body.replace(f"MATHPH{idx}ENDPH", math_str)
    
    full_html = HTML_TEMPLATE.replace("__CONTENT__", html_body).replace("__MATHJAX_PATH__", str(MATHJAX_PATH))
    
    out_html_path = md_path.with_suffix(".temp.html")
    out_html_path.write_text(full_html, encoding="utf-8")
    return out_html_path

def export_pdf_via_chrome(html_path: Path, pdf_path: Path):
    cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=8000",
        "--allow-file-access-from-files",
        "--enable-local-file-accesses",
        f"--print-to-pdf={pdf_path.resolve()}",
        str(html_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error printing {html_path}: {res.stderr}")
    else:
        print(f"Successfully generated: {pdf_path} (Size: {pdf_path.stat().st_size:,} bytes)")

def main():
    md_files = sorted(REPORTS_DIR.glob("BAO_CAO_*.md"))
    print(f"Found {len(md_files)} reports to export:")
    for f in md_files:
        print(f" - {f.name}")
    
    for md_path in md_files:
        print(f"\nProcessing {md_path.name}...")
        html_path = convert_md_to_html(md_path)
        pdf_path = md_path.with_suffix(".pdf")
        export_pdf_via_chrome(html_path, pdf_path)
        if html_path.exists():
            html_path.unlink()  # Clean up temporary HTML
            
    print("\nAll PDFs exported successfully!")

if __name__ == "__main__":
    main()
