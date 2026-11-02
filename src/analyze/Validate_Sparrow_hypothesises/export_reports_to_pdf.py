#!/usr/bin/env python3
"""
Convert all DFlash markdown reports to beautifully formatted PDF documents
using xhtml2pdf with embedded DejaVuSans TrueType fonts for 100% Unicode, Math, and Table layout support.
Supports automatic full-width Landscape pages for large composite figures and maximum-width Portrait pages.
"""

import re
from pathlib import Path
import markdown
from xhtml2pdf import pisa

BASE_DIR = Path("markdown/Baocao_DFlash_4Config_20261029")
FONTS_DIR = BASE_DIR / "fonts"

REPORTS = [
    "tong_hop_so_sanh_4_config_dflash.md",
    "bao_cao_dflash_3layer_high_only_26_28_30_32_34.md",
    "bao_cao_dflash_3layer_middle_high_17_19_25_33_34.md",
    "bao_cao_dflash_3layer_middle_high_2_17_28_30_32_34.md",
    "bao_cao_dflash_3layer_shallow_high_1_28_30_32_34.md",
    "bao_cao_dflash_3layer_balanced_1_9_17_25_33.md",
]

CSS_TEMPLATE = f"""
@font-face {{
    font-family: 'DejaVuSans';
    src: url('{FONTS_DIR / "DejaVuSans.ttf"}');
}}
@font-face {{
    font-family: 'DejaVuSans';
    src: url('{FONTS_DIR / "DejaVuSans-Bold.ttf"}');
    font-weight: bold;
}}
@font-face {{
    font-family: 'DejaVuSans';
    src: url('{FONTS_DIR / "DejaVuSans-Oblique.ttf"}');
    font-style: italic;
}}

@page portrait {{
    size: a4 portrait;
    margin: 0.8cm 0.8cm 1.1cm 0.8cm;
    @bottom-right {{
        content: "Trang " counter(page);
        font-family: 'DejaVuSans', sans-serif;
        font-size: 7.5pt;
        color: #718096;
    }}
    @bottom-left {{
        content: "Báo Cáo Nghiên Cứu Video-DFlash (Qwen2.5-VL-3B)";
        font-family: 'DejaVuSans', sans-serif;
        font-size: 7.5pt;
        color: #718096;
    }}
}}

@page landscape {{
    size: a4 landscape;
    margin: 0.8cm 0.8cm 1.0cm 0.8cm;
    @bottom-right {{
        content: "Trang " counter(page);
        font-family: 'DejaVuSans', sans-serif;
        font-size: 7.5pt;
        color: #718096;
    }}
    @bottom-left {{
        content: "Báo Cáo Nghiên Cứu Video-DFlash (Qwen2.5-VL-3B)";
        font-family: 'DejaVuSans', sans-serif;
        font-size: 7.5pt;
        color: #718096;
    }}
}}

body, p, table, th, td, div, span, h1, h2, h3, h4, h5, h6, li, pre, code {{
    font-family: 'DejaVuSans', sans-serif;
}}

body {{
    font-size: 8.5pt;
    line-height: 1.42;
    color: #2d3748;
}}

h1 {{
    color: #1a365d;
    font-size: 15pt;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 6px;
    font-weight: bold;
    border-bottom: 2pt solid #2b6cb0;
    padding-bottom: 4px;
}}

h2 {{
    color: #2b6cb0;
    font-size: 11.5pt;
    margin-top: 14px;
    margin-bottom: 6px;
    font-weight: bold;
    border-bottom: 0.8pt solid #cbd5e0;
    padding-bottom: 2px;
    page-break-after: avoid;
}}

h3 {{
    color: #2d3748;
    font-size: 9.8pt;
    margin-top: 11px;
    margin-bottom: 4px;
    font-weight: bold;
    page-break-after: avoid;
}}

h4 {{
    color: #4a5568;
    font-size: 8.8pt;
    margin-top: 8px;
    margin-bottom: 3px;
    font-weight: bold;
    page-break-after: avoid;
}}

p {{
    margin-top: 3px;
    margin-bottom: 5px;
}}

ul, ol {{
    margin-top: 2px;
    margin-bottom: 5px;
    padding-left: 18px;
}}

li {{
    margin-bottom: 2px;
}}

table {{
    width: 100%;
    margin-top: 8px;
    margin-bottom: 10px;
    border-collapse: collapse;
    font-size: 7.2pt;
}}

tr {{
    page-break-inside: avoid;
}}

th {{
    background-color: #2b6cb0;
    color: #ffffff;
    font-weight: bold;
    padding: 3.5px 4.5px;
    font-size: 7.2pt;
    text-align: left;
    border: 0.5pt solid #2b6cb0;
}}

td {{
    border: 0.5pt solid #cbd5e0;
    padding: 3px 4.5px;
    font-size: 7.2pt;
}}

tr:nth-child(even) {{
    background-color: #f7fafc;
}}

code {{
    font-family: 'DejaVuSans', monospace;
    background-color: #edf2f7;
    padding: 1px 3px;
    font-size: 7.5pt;
    color: #805ad5;
    border-radius: 2px;
}}

pre {{
    background-color: #f7fafc;
    border: 0.5pt solid #e2e8f0;
    padding: 6px 8px;
    font-size: 7pt;
    line-height: 1.25;
    page-break-inside: avoid;
    margin-top: 5px;
    margin-bottom: 7px;
}}

.math-block {{
    background-color: #f8fafc;
    border: 0.5pt solid #e2e8f0;
    padding: 5px 10px;
    margin-top: 6px;
    margin-bottom: 8px;
    text-align: center;
    font-size: 8pt;
    color: #1a202c;
    page-break-inside: avoid;
}}

.callout {{
    padding: 6px 10px;
    margin-top: 8px;
    margin-bottom: 10px;
    page-break-inside: avoid;
    font-size: 8pt;
    line-height: 1.38;
}}

.callout-note {{
    border-left: 3pt solid #3182ce;
    background-color: #ebf8ff;
}}

.callout-tip {{
    border-left: 3pt solid #38a169;
    background-color: #f0fff4;
}}

.callout-important {{
    border-left: 3pt solid #dd6b20;
    background-color: #fffaf0;
}}

.callout-warning {{
    border-left: 3pt solid #e53e3e;
    background-color: #fff5f5;
}}

.callout-title {{
    font-weight: bold;
    margin-bottom: 3px;
    font-size: 8.5pt;
}}

.figure-wrapper {{
    text-align: center;
    margin-top: 10px;
    margin-bottom: 12px;
    page-break-inside: avoid;
}}

.figure-wrapper img {{
    border: 0.5pt solid #cbd5e0;
}}

.figure-wrapper-landscape {{
    text-align: center;
    margin-top: 8px;
    margin-bottom: 8px;
}}

.figure-wrapper-landscape img {{
    border: 0.5pt solid #cbd5e0;
}}

.figure-caption {{
    font-size: 8pt;
    font-style: italic;
    color: #4a5568;
    margin-top: 4px;
}}

hr {{
    border: none;
    border-top: 0.5pt solid #cbd5e0;
    margin-top: 12px;
    margin-bottom: 12px;
}}
"""

def clean_math_inner(expr: str) -> str:
    """Recursively clean an inner mathematical string expression."""
    # 1. Clean formatting tags inside braces
    for _ in range(4):
        expr = re.sub(r"\\mathbf\{([^{}]+)\}", r"<b>\1</b>", expr)
        expr = re.sub(r"\\text\{([^{}]+)\}", r"\1", expr)
        expr = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", expr)

    # 2. Math symbols
    expr = expr.replace(r"\tau", "<i>&tau;</i>")
    expr = expr.replace(r"\Delta", "&Delta;")
    expr = expr.replace(r"\ge", "&ge;")
    expr = expr.replace(r"\le", "&le;")
    expr = expr.replace(r"\gg", "&gt;&gt;")
    expr = expr.replace(r"\approx", "&asymp;")
    expr = expr.replace(r"\times", "&times;")
    expr = expr.replace(r"\rightarrow", "&rarr;")
    expr = expr.replace(r"\pm", "&plusmn;")
    expr = expr.replace(r"\equiv", "&equiv;")
    expr = expr.replace(r"\in", "&isin;")
    expr = expr.replace(r"\%", "%")
    expr = expr.replace(r"\_", "_")
    expr = expr.replace(r"\ ", " ")

    # 3. Sum: \sum_{...} or \sum
    expr = re.sub(r"\\sum_\{([^{}]+)\}", r"&Sigma;<sub>\1</sub>", expr)
    expr = re.sub(r"\\sum", r"&Sigma;", expr)

    # 4. Fractions: \frac{A}{B}
    for _ in range(4):
        expr = re.sub(r"\\frac\{([^{}]+|\{[^{}]*\})\}\{([^{}]+|\{[^{}]*\})\}", r"(\1 / \2)", expr)

    # 5. Subscripts & superscripts
    expr = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", expr)
    expr = re.sub(r"_([a-zA-Z0-9]+)", r"<sub>\1</sub>", expr)
    expr = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", expr)
    expr = re.sub(r"\^([a-zA-Z0-9\-]+)", r"<sup>\1</sup>", expr)

    # 6. Clean any remaining backslashes
    expr = expr.replace("\\", "")
    return expr

def clean_math(text: str) -> str:
    """Convert common LaTeX math expressions into clean HTML representations."""
    # 1. Block math: $$ ... $$
    def repl_block_math(m):
        inner = m.group(1).strip()
        cleaned = clean_math_inner(inner)
        return f'\n\n<div class="math-block">{cleaned}</div>\n\n'

    text = re.sub(r"\$\$(.*?)\$\$", repl_block_math, text, flags=re.DOTALL)

    # 2. Inline math: $ ... $
    def repl_inline_math(m):
        inner = m.group(1).strip()
        return clean_math_inner(inner)

    text = re.sub(r"\$([^$]+)\$", repl_inline_math, text)

    # 3. Cleanup stray LaTeX escapes outside math mode
    text = text.replace(r"\%", "%")
    text = text.replace(r"\le", "&le;")
    text = text.replace(r"\ge", "&ge;")
    text = text.replace(r"\tau", "<i>&tau;</i>")
    text = text.replace(r"\Delta", "&Delta;")
    text = text.replace(r"\times", "&times;")
    text = text.replace(r"\approx", "&asymp;")
    text = text.replace(r"\rightarrow", "&rarr;")
    text = text.replace(r"\pm", "&plusmn;")

    return text

def convert_callouts(text: str) -> str:
    """Convert GitHub style alerts (> [!NOTE] ...) to clean HTML div boxes."""
    lines = text.split("\n")
    new_lines = []
    in_callout = False
    callout_type = ""
    callout_lines = []

    def flush_callout():
        nonlocal in_callout, callout_type, callout_lines
        if not in_callout:
            return ""
        type_name = callout_type.lower()
        title_map = {
            "note": "LƯU Ý",
            "tip": "MẸO & ĐIỂM TỐI ƯU",
            "important": "KẾT LUẬN QUAN TRỌNG",
            "warning": "CẢNH BÁO",
        }
        title = title_map.get(type_name, type_name.upper())
        body = "\n".join(callout_lines).strip()
        body_html = markdown.markdown(body, extensions=['tables', 'fenced_code'])
        res = f'<div class="callout callout-{type_name}"><div class="callout-title">{title}</div>{body_html}</div>'
        in_callout = False
        callout_type = ""
        callout_lines = []
        return res

    for line in lines:
        m = re.match(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$", line.strip())
        if m:
            if in_callout:
                new_lines.append(flush_callout())
            in_callout = True
            callout_type = m.group(1).lower()
            callout_lines = []
            continue

        if in_callout:
            if line.startswith(">"):
                content = line[1:].strip()
                callout_lines.append(content)
            elif line.strip() == "":
                callout_lines.append("")
            else:
                new_lines.append(flush_callout())
                new_lines.append(line)
        else:
            new_lines.append(line)

    if in_callout:
        new_lines.append(flush_callout())

    return "\n".join(new_lines)

def process_images(html_text: str, doc_dir: Path) -> str:
    """
    Wrap images with maximum width:
    Large multi-panel composite figures (master_exp* or fig3_attention_dynamics) get a dedicated
    full-width LANDSCAPE page (width="785") for 2.4x larger area and crystal-clear readability.
    Other figures get maximum full-width PORTRAIT display (width="540").
    """
    def repl_img(match):
        alt = match.group(1)
        src = match.group(2)
        full_src = (doc_dir / src).resolve() if not Path(src).is_absolute() else Path(src)
        src_name = Path(src).name

        # Determine if this figure warrants a full-width landscape page
        is_landscape_target = any(k in src_name for k in [
            "master_exp1", "master_exp2a", "master_exp2b", "master_exp3", "master_exp4",
            "fig3_attention_dynamics", "summary_fig"
        ])

        if is_landscape_target:
            # 785pt width on A4 Landscape (796pt printable area)
            return (
                '<pdf:nexttemplate name="landscape" />\n'
                '<pdf:nextpage />\n'
                '<div class="figure-wrapper-landscape">\n'
                f'    <img src="{full_src}" width="785" />\n'
                f'    <div class="figure-caption">{alt}</div>\n'
                '</div>\n'
                '<pdf:nexttemplate name="portrait" />\n'
                '<pdf:nextpage />\n'
            )
        else:
            # 540pt width on A4 Portrait (550pt printable area)
            return (
                '<div class="figure-wrapper">\n'
                f'    <img src="{full_src}" width="540" />\n'
                f'    <div class="figure-caption">{alt}</div>\n'
                '</div>\n'
            )

    pattern = r'<p><img[^>]*alt="([^"]*)"[^>]*src="([^"]*)"[^>]*\/><\/p>'
    html_text = re.sub(pattern, repl_img, html_text)

    # Catch any remaining <img> without wrapping <p>
    pattern2 = r'<img[^>]*alt="([^"]*)"[^>]*src="([^"]*)"[^>]*\/>'
    html_text = re.sub(pattern2, repl_img, html_text)
    return html_text

def convert_md_to_pdf(md_filename: str):
    md_path = BASE_DIR / md_filename
    pdf_filename = md_filename.replace(".md", ".pdf")
    pdf_path = BASE_DIR / pdf_filename

    print(f"Reading {md_filename}...")
    raw_md = md_path.read_text(encoding="utf-8")

    # Step 1: Pre-process Math expressions
    cleaned_md = clean_math(raw_md)

    # Step 2: Convert callouts
    cleaned_md = convert_callouts(cleaned_md)

    # Step 3: Markdown to HTML conversion
    html_body = markdown.markdown(cleaned_md, extensions=['tables', 'fenced_code', 'toc'])

    # Step 4: Process images for xhtml2pdf with large sizing & landscape switching
    html_body = process_images(html_body, BASE_DIR)

    # Step 5: Full HTML construction with initial portrait template
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{CSS_TEMPLATE}
</style>
</head>
<body>
<pdf:nexttemplate name="portrait" />
{html_body}
</body>
</html>
"""

    print(f"Rendering PDF: {pdf_filename}...")
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print(f"ERROR: Failed to generate {pdf_filename} (code {pisa_status.err})")
    else:
        size_kb = pdf_path.stat().st_size / 1024
        print(f"SUCCESS: {pdf_filename} generated successfully! Size: {size_kb:.1f} KB")

def main():
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else REPORTS
    print(f"Starting PDF export for {len(targets)} DFlash report(s) (Large-Figure Edition)...")
    for report_file in targets:
        convert_md_to_pdf(report_file)
    print("\nPDF export finished!")

if __name__ == "__main__":
    main()
