#!/usr/bin/env python3
"""
Convert Markdown benchmark report to a Google-styled HTML document.
Uses clean Unicode text (no LaTeX math), embedded Base64 image data URIs
(default for seamless Google Docs copy-pasting), unwrapped top-level <img> tags
(no <p> wrapping), and explicit font-weight: 700 so Google Docs preserves bold formatting.
"""

import os
import re
import base64
import argparse
import markdown


def sanitize_latex_to_unicode(text: str) -> str:
    """
    Sanitizes any LaTeX math syntax into clean Unicode for Google Docs copy-pasting.
    Replaces common LaTeX formulas, Greek letters, symbols, and formatting.
    """
    # 1. Replace fractions \frac{a}{b} -> a / b
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 / \2', text)

    # 2. Replace text formatting \text{...}, \mathbf{...}, \mathit{...}
    text = re.sub(r'\\(?:text|mathbf|mathit|mathrm)\{([^}]+)\}', r'\1', text)

    # 3. Replace common math symbols & Greek letters
    replacements = [
        (r'\\times', '×'),
        (r'\\cdot', '·'),
        (r'\\approx', '≈'),
        (r'\\leq', '≤'),
        (r'\\le\b', '≤'),
        (r'\\geq', '≥'),
        (r'\\ge\b', '≥'),
        (r'\\neq', '≠'),
        (r'\\pm', '±'),
        (r'\\to', '→'),
        (r'\\rightarrow', '→'),
        (r'\\leftarrow', '←'),
        (r'\\in', 'in'),
        (r'\\subset', '⊂'),
        (r'\\sigma', 'σ'),
        (r'\\mu', 'μ'),
        (r'\\alpha', 'α'),
        (r'\\beta', 'β'),
        (r'\\gamma', 'γ'),
        (r'\\theta', 'θ'),
        (r'\\lambda', 'λ'),
        (r'\\Delta', 'Δ'),
        (r'\\Omega', 'Ω'),
        (r'\\%', '%'),
        (r'\\_', '_'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # 4. Remove display math $$...$$ and inline math $...$
    text = re.sub(r'\$\$([^\$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)

    return text


def convert_md_to_html(md_path: str, output_html_path: str, fig_dir: str = None, embed_base64: bool = True):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    if fig_dir is None:
        fig_dir = os.path.join(os.path.dirname(md_path), "figures")

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Sanitize any accidental LaTeX math to clean Unicode
    md_content = sanitize_latex_to_unicode(md_content)

    # Embed local images as Base64 data URIs by default for Google Docs clipboard copyability
    if embed_base64 and os.path.exists(fig_dir):
        for fig_file in os.listdir(fig_dir):
            if fig_file.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.webp')):
                fig_path = os.path.join(fig_dir, fig_file)
                ext = os.path.splitext(fig_file)[1][1:].lower()
                mime = 'image/svg+xml' if ext == 'svg' else ('image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}')

                with open(fig_path, "rb") as img_f:
                    b64_str = base64.b64encode(img_f.read()).decode("utf-8")
                    data_uri = f"data:{mime};base64,{b64_str}"
                    md_content = md_content.replace(f"figures/{fig_file}", data_uri)
                    md_content = md_content.replace(f"./figures/{fig_file}", data_uri)

    # Convert Markdown to HTML using standard extensions
    html_body = markdown.markdown(
        md_content,
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'sane_lists'
        ]
    )

    # Wrap every <img> in a dedicated zero-indent paragraph container with explicit inline styles.
    # This completely prevents Google Docs from annexing bare inline <img> tags into adjacent <ul>/<ol> list indents.
    def repl_img_block(m):
        img_tag = m.group(1) if m.lastindex else m.group(0)
        if 'style=' not in img_tag:
            img_tag = img_tag.replace('<img ', '<img style="display: block; max-width: 100%; height: auto; margin: 0; text-align: left; border: 1px solid #e8eaed; border-radius: 8px;" ')
        return f'<p style="margin: 16px 0; padding: 0; text-indent: 0; margin-left: 0; margin-right: 0; text-align: left; clear: both;">{img_tag}</p>'

    html_body = re.sub(r'<p>\s*(<img[^>]+>)\s*</p>', repl_img_block, html_body)
    html_body = re.sub(r'(?<!clear: both;">)(<img[^>]+>)', repl_img_block, html_body)
    html_body = re.sub(r'<hr\s*/?>', '<hr style="border: none; border-top: 1px solid #dadce0; margin: 24px 0; height: 1px; clear: both;" />', html_body)
    # Explicitly enforce inline styling on tables, th, td, strong for Google Docs clipboard copyability
    html_body = re.sub(r'<strong>', '<strong style="font-weight: 700;">', html_body)
    html_body = re.sub(r'<b>', '<b style="font-weight: 700;">', html_body)
    html_body = re.sub(r'<table>', '<table style="border-collapse: collapse; width: 100%; max-width: 100%; font-size: 11px; margin: 10px 0 16px 0; border: 1px solid #dadce0;">', html_body)
    html_body = re.sub(r'<th>', '<th style="background-color: #f8f9fa; color: #202124; font-family: -apple-system, Roboto, sans-serif; font-weight: 700; font-size: 11px; text-align: left; padding: 5px 6px; border: 1px solid #dadce0; border-bottom: 2px solid #dadce0; line-height: 1.25;">', html_body)
    html_body = re.sub(r'<td>', '<td style="padding: 4px 6px; font-size: 11px; border: 1px solid #dadce0; color: #3c4043; vertical-align: top; line-height: 1.35;">', html_body)
    # Google Material / Docs Design Spec
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Serving Benchmark & Profiling Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Mono:wght@400;500&family=Google+Sans+Text:wght@400;500;700&display=swap');

body {{
    font-family: "Google Sans Text", "Google Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.55;
    color: #3c4043;
    background-color: #ffffff;
    max-width: 680px; /* Standard portrait printable width */
    margin: 20px auto;
    padding: 0 12px;
}}

/* Headings */
h1 {{
    font-family: "Google Sans", sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #1a73e8; /* Google Blue */
    margin-top: 36px;
    margin-bottom: 8px;
    line-height: 1.25;
    letter-spacing: -0.2px;
}}

h2 {{
    font-family: "Google Sans", sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #5f6368;
    margin-top: 4px;
    margin-bottom: 24px;
    line-height: 1.4;
}}

h3 {{
    font-family: "Google Sans", sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #202124;
    margin-top: 32px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e8eaed;
}}

h4 {{
    font-family: "Google Sans", sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #202124;
    margin-top: 20px;
    margin-bottom: 8px;
}}

/* Clean Section Dividers */
hr {{
    border: none;
    border-top: 1px solid #dadce0;
    margin: 24px 0;
    clear: both;
    height: 1px;
}}

p {{
    margin-top: 0;
    margin-bottom: 12px;
    color: #3c4043;
}}

/* Tables - Google Docs Material Style Optimized for Thin / Portrait Pages */
table {{
    border-collapse: collapse;
    width: 100%;
    max-width: 100%;
    margin: 12px 0 18px 0;
    font-size: 11px;
    border: 1px solid #dadce0;
    border-radius: 4px;
    word-break: normal;
    table-layout: auto;
}}

th {{
    background-color: #f8f9fa;
    color: #202124;
    font-family: "Google Sans", sans-serif;
    font-weight: 700;
    font-size: 11px;
    text-align: left;
    padding: 5px 6px;
    border: 1px solid #dadce0;
    border-bottom: 2px solid #dadce0;
    line-height: 1.25;
}}

td {{
    padding: 4px 6px;
    font-size: 11px;
    border: 1px solid #dadce0;
    color: #3c4043;
    vertical-align: top;
    line-height: 1.35;
}}
tr:nth-child(even) {{
    background-color: #fcfdfe;
}}

tr:hover {{
    background-color: #f1f3f4;
}}

/* Lists */
ul, ol {{
    margin-top: 4px;
    margin-bottom: 14px;
    padding-left: 24px;
}}

li {{
    margin-bottom: 6px;
    color: #3c4043;
}}

/* Images / Charts */
img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 0;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    box-shadow: 0 1px 2px 0 rgba(60,64,67,0.15), 0 1px 3px 1px rgba(60,64,67,0.08);
}}

/* Inline Code & Code Blocks */
code {{
    font-family: "Google Sans Mono", "Roboto Mono", monospace;
    font-size: 13px;
    background-color: #f1f3f4;
    color: #1967d2; /* Google Dark Blue */
    padding: 2px 6px;
    border-radius: 4px;
}}

pre {{
    background-color: #f8f9fa;
    border: 1px solid #dadce0;
    border-left: 4px solid #1a73e8;
    border-radius: 6px;
    padding: 14px;
    overflow-x: auto;
    margin: 16px 0;
}}

pre code {{
    background-color: transparent;
    color: #202124;
    padding: 0;
    font-size: 13px;
    line-height: 1.45;
}}

strong, b {{
    color: #202124;
    font-weight: 700;
}}

/* Print & PDF Pagination Rules */
@media print {{
    body {{
        max-width: 100%;
        margin: 0;
        padding: 0;
    }}
    h2, h3 {{
        break-before: page;
        page-break-before: always;
    }}
    table, img {{
        break-inside: avoid;
        page-break-inside: avoid;
    }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Successfully generated Google-styled HTML: {output_html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown report to Google-styled HTML with Base64 embedded images.")
    parser.add_argument("--input", "-i", required=True, help="Input Markdown file path")
    parser.add_argument("--output", "-o", required=True, help="Output HTML file path")
    parser.add_argument("--fig-dir", "-f", default=None, help="Figures directory path")
    parser.add_argument("--no-embed-base64", action="store_true", default=False, help="Disable Base64 embedding (use standard file links)")
    args = parser.parse_args()

    convert_md_to_html(args.input, args.output, args.fig_dir, embed_base64=not args.no_embed_base64)
