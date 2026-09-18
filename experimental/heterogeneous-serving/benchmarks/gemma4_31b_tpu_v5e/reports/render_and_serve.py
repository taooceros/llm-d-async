#!/usr/bin/env python3
"""
Automated Markdown-to-HTML converter & live local preview server.
- Converts input Markdown to Google-styled HTML with embedded base64 figures.
- Starts a local HTTP server on a specified/free port (default: 8080).
- Automatically invokes the omp agent CLI if any failure occurs.
"""

import os
import sys
import time
import base64
import socket
import argparse
import subprocess
import http.server
import socketserver
from pathlib import Path

def call_omp_agent_on_error(context: str, error_msg: str):
    """Invokes omp non-interactively to report and auto-diagnose errors."""
    print(f"\n[!] Triggering omp agent auto-diagnosis for: {context}")
    prompt = (
        f"An error occurred in automated markdown conversion/port-forward script:\n"
        f"Context: {context}\n"
        f"Error details:\n{error_msg}\n\n"
        f"Please investigate and recommend a fix."
    )
    try:
        subprocess.run(
            ["omp", "--message", prompt],
            check=False
        )
    except Exception as e:
        print(f"[!] Failed to invoke omp CLI: {e}", file=sys.stderr)

def find_available_port(start_port: int = 8080) -> int:
    """Finds the first open TCP port starting from start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start_port}-{start_port+100}")

def convert_md_to_html(md_path: str, output_html_path: str, fig_dir: str = None):
    """Converts Markdown to Google-styled self-contained HTML with embedded images."""
    import markdown

    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown source not found: {md_path}")

    if fig_dir is None:
        fig_dir = os.path.join(os.path.dirname(md_path), "figures")

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Embed figures as base64 data URIs
    if os.path.exists(fig_dir):
        for fig_file in os.listdir(fig_dir):
            if fig_file.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".webp")):
                fig_path = os.path.join(fig_dir, fig_file)
                ext = os.path.splitext(fig_file)[1][1:].lower()
                mime = "image/svg+xml" if ext == "svg" else ("image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}")

                with open(fig_path, "rb") as img_f:
                    b64_str = base64.b64encode(img_f.read()).decode("utf-8")
                    data_uri = f"data:{mime};base64,{b64_str}"
                    md_content = md_content.replace(f"figures/{fig_file}", data_uri)
                    md_content = md_content.replace(f"./figures/{fig_file}", data_uri)

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"]
    )

    # Google Docs / Material Design Styling (720px width for Google Docs fidelity)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{Path(md_path).stem}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Mono:wght@400;500&family=Google+Sans+Text:wght@400;500&display=swap');

body {{
    font-family: "Google Sans Text", "Google Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: #3c4043;
    background-color: #ffffff;
    max-width: 720px;
    margin: 24px auto;
    padding: 0 16px;
}}

h1 {{
    font-family: "Google Sans", sans-serif;
    font-size: 24px;
    font-weight: 500;
    color: #1a73e8;
    margin-top: 32px;
    margin-bottom: 8px;
    line-height: 1.25;
}}

h2 {{
    font-family: "Google Sans", sans-serif;
    font-size: 17px;
    font-weight: 400;
    color: #5f6368;
    margin-top: 4px;
    margin-bottom: 20px;
}}

h3 {{
    font-family: "Google Sans", sans-serif;
    font-size: 17px;
    font-weight: 500;
    color: #202124;
    margin-top: 28px;
    margin-bottom: 12px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e8eaed;
}}

h4 {{
    font-family: "Google Sans", sans-serif;
    font-size: 14.5px;
    font-weight: 500;
    color: #202124;
    margin-top: 18px;
    margin-bottom: 8px;
}}

p {{
    margin-top: 0;
    margin-bottom: 12px;
    color: #3c4043;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0 20px 0;
    font-size: 12.5px;
    border: 1px solid #dadce0;
    border-radius: 6px;
    overflow: hidden;
}}

th {{
    background-color: #f8f9fa;
    color: #202124;
    font-family: "Google Sans", sans-serif;
    font-weight: 500;
    text-align: left;
    padding: 7px 9px;
    border: 1px solid #dadce0;
    border-bottom: 2px solid #dadce0;
}}

td {{
    padding: 6px 9px;
    border: 1px solid #dadce0;
    color: #3c4043;
    vertical-align: top;
}}

tr:nth-child(even) {{
    background-color: #fcfdfe;
}}

ul, ol {{
    margin-top: 4px;
    margin-bottom: 14px;
    padding-left: 22px;
}}

li {{
    margin-bottom: 5px;
    color: #3c4043;
}}

img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 16px auto;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    box-shadow: 0 1px 2px 0 rgba(60,64,67,0.15), 0 1px 3px 1px rgba(60,64,67,0.08);
}}

code {{
    font-family: "Google Sans Mono", "Roboto Mono", monospace;
    font-size: 12.5px;
    background-color: #f1f3f4;
    color: #1967d2;
    padding: 2px 5px;
    border-radius: 4px;
}}

pre {{
    background-color: #f8f9fa;
    border: 1px solid #dadce0;
    border-left: 4px solid #1a73e8;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    margin: 14px 0;
}}

pre code {{
    background-color: transparent;
    color: #202124;
    padding: 0;
    font-size: 12.5px;
    line-height: 1.45;
}}

strong, b {{
    color: #202124;
    font-weight: 500;
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

    print(f"[✓] Generated self-contained HTML: {output_html_path}")

def serve_directory(directory: str, filename: str, port: int):
    """Serves the HTML report via a local HTTP server for previewing and copying."""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            # Suppress normal access logs for clean CLI experience
            pass

    server_address = ("127.0.0.1", port)
    with socketserver.TCPServer(server_address, QuietHandler) as httpd:
        url = f"http://127.0.0.1:{port}/{filename}"
        print(f"\n[✓] Live Preview Server running!")
        print(f"[➔] URL: {url}")
        print(f"[i] Open the URL above in Chrome, press Ctrl+A -> Ctrl+C, and paste directly into Google Docs.")
        print(f"[i] Press Ctrl+C to terminate the server when done.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[✓] Server stopped gracefully.")

def main():
    parser = argparse.ArgumentParser(description="Auto convert Markdown to HTML and serve locally with error fallback.")
    parser.add_argument("--input", "-i", default="admission-control-vllm/benchmarks/gemma4_31b_tpu_v5e/reports/benchmark_report.md", help="Input Markdown file path")
    parser.add_argument("--output", "-o", default="admission-control-vllm/benchmarks/gemma4_31b_tpu_v5e/reports/benchmark_report.html", help="Output HTML file path")
    parser.add_argument("--fig-dir", "-f", default=None, help="Figures directory")
    parser.add_argument("--port", "-p", type=int, default=8080, help="HTTP server port (default: 8080 or next available)")
    parser.add_argument("--no-serve", action="store_true", help="Convert only, do not launch preview HTTP server")
    args = parser.parse_args()

    try:
        # Phase 1: Convert Markdown to HTML
        convert_md_to_html(args.input, args.output, args.fig_dir)
    except Exception as e:
        error_details = f"Markdown Conversion Failure: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_details}", file=sys.stderr)
        call_omp_agent_on_error("Markdown Conversion Phase", error_details)
        sys.exit(1)

    if args.no_serve:
        return

    try:
        # Phase 2: Start local preview server with auto-selected port
        active_port = find_available_port(args.port)
        out_dir = os.path.dirname(os.path.abspath(args.output))
        out_file = os.path.basename(args.output)
        serve_directory(out_dir, out_file, active_port)
    except Exception as e:
        error_details = f"HTTP Port-Forward Server Failure: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_details}", file=sys.stderr)
        call_omp_agent_on_error("Preview Server Phase", error_details)
        sys.exit(1)

if __name__ == "__main__":
    main()
