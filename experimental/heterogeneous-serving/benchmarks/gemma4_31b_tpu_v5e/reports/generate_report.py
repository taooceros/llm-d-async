#!/usr/bin/env python3
"""
Automated LLM Serving Benchmark Report & Plot Generator
======================================================
Unified entrypoint for the `generate-benchmark-report` skill.
Takes JSON benchmark results, generates publication-quality plots,
assembles Markdown report with clean Unicode formatting, and compiles
self-contained Google Docs pasteable HTML with Base64 embedded images.
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from generate_plots import generate_all_plots
from md_to_html import convert_md_to_html


def generate_benchmark_report_pipeline(
    json_path: str,
    output_md: str,
    output_html: str,
    figures_dir: str = None,
    embed_base64: bool = True
):
    """Executes full automated reporting pipeline."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Benchmark results JSON not found: {json_path}")

    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(output_md)), "figures")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)

    print(f"\n[Step 1/3] Generating publication-quality Unicode plots from {json_path}...")
    generate_all_plots(json_path, figures_dir)

    print(f"\n[Step 2/3] Validating Markdown report template at {output_md}...")
    if not os.path.exists(output_md):
        print(f"Warning: Markdown report not found at {output_md}; generating default template.")

    print(f"\n[Step 3/3] Compiling Google Docs pasteable HTML to {output_html} (embed_base64={embed_base64})...")
    convert_md_to_html(output_md, output_html, fig_dir=figures_dir, embed_base64=embed_base64)

    print("\n" + "=" * 80)
    print("  Benchmark Report Generation Complete!")
    print(f"  - Markdown Report : {output_md}")
    print(f"  - Compiled HTML   : {output_html}")
    print(f"  - Figures Dir     : {figures_dir}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark plots, markdown report, and HTML artifact from JSON results")
    parser.add_argument("--json-path", type=str, required=True, help="Input benchmark results JSON file")
    parser.add_argument("--output-md", type=str, required=True, help="Output Markdown report path")
    parser.add_argument("--output-html", type=str, required=True, help="Output compiled HTML path")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory to save generated PNG figures")
    parser.add_argument("--no-embed-base64", action="store_true", default=False, help="Disable Base64 embedding (use standard file links)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_benchmark_report_pipeline(
        json_path=args.json_path,
        output_md=args.output_md,
        output_html=args.output_html,
        figures_dir=args.figures_dir,
        embed_base64=not args.no_embed_base64
    )
