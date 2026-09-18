"""
Benchmark Reporting Utilities.
Generates structured Markdown and self-contained HTML reports with Google Docs copyable Unicode.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

from .metrics import ServingBenchmarkMetrics


def render_markdown_report(
    title: str,
    model_name: str,
    topology_name: str,
    metrics: ServingBenchmarkMetrics,
    breakdown_table: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str | pathlib.Path] = None,
) -> str:
    """Generates a clean Markdown report with native tables and Unicode formatting."""
    lines = [
        f"# {title}",
        f"**Model**: `{model_name}` | **Topology**: `{topology_name}` | **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Metric | Value | Unit | Description |",
        "|---|---|---|---|",
        f"| **Serving Throughput** | **{metrics.throughput_tokens_per_sec:,.1f}** | tok/s | Aggregate generation rate across batch |",
        f"| **Prefill TTFT (Mean)** | **{metrics.ttft_mean_ms:.2f}** | ms | Time To First Token |",
        f"| **Prefill TTFT (p90)** | **{metrics.ttft_p90_ms:.2f}** | ms | 90th percentile initial prompt latency |",
        f"| **Decode TPOT (Mean)** | **{metrics.tpot_mean_ms:.2f}** | ms | Time Per Output Token |",
        f"| **Decode TPOT (p90)** | **{metrics.tpot_p90_ms:.2f}** | ms | 90th percentile decode step latency |",
        f"| **Compute Throughput** | **{metrics.tflops_achieved:,.1f}** | TFLOPS | Effective forward compute rate |",
        f"| **Model FLOPs MFU** | **{metrics.mfu_percent:.1f}%** | % | Hardware utilization efficiency |",
        f"| **Memory Bandwidth** | **{metrics.memory_bandwidth_gb_per_sec:,.1f}** | GB/s | Effective HBM read bandwidth |",
        "",
    ]

    if breakdown_table:
        lines.extend([
            "## 2. Per-Configuration Breakdown",
            "",
            "| Batch Size | Prompt Len | Output Len | Throughput (tok/s) | TTFT (ms) | TPOT (ms) |",
            "|---|---|---|---|---|---|",
        ])
        for row in breakdown_table:
            lines.append(
                f"| {row.get('batch_size', '-')} | {row.get('prompt_len', '-')} | "
                f"{row.get('output_len', '-')} | {row.get('throughput', 0.0):,.1f} | "
                f"{row.get('ttft_ms', 0.0):.2f} | {row.get('tpot_ms', 0.0):.2f} |"
            )
        lines.append("")

    content = "\n".join(lines)
    if output_path:
        out_p = pathlib.Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(content)

    return content


def render_html_report(
    markdown_content: str,
    output_path: Optional[str | pathlib.Path] = None,
) -> str:
    """Wraps Markdown into a styled, self-contained HTML page for browser inspection."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Serving Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #202124; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #dadce0; padding: 10px 14px; text-align: left; }}
  th {{ background-color: #f8f9fa; font-weight: 600; }}
  tr:nth-child(even) {{ background-color: #f8f9fa; }}
  h1 {{ color: #1a73e8; border-bottom: 2px solid #e8eaed; padding-bottom: 10px; }}
  h2 {{ color: #202124; margin-top: 30px; }}
  code {{ background: #f1f3f4; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
</style>
</head>
<body>
<pre style="white-space: pre-wrap; font-family: inherit;">{markdown_content}</pre>
</body>
</html>"""

    if output_path:
        out_p = pathlib.Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html)

    return html
