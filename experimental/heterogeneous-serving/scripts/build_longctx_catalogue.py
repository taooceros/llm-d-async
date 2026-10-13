#!/usr/bin/env python3
"""Build the long-context catalogue for W6 from public GovReport reports.

Each record's prompt is a real government report behind a fixed summarisation
instruction; its decode length is the Gemma-tokenised length of the report's
real human-written summary. Only requests whose prompt exceeds a bf16 TP8 KV
pool (52,864 tokens) and whose full context fits 131,072 tokens are kept, so
they can only be served by a TP16 engine. Runs use ignore_eos with max_tokens
set to that length. Nothing is truncated: out-of-band reports are dropped.

Source: ccdv/govreport-summarization (all four cached shards).
Usage: build_longctx_catalogue.py <out.json> <tokenizer.json> <parquet...>
"""
import hashlib
import json
import pathlib
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

INSTRUCTION = "Summarize the following government report.\n\n"
SUFFIX = "\n\nSummary:"
MIN_PROMPT = 52865          # prompt_tokens > 52,864 (bf16 TP8 pool)
MAX_CONTEXT = 131072
MIN_DECODE, MAX_DECODE = 64, 2048
# Cheap pre-filter before tokenising. Verified lossless on all 19,463 cached
# reports: among the 1,479 prompts >=100k chars (all tokenised), the largest
# token count below 150k chars is 37,749 (< 52,865), and the minimum
# chars/token among prompts with > 52,864 tokens is 3.88 (min over all >=100k
# char prompts is 3.03), i.e. a >52,864-token prompt needs >=160k chars even at
# the extreme 3.03 chars/token, comfortably above this threshold.
MIN_REPORT_CHARS = 150000


def main(out, tokenizer_path, *parquets):
    tok = Tokenizer.from_file(tokenizer_path)
    seen, records, min_cpt = set(), [], None
    for parquet in parquets:
        split = pathlib.Path(parquet).stem
        for row in pq.read_table(parquet).to_pylist():
            report, summary = row["report"], row["summary"]
            if not report or not summary or len(report) < MIN_REPORT_CHARS:
                continue
            digest = hashlib.sha256(report.encode()).hexdigest()[:16]
            if digest in seen:
                continue
            prompt = INSTRUCTION + report.strip() + SUFFIX
            prompt_tokens = len(tok.encode(prompt).ids)
            decode = len(tok.encode(summary).ids)
            if (prompt_tokens < MIN_PROMPT or prompt_tokens + decode > MAX_CONTEXT
                    or not MIN_DECODE <= decode <= MAX_DECODE):
                continue  # never truncate
            seen.add(digest)
            cpt = len(prompt) / prompt_tokens
            min_cpt = cpt if min_cpt is None else min(min_cpt, cpt)
            records.append({"id": f"govreport:{digest}", "source": "govreport", "split": split,
                            "prompt": prompt, "prompt_tokens": prompt_tokens,
                            "reference_decode_tokens": decode})
    records.sort(key=lambda r: r["id"])
    blob = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    pathlib.Path(out).write_bytes(blob)
    print(len(records), "records", hashlib.sha256(blob).hexdigest(), "min chars/token", min_cpt)


if __name__ == "__main__":
    main(*sys.argv[1:])
