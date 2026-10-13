#!/usr/bin/env python3
"""Build the mid-length catalogue for W7 from public GovReport reports.

Each record's prompt is a real government report behind a fixed summarisation
instruction (same template as W6); its decode length is the Gemma-tokenised
length of the report's real human-written summary. Kept: 16,384 <= prompt
tokens <= 49,152, prompt + decode <= 51,200 (fits a bf16 TP8 pool at
max_model_len 51,200 and a TP16), 64 <= decode <= 2,048. Reports are deduped
by report hash; nothing is truncated: out-of-band reports are dropped. All
eligible records are counted, then a seeded sample of 240 (random.Random
(20261007).sample, records sorted by id) is written. Runs use ignore_eos with
max_tokens set to the decode length.

Source: ccdv/govreport-summarization (all four cached shards).
Usage: build_midctx_catalogue.py <out.json> <tokenizer.json> <parquet...>

MIN_REPORT_CHARS pre-filter verification (lossless), run over all 19,463
cached reports: the superset of the 13,512 reports with >= 30,000 chars was
tokenised with the Gemma tokenizer (INSTRUCTION + report.strip() + SUFFIX);
among those prompts the largest token count below 40,000 chars is 10,480,
below 45,000 chars is 12,300 (4,364 prompts) and below 50,000 chars is 12,584,
all < 16,384. The minimum chars/token among prompts with >= 16,384 tokens
(2,387 of them) is 3.03 (also the minimum over the whole superset), so a
16,384-token prompt needs >= ~49.6k chars; 45,000 is safely below that.
"""
import hashlib
import json
import pathlib
import random
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

INSTRUCTION = "Summarize the following government report.\n\n"
SUFFIX = "\n\nSummary:"
MIN_PROMPT, MAX_PROMPT = 16384, 49152
MAX_CONTEXT = 51200
SEED, SAMPLE = 20261007, 240
MIN_DECODE, MAX_DECODE = 64, 2048
MIN_REPORT_CHARS = 45000  # verified lossless, see the module docstring


def main(out, tokenizer_path, *parquets):
    tok = Tokenizer.from_file(tokenizer_path)
    seen, eligible = set(), []
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
            if (not MIN_PROMPT <= prompt_tokens <= MAX_PROMPT or prompt_tokens + decode > MAX_CONTEXT
                    or not MIN_DECODE <= decode <= MAX_DECODE):
                continue  # never truncate
            seen.add(digest)
            eligible.append({"id": f"govreport:{digest}", "source": "govreport", "split": split,
                             "prompt": prompt, "prompt_tokens": prompt_tokens,
                             "reference_decode_tokens": decode})
    print(len(eligible), "eligible records across", len(parquets), "shards")
    records = random.Random(SEED).sample(eligible, SAMPLE)
    records.sort(key=lambda r: r["id"])
    blob = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    pathlib.Path(out).write_bytes(blob)
    print(len(records), "records", hashlib.sha256(blob).hexdigest(), len(blob), "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
