#!/usr/bin/env python3
"""Build the long-document catalogue for W4 from public GovReport reports.

Each record's prompt is a real government report behind a fixed summarisation
instruction; its decode length is the Gemma-tokenised length of the report's
real human-written summary. Only reports whose full context (prompt + summary)
exceeds one TP8 engine's whole KV pool are kept, so such a request can only be
served by a TP16 engine. Runs use ignore_eos with max_tokens set to that length.

Source: ccdv/govreport-summarization (train shard 0).
Usage: build_longdoc_catalogue.py <govreport.parquet> <tokenizer.json> <out.json>
"""
import hashlib
import json
import random
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

INSTRUCTION = "Summarize the following government report.\n\n"
SUFFIX = "\n\nSummary:"
# A TP8 engine holds 412 blocks x 64 tokens = 26368 KV tokens; the TP16
# engines run max_model_len 36864.
MIN_CONTEXT = 27000
MAX_CONTEXT = 36864
MAX_DECODE = 2048
LIMIT = 400


def main(parquet, tokenizer_path, out):
    tok = Tokenizer.from_file(tokenizer_path)
    rows = pq.read_table(parquet).to_pylist()
    random.Random(20260930).shuffle(rows)
    records = []
    for index, row in enumerate(rows):
        report, summary = row["report"], row["summary"]
        if not report or not summary:
            continue
        prompt = INSTRUCTION + report.strip() + SUFFIX
        # Cheap character pre-filter before tokenising 30k-token documents.
        if not 60000 <= len(prompt) <= 200000:
            continue
        prompt_tokens = len(tok.encode(prompt).ids)
        decode_tokens = len(tok.encode(summary).ids)
        context = prompt_tokens + decode_tokens
        if not (MIN_CONTEXT <= context <= MAX_CONTEXT) or not 64 <= decode_tokens <= MAX_DECODE:
            continue  # never truncate a report or its summary
        digest = hashlib.sha256(report.encode()).hexdigest()[:16]
        records.append({"id": f"govreport:{digest}", "source": "govreport", "prompt": prompt,
                        "prompt_tokens": prompt_tokens, "reference_decode_tokens": decode_tokens})
        if len(records) == LIMIT:
            break
    blob = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    open(out, "wb").write(blob)
    print(len(records), "records", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
