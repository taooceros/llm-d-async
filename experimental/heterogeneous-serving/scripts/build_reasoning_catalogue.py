#!/usr/bin/env python3
"""Build the long-decode reasoning catalogue for W3 from public R1 traces.

Each record keeps the real problem text as the prompt, and the Gemma-tokenised
length of a real DeepSeek-R1 reasoning trace for that problem as its decode
length. Runs use ignore_eos with max_tokens set to that length, so every
request decodes for as long as a real reasoning answer did. The gateway must
schedule without reading max_tokens (length-blind policy).

Sources (Apache-2.0): open-r1/OpenR1-Math-220k default shard 0 and
open-thoughts/OpenThoughts-114k shard 0.
Usage: build_reasoning_catalogue.py <openr1.parquet> <openthoughts.parquet> <tokenizer.json> <out.json>
"""
import hashlib
import json
import random
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

MAX_MODEL_LEN = 16384  # prompt + decode must fit the engines' max_model_len
PER_SOURCE = 2000


def main(openr1, openthoughts, tokenizer_path, out):
    tok = Tokenizer.from_file(tokenizer_path)
    rng = random.Random(20260929)
    records = []
    for source, path in (("openr1-math", openr1), ("openthoughts", openthoughts)):
        rows = pq.read_table(path).to_pylist()
        rng.shuffle(rows)
        kept = 0
        for index, row in enumerate(rows):
            if source == "openr1-math":
                prompt = row["problem"]
                traces = [g for g, done in zip(row["generations"] or [], row["is_reasoning_complete"] or [])
                          if done]
                trace, uid = (traces[0] if traces else None), row["uuid"]
            else:
                turns = row["conversations"]
                prompt, trace, uid = turns[0]["value"], turns[1]["value"], f"ot-{index}"
            if not prompt or not prompt.strip() or not trace:
                continue
            prompt_tokens = len(tok.encode(prompt).ids)
            decode_tokens = len(tok.encode(trace).ids)
            if prompt_tokens + decode_tokens > MAX_MODEL_LEN or decode_tokens < 256:
                continue  # never truncate a trace; drop those that cannot fit
            records.append({"id": f"{source}:{uid}", "source": source, "prompt": prompt,
                            "prompt_tokens": prompt_tokens, "reference_decode_tokens": decode_tokens})
            kept += 1
            if kept == PER_SOURCE:
                break
    rng.shuffle(records)
    blob = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    open(out, "wb").write(blob)
    print(len(records), "records", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
