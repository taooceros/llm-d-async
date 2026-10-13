"""Frozen workload manifests for the evaluation arms.

Every arm replays byte-identical token sequences, so manifests are built once
on the cluster (the tokenizer lives with the weights) and stored as explicit
``prompt_token_ids``.  This removes tokenizer drift between arms and makes the
useful-token budget exact rather than approximate.

Three predeclared workloads:

* ``W0`` primary mixed backlog.  Reuses the saved B1/W-SMALL manifest's
  ``(prompt_tokens, decode_tokens, arrival_index)`` triples -- so the demanded
  shape and the 721,595 useful-token budget are the historical ones -- but
  binds every row to a *distinct* real conversation truncated to the exact
  prompt length.  The historical driver sampled its 500-prompt catalogue with
  replacement; uniqueness is recorded as a deliberate deviation.
* ``W1`` held-out long-context/growing requests, disjoint conversations.
* ``W2`` low-migration compact control, disjoint conversations.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .workload import load_catalogue

HISTORICAL_MANIFEST = "results/scheduling_headroom_20260915/manifest.json"
HISTORICAL_MANIFEST_SHA256 = (
    "5f30f9fa9ff6eb097618448abd434843b7c04986187561da9419bfd80013936a"
)

# Disjoint index windows into the tokenised large catalogue.  Fixed before any
# timing run so that the three workloads never share a conversation.
CATALOGUE_WINDOWS = {
    "W0": (0, 12000),
    "W1": (12000, 20000),
    "W2": (20000, 27000),
    "E_SHORT": (0, 27000),
    "E_LONG": (0, 27000),
    "W3": (0, 4000),
    "W4": (0, 12000),
    "W5": (0, 1460),
    "W6": (0, 63),
    "W7": (0, 240),  # all 240 midctx catalogue records
}

# Long-decode offline reasoning backlog: every request decodes for the length
# of a real R1 trace for its problem (median ~5.1k, p90 ~12k tokens), so a TP8
# instance (~26k KV tokens) cannot hold many to completion. Intended for
# length-blind scheduling: the gateway must not read max_tokens, so growth is
# discovered at run time and pressure forces preemption or migration.
W3_SPEC = {"requests": 160, "catalogue": "reasoning", "seed": 20260929,
           "max_context_tokens": 16384}

# Mixed long-document + chat bulk backlog.  Short requests reuse a seeded
# sample of the historical W0 (prompt, decode) demands bound to real ShareGPT
# conversations; long requests are real GovReport reports whose full context
# exceeds a TP8 engine's entire KV pool (26368 tokens), so only TP16 engines
# can serve them.  Long requests are spread uniformly through arrival order.
W4_SPEC = {"short_requests": 880, "long_requests": 24, "long_catalogue": "longdoc",
           "seed": 20260930, "min_long_context_tokens": 27000,
           "max_context_tokens": 36864}

# Long-prompt + long-output bulk backlog: translate a whole English TED talk
# transcript into French or German; decode length is the real published
# translation's length (median prompt ~3.5k, decode ~4k tokens).
W5_SPEC = {"requests": 250, "catalogue": "translation", "seed": 20260930,
           "max_context_tokens": 16384}

# Long-context bulk backlog: real GovReport reports whose prompt exceeds a bf16
# TP8 KV pool (52,864 tokens); only a TP16 engine with max_model_len 131072 can
# serve them.  Decode length is the real summary's length; run with ignore_eos.
W6_SPEC = {"requests": 40, "catalogue": "longctx", "seed": 20261006,
           "min_prompt_tokens": 52865, "max_context_tokens": 131072}

# Mid-length band: real GovReport reports with 16,384-49,152-token prompts;
# fits both a bf16 TP8 pool (52.9k; S deployed at max_model_len 51,200) and a
# TP16.  Decode length is the real summary's length; run with ignore_eos.
W7_SPEC = {"requests": 120, "catalogue": "midctx", "seed": 20261007,
           "min_prompt_tokens": 16384, "max_prompt_tokens": 49152,
           "max_context_tokens": 51200}

W1_SPEC = {
    "requests": 320,
    "prompt_tokens": [512, 1536],
    "decode_tokens": [1024, 3072],
    "seed": 20260916,
}
W2_SPEC = {
    "requests": 800,
    "prompt_tokens": [64, 256],
    "decode_tokens": [64, 256],
    "seed": 20260917,
}

# Two real-conversation operating regions, not a tuning sweep.
EFFICIENCY_SPEC = {
    "E_SHORT": {"requests": 256, "prompt_tokens": 512, "decode_tokens": 256},
    "E_LONG": {"requests": 48, "prompt_tokens": 4096, "decode_tokens": 256},
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


_TOKENISED: Dict[tuple, Dict[str, Any]] = {}


def _tokenise_catalogue(model: str, catalogue_name: str) -> Dict[str, Any]:
    """Tokenises a catalogue once per process with the real model tokenizer."""
    cached = _TOKENISED.get((model, catalogue_name))
    if cached is not None:
        return cached
    from transformers import AutoTokenizer

    catalogue = load_catalogue(catalogue_name)
    tokenizer = AutoTokenizer.from_pretrained(model)
    texts = [record["prompt"] for record in catalogue["records"]]
    encoded = tokenizer(texts, add_special_tokens=True)["input_ids"]
    rows = [
        {
            "id": record["id"],
            "index": index,
            "token_ids": ids,
            "num_tokens": len(ids),
        }
        for index, (record, ids) in enumerate(zip(catalogue["records"], encoded))
    ]
    tokenised = {
        "catalogue": catalogue_name,
        "path": catalogue["path"],
        "sha256": catalogue["sha256"],
        "model": model,
        "rows": rows,
    }
    _TOKENISED[(model, catalogue_name)] = tokenised
    return tokenised


def _window(tokenised: Dict[str, Any], workload: str) -> List[Dict[str, Any]]:
    lo, hi = CATALOGUE_WINDOWS[workload]
    rows = tokenised["rows"]
    if hi > len(rows):
        raise ValueError(
            f"catalogue has {len(rows)} conversations; window {workload}={lo}:{hi} "
            "does not fit"
        )
    return rows[lo:hi]


def _assign_unique(
    rows: Sequence[Dict[str, Any]], demands: Sequence[int]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Binds each demand to a distinct conversation and truncated token vector.

    Longest demands are satisfied first from the pool of conversations that are
    long enough, taking the shortest sufficient one so that long conversations
    stay available.  Truncation keeps the real leading text of a real
    conversation; a shortfall is recorded rather than padded with filler.
    """
    pool = sorted(rows, key=lambda r: r["num_tokens"])
    lengths = [r["num_tokens"] for r in pool]
    used = [False] * len(pool)
    used_sources = set()
    used_prompts = set()
    import bisect

    assignments: List[Optional[Dict[str, Any]]] = [None] * len(demands)
    shortfalls = 0
    for position in sorted(range(len(demands)), key=lambda i: -demands[i]):
        want = demands[position]
        index = bisect.bisect_left(lengths, want)
        picked = None
        for probe in range(index, len(pool)):
            if (not used[probe] and pool[probe]["id"] not in used_sources
                    and tuple(pool[probe]["token_ids"][:want]) not in used_prompts):
                picked = probe
                break
        if picked is None:
            for probe in range(len(pool) - 1, -1, -1):
                if (not used[probe] and pool[probe]["id"] not in used_sources
                        and tuple(pool[probe]["token_ids"][:want]) not in used_prompts):
                    picked = probe
                    break
            shortfalls += 1
        if picked is None:
            raise ValueError("catalogue window exhausted before all demands were bound")
        used[picked] = True
        row = pool[picked]
        used_sources.add(row["id"])
        used_prompts.add(tuple(row["token_ids"][:want]))
        assignments[position] = {
            "source_id": row["id"],
            "source_index": row["index"],
            "source_tokens": row["num_tokens"],
            "token_ids": list(row["token_ids"][:want]),
        }
    stats = {
        "shortfalls": shortfalls,
        "unique_conversations": len(used_sources),
        "unique_tokenized_prompts": len(used_prompts),
    }
    return [a for a in assignments], stats


def _historical_manifest_path() -> pathlib.Path:
    import os

    staged_root = os.environ.get("OMP_HETERO_DATA_DIR")
    if staged_root:
        staged = pathlib.Path(staged_root) / HISTORICAL_MANIFEST
        if staged.exists():
            return staged
    return _repo_root() / HISTORICAL_MANIFEST


def build_w0(model: str) -> Dict[str, Any]:
    raw = _historical_manifest_path().read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != HISTORICAL_MANIFEST_SHA256:
        raise ValueError(
            f"historical manifest hash changed: {digest} != {HISTORICAL_MANIFEST_SHA256}"
        )
    triples = json.loads(raw)
    tokenised = _tokenise_catalogue(model, "large")
    rows = _window(tokenised, "W0")
    demands = [int(p) for p, _, _ in triples]
    assignments, stats = _assign_unique(rows, demands)

    requests = []
    for index, ((prompt_tokens, decode_tokens, arrival), bound) in enumerate(
        zip(triples, assignments)
    ):
        requests.append(
            {
                "request_id": f"W0-{index:05d}",
                "arrival_index": int(arrival),
                "prompt_token_ids": bound["token_ids"],
                "prompt_tokens": len(bound["token_ids"]),
                "demanded_prompt_tokens": int(prompt_tokens),
                "max_tokens": int(decode_tokens),
                "source_id": bound["source_id"],
            }
        )
    requests.sort(key=lambda r: r["arrival_index"])
    return _finalise(
        "W0",
        requests,
        tokenised,
        provenance={
            "historical_manifest": HISTORICAL_MANIFEST,
            "historical_manifest_sha256": digest,
            "reuse": "triples (prompt_tokens, decode_tokens, arrival_index)",
            "deviation": (
                "prompts are unique real conversations truncated to the demanded "
                "prompt length; the historical driver sampled a 500-prompt "
                "catalogue with replacement"
            ),
            "assignment": stats,
        },
    )


def _build_synthetic_shape(
    name: str, model: str, spec: Dict[str, Any]
) -> Dict[str, Any]:
    tokenised = _tokenise_catalogue(model, "large")
    rows = _window(tokenised, name)
    rng = random.Random(spec["seed"])
    lo_p, hi_p = spec["prompt_tokens"]
    lo_d, hi_d = spec["decode_tokens"]
    demands = [rng.randint(lo_p, hi_p) for _ in range(spec["requests"])]
    decodes = [rng.randint(lo_d, hi_d) for _ in range(spec["requests"])]
    assignments, stats = _assign_unique(rows, demands)
    requests = [
        {
            "request_id": f"{name}-{index:05d}",
            "arrival_index": index,
            "prompt_token_ids": bound["token_ids"],
            "prompt_tokens": len(bound["token_ids"]),
            "demanded_prompt_tokens": demanded,
            "max_tokens": decode,
            "source_id": bound["source_id"],
        }
        for index, (bound, demanded, decode) in enumerate(
            zip(assignments, demands, decodes)
        )
    ]
    return _finalise(
        name,
        requests,
        tokenised,
        provenance={"spec": spec, "assignment": stats},
    )

def _build_efficiency(name: str, model: str) -> Dict[str, Any]:
    tokenised = _tokenise_catalogue(model, "large")
    spec = EFFICIENCY_SPEC[name]
    excluded = {row["source_id"] for workload in ("W0", "W1", "W2")
                for row in load_manifest(workload)["requests"]}
    rows = [row for row in _window(tokenised, name)
            if row["id"] not in excluded
            and (row["num_tokens"] < 4096 if name == "E_SHORT" else row["num_tokens"] >= 4096)]
    assignments, stats = _assign_unique(rows, [spec["prompt_tokens"]] * spec["requests"])
    if stats["shortfalls"] or len({row["source_id"] for row in assignments}) != spec["requests"]:
        raise ValueError(f"real catalogue cannot satisfy the declared {name} regime")
    requests = [
        {"request_id": f"{name}-{index:05d}", "arrival_index": index,
         "prompt_token_ids": row["token_ids"], "prompt_tokens": len(row["token_ids"]),
         "demanded_prompt_tokens": spec["prompt_tokens"],
         "max_tokens": spec["decode_tokens"], "source_id": row["source_id"]}
        for index, row in enumerate(assignments)
    ]
    return _finalise(name, requests, tokenised, provenance={
        "spec": spec, "assignment": stats,
        "selection": "unique real conversations excluding all W0/W1/W2 source IDs; indices below warmup window; E_SHORT source length below 4096, E_LONG at least 4096",
        "scope": "32-chip H versus S finite concurrent batch, real prefill plus decode; not a sustained deep-backlog claim",
    })


def _build_reasoning(model: str, name: str = "W3") -> Dict[str, Any]:
    spec = {"W3": W3_SPEC, "W5": W5_SPEC}[name]
    tokenised = _tokenise_catalogue(model, spec["catalogue"])
    records = load_catalogue(spec["catalogue"])["records"]
    rows = _window(tokenised, name)
    rng = random.Random(spec["seed"])
    order = list(range(len(rows)))
    rng.shuffle(order)
    requests = []
    for index in order:
        row, record = rows[index], records[rows[index]["index"]]
        decode = int(record["reference_decode_tokens"])
        if row["num_tokens"] + decode > spec["max_context_tokens"]:
            continue  # model tokenizer disagrees with the catalogue; never truncate
        requests.append({
            "request_id": f"{name}-{len(requests):05d}", "arrival_index": len(requests),
            "prompt_token_ids": row["token_ids"], "prompt_tokens": row["num_tokens"],
            "demanded_prompt_tokens": row["num_tokens"], "max_tokens": decode,
            "source_id": row["id"],
        })
        if len(requests) == spec["requests"]:
            break
    if len(requests) != spec["requests"]:
        raise ValueError(f"{spec['catalogue']} catalogue cannot satisfy {name}")
    sources = {
        "W3": ("seeded shuffle of the reasoning catalogue; decode length is the real R1 trace length for the same problem; run with ignore_eos",
               "open-r1/OpenR1-Math-220k default shard 0, open-thoughts/OpenThoughts-114k shard 0 (Apache-2.0)"),
        "W5": ("seeded shuffle of the translation catalogue; decode length is the real published fr/de translation length of the same talk; run with ignore_eos",
               "IWSLT/ted_talks_iwslt XML_releases/xml-20150616 ted_{en,fr,de}-20150530 (CC BY-NC-ND 4.0)"),
    }[name]
    return _finalise(name, requests, tokenised, provenance={
        "spec": spec, "selection": sources[0], "sources": sources[1],
        "intended_policy": "decode_length_blind",
    })


def _build_longdoc_mix(model: str) -> Dict[str, Any]:
    spec = W4_SPEC
    rng = random.Random(spec["seed"])
    raw = _historical_manifest_path().read_bytes()
    if hashlib.sha256(raw).hexdigest() != HISTORICAL_MANIFEST_SHA256:
        raise ValueError("historical manifest hash changed")
    triples = json.loads(raw)
    short_triples = sorted(rng.sample(range(len(triples)), spec["short_requests"]))
    tokenised = _tokenise_catalogue(model, "large")
    assignments, stats = _assign_unique(
        _window(tokenised, "W4"), [int(triples[i][0]) for i in short_triples])
    short = [
        {"prompt_token_ids": bound["token_ids"], "prompt_tokens": len(bound["token_ids"]),
         "demanded_prompt_tokens": int(triples[i][0]), "max_tokens": int(triples[i][1]),
         "source_id": bound["source_id"], "request_class": "short"}
        for i, bound in zip(short_triples, assignments)
    ]

    long_tokenised = _tokenise_catalogue(model, spec["long_catalogue"])
    long_catalogue = load_catalogue(spec["long_catalogue"])
    long = []
    for row in long_tokenised["rows"]:
        decode = int(long_catalogue["records"][row["index"]]["reference_decode_tokens"])
        context = row["num_tokens"] + decode
        if not spec["min_long_context_tokens"] <= context <= spec["max_context_tokens"]:
            continue  # model tokenizer disagrees with the catalogue; never truncate
        long.append({"prompt_token_ids": row["token_ids"], "prompt_tokens": row["num_tokens"],
                     "demanded_prompt_tokens": row["num_tokens"], "max_tokens": decode,
                     "source_id": row["id"], "request_class": "long"})
        if len(long) == spec["long_requests"]:
            break
    if len(long) != spec["long_requests"]:
        raise ValueError("long-document catalogue cannot satisfy W4")

    # Long request k sits at a seeded position inside the k-th equal slice of
    # the arrival order, so long work arrives throughout the backlog.
    total = len(short) + len(long)
    slots = {int((k + rng.random()) * total / len(long)): row for k, row in enumerate(long)}
    shorts = iter(short)
    requests = []
    for index in range(total):
        row = slots.get(index) or next(shorts)
        requests.append({"request_id": f"W4-{index:05d}", "arrival_index": index, **row})
    return _finalise("W4", requests, tokenised, provenance={
        "spec": spec,
        "short": {"historical_manifest": HISTORICAL_MANIFEST,
                  "historical_manifest_sha256": HISTORICAL_MANIFEST_SHA256,
                  "triple_indices_sha256": hashlib.sha256(
                      json.dumps(short_triples).encode()).hexdigest(),
                  "assignment": stats},
        "long": {"catalogue": long_catalogue["path"], "catalogue_sha256": long_catalogue["sha256"],
                 "source": "ccdv/govreport-summarization train shard 0; decode length is the real summary length; run with ignore_eos"},
        "intended_policy": "decode_length_blind",
    })


def _build_longctx(model: str) -> Dict[str, Any]:
    spec = W6_SPEC
    tokenised = _tokenise_catalogue(model, spec["catalogue"])
    catalogue = load_catalogue(spec["catalogue"])
    eligible = []
    for row in tokenised["rows"]:
        decode = int(catalogue["records"][row["index"]]["reference_decode_tokens"])
        if row["num_tokens"] < spec["min_prompt_tokens"] or \
                row["num_tokens"] + decode > spec["max_context_tokens"]:
            continue  # model tokenizer disagrees with the catalogue; never truncate
        eligible.append((row, decode))
    if len(eligible) < spec["requests"]:
        raise ValueError("long-context catalogue cannot satisfy W6")
    rng = random.Random(spec["seed"])
    chosen = rng.sample(eligible, spec["requests"])
    rng.shuffle(chosen)  # arrival order
    requests = [
        {"request_id": f"W6-{index:05d}", "arrival_index": index,
         "prompt_token_ids": row["token_ids"], "prompt_tokens": row["num_tokens"],
         "demanded_prompt_tokens": row["num_tokens"], "max_tokens": decode,
         "source_id": row["id"], "request_class": "long"}
        for index, (row, decode) in enumerate(chosen)
    ]
    shards = sorted({r["split"] for r in catalogue["records"]})
    return _finalise("W6", requests, tokenised, provenance={
        "spec": spec, "catalogue": catalogue["path"], "catalogue_sha256": catalogue["sha256"],
        "source": "ccdv/govreport-summarization shards " + ", ".join(shards)
                  + "; decode length is the real summary length; run with ignore_eos",
        "eligible_after_model_tokenizer": len(eligible),
        "intended_policy": "decode_length_blind",
    })


def _build_midctx(model: str) -> Dict[str, Any]:
    spec = W7_SPEC
    tokenised = _tokenise_catalogue(model, spec["catalogue"])
    catalogue = load_catalogue(spec["catalogue"])
    eligible = []
    for row in tokenised["rows"]:
        decode = int(catalogue["records"][row["index"]]["reference_decode_tokens"])
        if not spec["min_prompt_tokens"] <= row["num_tokens"] <= spec["max_prompt_tokens"] or \
                row["num_tokens"] + decode > spec["max_context_tokens"]:
            continue  # model tokenizer disagrees with the catalogue; never truncate
        eligible.append((row, decode))
    if len(eligible) < spec["requests"]:
        raise ValueError("mid-context catalogue cannot satisfy W7")
    rng = random.Random(spec["seed"])
    chosen = rng.sample(eligible, spec["requests"])
    rng.shuffle(chosen)  # arrival order
    requests = [
        {"request_id": f"W7-{index:05d}", "arrival_index": index,
         "prompt_token_ids": row["token_ids"], "prompt_tokens": row["num_tokens"],
         "demanded_prompt_tokens": row["num_tokens"], "max_tokens": decode,
         "source_id": row["id"], "request_class": "mid"}
        for index, (row, decode) in enumerate(chosen)
    ]
    shards = sorted({r["split"] for r in catalogue["records"]})
    return _finalise("W7", requests, tokenised, provenance={
        "spec": spec, "catalogue": catalogue["path"], "catalogue_sha256": catalogue["sha256"],
        "source": "ccdv/govreport-summarization shards " + ", ".join(shards)
                  + "; decode length is the real summary length; run with ignore_eos",
        "eligible_after_model_tokenizer": len(eligible),
        "intended_policy": "decode_length_blind",
    })


def _finalise(
    name: str,
    requests: List[Dict[str, Any]],
    tokenised: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    useful = sum(r["max_tokens"] for r in requests)
    prompt_total = sum(r["prompt_tokens"] for r in requests)
    body = json.dumps(
        [[r["request_id"], r["prompt_token_ids"], r["max_tokens"]] for r in requests],
        separators=(",", ":"),
    )
    contexts = sorted(r["prompt_tokens"] + r["max_tokens"] for r in requests)
    return {
        "workload": name,
        "model": tokenised["model"],
        "catalogue": tokenised["catalogue"],
        "catalogue_sha256": tokenised["sha256"],
        "catalogue_window": CATALOGUE_WINDOWS[name],
        "requests": requests,
        "num_requests": len(requests),
        "useful_output_tokens": useful,
        "prompt_tokens_total": prompt_total,
        "context_p50": contexts[len(contexts) // 2],
        "context_max": contexts[-1],
        "manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "provenance": provenance,
    }


def build_manifest(name: str, model: str) -> Dict[str, Any]:
    if name == "W0":
        return build_w0(model)
    if name == "W1":
        return _build_synthetic_shape("W1", model, W1_SPEC)
    if name == "W2":
        return _build_synthetic_shape("W2", model, W2_SPEC)
    if name in EFFICIENCY_SPEC:
        return _build_efficiency(name, model)
    if name == "W3":
        return _build_reasoning(model)
    if name == "W4":
        return _build_longdoc_mix(model)
    if name == "W5":
        return _build_reasoning(model, "W5")
    if name == "W6":
        return _build_longctx(model)
    if name == "W7":
        return _build_midctx(model)
    raise KeyError(f"unknown workload {name}")


def manifest_path(name: str, root: Optional[pathlib.Path] = None) -> pathlib.Path:
    root = root or _repo_root()
    return root / "data" / "manifests" / f"{name}.json"


# Multi-phase bulk jobs for the reconfiguration study, built from frozen
# manifests: (part, repetitions) in arrival order.  Repeated parts reuse the
# same prompts under new request ids; prefix caching is off, so every repeat
# costs the same prefill and decode as the first.
COMPOSITES = {
    "W0x4": (("W0", 4),),
    "MIX_W4_W0x4": (("W4", 1), ("W0", 4)),
    "MIX_W4_W5": (("W4", 1), ("W5", 1)),
    "MIX_W6_W0": (("W6", 1), ("W0", 1)),
    "W6x2": (("W6", 2),),
    "W6x4": (("W6", 4),),
}


def _manifest_body(requests: List[Dict[str, Any]]) -> str:
    return json.dumps([[r["request_id"], r["prompt_token_ids"], r["max_tokens"]] for r in requests],
                      separators=(",", ":"))


def _expand_composite(name: str, parts: Sequence[Sequence[Any]],
                      root: Optional[pathlib.Path]) -> Dict[str, Any]:
    requests: List[Dict[str, Any]] = []
    sources = []
    for index, (part, repeats) in enumerate(parts):
        base = load_manifest(part, root)
        sources.append({"part": part, "repeats": repeats, "manifest_sha256": base["manifest_sha256"]})
        for repeat in range(repeats):
            for row in base["requests"]:
                requests.append({**row, "request_id": f"{name}-p{index}r{repeat}-{row['request_id']}",
                                 "arrival_index": len(requests), "phase": index, "phase_workload": part,
                                 "repeat": repeat})
    contexts = sorted(r["prompt_tokens"] + r["max_tokens"] for r in requests)
    first = load_manifest(parts[0][0], root)
    return {"workload": name, "model": first["model"], "catalogue": first["catalogue"],
            "catalogue_sha256": first["catalogue_sha256"], "requests": requests,
            "num_requests": len(requests),
            "useful_output_tokens": sum(r["max_tokens"] for r in requests),
            "prompt_tokens_total": sum(r["prompt_tokens"] for r in requests),
            "context_p50": contexts[len(contexts) // 2], "context_max": contexts[-1],
            "manifest_sha256": hashlib.sha256(_manifest_body(requests).encode()).hexdigest(),
            "composite": [list(p) for p in parts], "provenance": {"parts": sources}}


def write_composite(name: str, root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Freezes a small descriptor; requests are expanded from the parts on load."""
    full = _expand_composite(name, COMPOSITES[name], root)
    descriptor = {k: v for k, v in full.items() if k != "requests"}
    manifest_path(name, root).write_text(json.dumps(descriptor, indent=1))
    return descriptor


def load_manifest(name: str, root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    path = manifest_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"workload manifest {name} is not frozen at {path}; run `build-manifest` first"
        )
    manifest = json.loads(path.read_text())
    if "composite" in manifest:
        expanded = _expand_composite(name, manifest["composite"], root)
        if expanded["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"composite manifest {name} no longer matches its frozen parts")
        return expanded
    digest = hashlib.sha256(_manifest_body(manifest["requests"]).encode()).hexdigest()
    if digest != manifest["manifest_sha256"]:
        raise ValueError(f"manifest {name} failed its own hash check")
    return manifest
