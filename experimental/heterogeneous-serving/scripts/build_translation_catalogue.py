#!/usr/bin/env python3
"""Build the long-prompt + long-output catalogue for W5 from real TED talk translations.

Each record's prompt is a full English TED talk transcript behind a fixed instruction to translate
it; its decode length is the Gemma-tokenised length of the talk's real published French or German
translation (WIT3 release 2015-05-30).  Each talk is used once, with a seeded choice of target
language.  Runs use ignore_eos with max_tokens set to that length.

Source: IWSLT/ted_talks_iwslt, XML_releases/xml-20150616/ted_{en,fr,de}-20150530.xml.
Usage: build_translation_catalogue.py <xml_dir> <tokenizer.json> <out.json>
"""
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET

from tokenizers import Tokenizer

LANGUAGES = {"fr": "French", "de": "German"}
INSTRUCTION = "Translate the following TED talk transcript from English to {language}. " \
              "Keep every sentence.\n\n"
SUFFIX = "\n\n{language} translation:"
# Both sides at least this long, and the whole context within the 16384-token engines.
MIN_SIDE = 1500
MAX_CONTEXT = 16000


def transcripts(path):
    talks = {}
    for f in ET.parse(path).getroot().iter("file"):
        head = f.find("head")
        segs = [s.text.strip() for s in head.iter("seekvideo") if s.text and s.text.strip()]
        if segs:
            talks[head.findtext("talkid")] = "\n".join(segs)
    return talks


def main(xml_dir, tokenizer_path, out):
    tok = Tokenizer.from_file(tokenizer_path)
    en = transcripts(f"{xml_dir}/ted_en-20150530.xml")
    targets = {lang: transcripts(f"{xml_dir}/ted_{lang}-20150530.xml") for lang in LANGUAGES}
    rng = random.Random(20260930)
    records = []
    for talk_id in sorted(en, key=int):
        choices = [lang for lang in LANGUAGES if talk_id in targets[lang]]
        if not choices:
            continue
        lang = rng.choice(choices)
        language = LANGUAGES[lang]
        prompt = INSTRUCTION.format(language=language) + en[talk_id] + SUFFIX.format(language=language)
        prompt_tokens = len(tok.encode(prompt).ids)
        decode_tokens = len(tok.encode(targets[lang][talk_id]).ids)
        if min(prompt_tokens, decode_tokens) < MIN_SIDE or prompt_tokens + decode_tokens > MAX_CONTEXT:
            continue  # never truncate a talk or its translation
        records.append({"id": f"ted:{talk_id}:{lang}", "source": "ted_talks_iwslt", "prompt": prompt,
                        "prompt_tokens": prompt_tokens, "reference_decode_tokens": decode_tokens})
    blob = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    open(out, "wb").write(blob)
    print(len(records), "records", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
