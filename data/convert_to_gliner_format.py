"""
Converts AntShield packet records into GLiNER's expected format:

    {"text": "...", "entities": [{"start": 12, "end": 24, "label": "email address"}]}

Span offsets are recovered in two ways, tried in order per pii_type:
  1. Exact string search against configs/known_pii_values.json (highest precision,
     requires you to have filled in the real values -- see that file's comment).
  2. Regex heuristics for types with a recognizable shape (email, phone).
  3. Key=value heuristic for opaque device identifiers in form-encoded bodies
     (e.g. "androidid=8f2e1..." -> span is the value after the matching key).

Packets where no span can be recovered for a labeled pii_type are counted and
reported, not silently dropped -- check the printed summary before trusting
downstream numbers, since a high skip rate means the heuristics need tuning
against your actual data.
"""
import argparse
import json
import re
from typing import List, Optional, Tuple

from pii_type_mapping import normalize_type
from antshield_loader import load_antshield_json, PacketRecord

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,14}\d")

# key names commonly seen in form-encoded AntShield post bodies, per opaque type
KEY_HINTS = {
    "android device id": ["androidid", "android_id", "deviceid"],
    "imei number": ["imei"],
    "imsi number": ["imsi"],
    "device serial number": ["serial", "serialnumber"],
    "sim card iccid": ["iccid"],
    "mac address": ["mac", "macaddress"],
    "advertiser id": ["advertiserid", "gaid", "adid"],
}

REGEX_HINTS = {
    "email address": EMAIL_RE,
    "phone number": PHONE_RE,
}


def find_span_by_known_value(text: str, label: str, known_values: dict) -> Optional[Tuple[int, int]]:
    for value in known_values.get(label, []):
        if value:
            idx = text.find(value)
            if idx != -1:
                return idx, idx + len(value)
    return None


def find_span_by_regex(text: str, label: str) -> Optional[Tuple[int, int]]:
    pattern = REGEX_HINTS.get(label)
    if pattern:
        m = pattern.search(text)
        if m:
            return m.start(), m.end()
    return None


def find_span_by_key_hint(text: str, label: str) -> Optional[Tuple[int, int]]:
    hints = KEY_HINTS.get(label)
    if not hints:
        return None
    for key in hints:
        # matches key=value inside a form-encoded string, value ends at & or end of string
        m = re.search(rf"{re.escape(key)}=([^&\s]+)", text, re.IGNORECASE)
        if m:
            return m.start(1), m.end(1)
    return None


def resolve_span(text: str, label: str, known_values: dict) -> Optional[Tuple[int, int]]:
    for finder in (
        lambda: find_span_by_known_value(text, label, known_values),
        lambda: find_span_by_regex(text, label),
        lambda: find_span_by_key_hint(text, label),
    ):
        span = finder()
        if span:
            return span
    return None


def convert(records: List[PacketRecord], known_values: dict):
    examples = []
    skipped = 0
    skipped_by_label = {}

    for r in records:
        if not r.pii_types:
            # negative example -- still useful for eval (should predict no entities)
            examples.append({"text": r.text, "entities": []})
            continue

        entities = []
        for raw_type in r.pii_types:
            try:
                label = normalize_type(raw_type)
            except KeyError as e:
                print(f"[warn] {e}")
                continue

            span = resolve_span(r.text, label, known_values)
            if span is None:
                skipped += 1
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue

            start, end = span
            entities.append({"start": start, "end": end, "label": label})

        examples.append({"text": r.text, "entities": entities})

    print(f"Converted {len(examples)} examples, {skipped} labeled pii_types skipped (no span found)")
    if skipped_by_label:
        print("Skips by label:", skipped_by_label)

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw AntShield JSON file")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--known-values", default="configs/known_pii_values.json")
    args = ap.parse_args()

    records = load_antshield_json(args.input)
    with open(args.known_values) as f:
        known_values = json.load(f)

    examples = convert(records, known_values)

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
