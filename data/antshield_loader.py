"""
Parses raw AntShield packet JSON into flat records this project works with.

AntShield's documented schema (per the dataset README and SPAWC'18 paper) is
a dict keyed by "<timestamp>,<uuid>" with fields including post_body, headers,
label (0/1), and pii_types (list of strings). This loader is written against
that schema. If your downloaded copy differs, this is the one file to edit --
everything downstream consumes the flat record shape below and doesn't care
about AntShield's raw nesting.
"""
import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class PacketRecord:
    record_id: str
    text: str            # concatenation of the text-bearing fields we search over
    label: int            # 0/1 binary PII presence, kept for continuity with the BERT experiment
    pii_types: List[str] = field(default_factory=list)
    package_name: str = ""


def _extract_text(packet: dict) -> str:
    """
    Concatenate the fields where PII actually shows up: POST body and header
    values. We deliberately skip header *names* (Content-Type, Host, etc.)
    since those are structural, not PII-bearing.
    """
    parts = []
    post_body = packet.get("post_body")
    if post_body:
        parts.append(str(post_body))

    headers = packet.get("headers", {}) or {}
    for v in headers.values():
        parts.append(str(v))

    return " ".join(parts)


def load_antshield_json(path: str) -> List[PacketRecord]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for record_id, packet in raw.items():
        records.append(
            PacketRecord(
                record_id=record_id,
                text=_extract_text(packet),
                label=int(packet.get("label", 0)),
                pii_types=list(packet.get("pii_types", []) or []),
                package_name=packet.get("package_name", ""),
            )
        )
    return records


if __name__ == "__main__":
    import sys
    recs = load_antshield_json(sys.argv[1])
    print(f"Loaded {len(recs)} records, {sum(r.label for r in recs)} positive")
    for r in recs[:3]:
        print(r)
