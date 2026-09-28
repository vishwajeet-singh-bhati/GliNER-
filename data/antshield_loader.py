"""
Loader for the actual antshield_public_dataset.zip layout:

    antshield_public_dataset/
      exposures.csv         <- aggregate counts per (pii type, package, dest), not per-packet
      apps_info.csv          <- app metadata, not needed for this pipeline
      raw_data/
        <collection_method>/         e.g. auto_anteater, manual_anteater
          <batch>/                   e.g. batch1..batch6
            <protocol>/               HTTP or HTTPS
              <package_name>.json     dict keyed by "<ts>,<uuid>" -> packet

Each packet dict has (confirmed by direct inspection): post_body, headers
(dict), uri, user_agent, referrer, label (0/1), pii_types (list or null),
package_name, protocol, platform, method, host, dst_ip, dst_port, etc.

Critical finding: this public release is REDACTED. Every leaked PII value is
replaced with a literal sentinel token of the form RECON_<Type>, e.g.
"RECON_AndroidId", "RECON_Email". Confirmed present in uri, post_body,
headers, user_agent, and referrer -- covering 8579/8579 (100%) of positive
records. See convert_to_gliner_format.py for why this matters (label leakage)
and how it's handled.
"""
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_DEFAULT_PRODUCTIVE_HEADERS_PATH = Path(__file__).resolve().parent.parent / "configs" / "productive_header_keys.json"


def _load_productive_headers(path=_DEFAULT_PRODUCTIVE_HEADERS_PATH) -> Optional[set]:
    """
    Loads the data-driven whitelist of header keys that have ever carried a
    RECON_ token (see analyze_pii_bearing_fields.py). Out of ~400 distinct
    header keys in this dataset, ~30 are productive -- the rest are pure
    token bloat (Content-Type, Accept-Encoding, dozens of per-SDK tracking
    headers) that push packets past GLiNER's 384-token limit for no benefit.
    Returns None (include everything, old behavior) if the whitelist hasn't
    been generated yet, with a one-time warning.
    """
    if not os.path.exists(path):
        print(
            f"[warn] {path} not found -- including ALL header keys (verbose, more truncation). "
            f"Run: python data/analyze_pii_bearing_fields.py --dataset-root <path> to generate it."
        )
        return None
    with open(path) as f:
        return set(json.load(f))


@dataclass
class PacketRecord:
    record_id: str          # "<file_path>::<ts,uuid key>"
    text: str                 # concatenation of every field that can carry PII
    label: int
    pii_types: List[str] = field(default_factory=list)
    package_name: str = ""
    protocol: str = ""


def _extract_text(packet: dict, productive_headers: Optional[set]) -> str:
    """
    Concatenate every field confirmed (by direct scan) to carry PII, each
    prefixed with a tag so you can tell where a match came from if you inspect
    the converted output later. Order matters only for readability, not
    correctness -- span offsets are computed after this string is built.

    productive_headers: lowercased set of header keys worth including
    (from analyze_pii_bearing_fields.py). None means include all headers.
    """
    pieces = []
    for field_name in ("uri", "post_body", "user_agent", "referrer"):
        val = packet.get(field_name)
        if val:
            pieces.append(f"{field_name}: {val}")

    headers = packet.get("headers") or {}
    for k, v in headers.items():
        if not v:
            continue
        if productive_headers is not None and k.lower() not in productive_headers:
            continue
        pieces.append(f"header_{k}: {v}")

    return " | ".join(pieces)


def load_antshield_dataset(dataset_root: str) -> List[PacketRecord]:
    """
    dataset_root: path to the extracted antshield_public_dataset/ folder
    (the one directly containing raw_data/, exposures.csv, apps_info.csv).
    """
    pattern = os.path.join(dataset_root, "raw_data", "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No JSON files found under {pattern}. "
            f"Check that dataset_root points at the folder containing raw_data/."
        )

    productive_headers = _load_productive_headers()

    records = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                packets = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[warn] skipping unparseable file {fp}: {e}")
            continue

        for key, packet in packets.items():
            records.append(
                PacketRecord(
                    record_id=f"{fp}::{key}",
                    text=_extract_text(packet, productive_headers),
                    label=int(packet.get("label", 0)),
                    pii_types=list(packet.get("pii_types") or []),
                    package_name=packet.get("package_name", ""),
                    protocol=packet.get("protocol", ""),
                )
            )
    return records


if __name__ == "__main__":
    import sys
    recs = load_antshield_dataset(sys.argv[1])
    n_pos = sum(r.label for r in recs)
    print(f"Loaded {len(recs)} records from {sys.argv[1]}, {n_pos} positive ({n_pos/len(recs):.1%})")
    for r in recs:
        if r.label == 1:
            print(r)
            break
