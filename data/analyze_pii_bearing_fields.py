"""
Scans the full AntShield dataset and finds every header key that has ever
carried a RECON_ redaction token. Out of ~400 distinct header keys seen in
this dataset, the overwhelming majority (Content-Type, Accept-Encoding,
Connection, X-Unity-Version, dozens of per-SDK tracking headers, etc.) never
carry PII once -- they're pure token bloat that pushes packets past GLiNER's
384-token limit for no benefit. This script finds the ~30 keys that actually
matter, data-driven rather than guessed, so the whitelist stays correct if
you re-run it against an updated dataset.

Also filters out a handful of malformed "header keys" that are actually
JSON fragments leaked into the headers dict by a handful of corrupted source
packets (keys containing '{', '"', or whitespace) -- these are dataset
artifacts, not real headers, and keeping them adds noise for ~0 benefit
(single-digit occurrence counts).
"""
import argparse
import glob
import json
import re
from collections import Counter

_VALID_HEADER_KEY = re.compile(r"^[A-Za-z0-9._-]{1,60}$")


def analyze(dataset_root: str):
    recon_count = Counter()
    total_count = Counter()

    for fp in glob.glob(f"{dataset_root}/raw_data/**/*.json", recursive=True):
        try:
            packets = json.load(open(fp, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for packet in packets.values():
            headers = packet.get("headers") or {}
            for hk, hv in headers.items():
                total_count[hk] += 1
                if hv and "RECON_" in str(hv):
                    recon_count[hk] += 1

    productive = sorted(
        hk.lower()
        for hk in recon_count
        if recon_count[hk] > 0 and _VALID_HEADER_KEY.match(hk)
    )
    productive = sorted(set(productive))
    return productive, recon_count, total_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", default="configs/productive_header_keys.json")
    args = ap.parse_args()

    productive, recon_count, total_count = analyze(args.dataset_root)

    print(f"{len(productive)} header keys (lowercased, deduped) ever carried a RECON_ token:")
    for hk in productive:
        # report using whichever original-case variant had the highest count
        best_variant = max(
            (k for k in recon_count if k.lower() == hk), key=lambda k: recon_count[k]
        )
        print(f"  {hk:35s} seen {recon_count[best_variant]} times (as '{best_variant}')")

    with open(args.output, "w") as f:
        json.dump(productive, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
