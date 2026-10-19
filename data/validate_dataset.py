#!/usr/bin/env python3
"""
Validates data/{train,test}_finetune.jsonl + manifests written by build_dataset.py.
Structural: schema, spans in range, labels in taxonomy, no overlapping spans.
Semantic/leakage: synthetic-only classes absent from real train windows, no synthetic
data in test, no train/test overlap (exact window or same source flow), manifest alignment.
Exit code 1 if any check fails.   Usage: python data/validate_dataset.py [--data-dir data]
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_type_mapping import all_gliner_labels  # noqa: E402
import synth_config as cfg  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fp(w):
    return hashlib.md5(json.dumps([w["tokenized_text"], w["ner"]], separators=(",", ":")).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-window", type=int, default=384, help="GLiNER token limit")
    a = ap.parse_args()
    d = Path(a.data_dir)
    labels = set(all_gliner_labels())
    fails = []

    def check(ok, msg):
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    data, man = {}, {}
    for s in ("train", "test"):
        data[s], man[s] = read_jsonl(d / f"{s}_finetune.jsonl"), read_jsonl(d / f"{s}_manifest.jsonl")
        check(len(data[s]) == len(man[s]), f"{s}: {len(data[s])} windows, manifest aligned")
        bad, real_overlap = Counter(), 0
        for w, m in zip(data[s], man[s]):
            if set(w) != {"tokenized_text", "ner"}:
                bad["schema"] += 1
                continue
            toks = w["tokenized_text"]
            if not toks or not all(isinstance(t, str) and t for t in toks) or len(toks) > a.max_window:
                bad["tokens"] += 1
            spans = sorted(w["ner"])
            for k, n in enumerate(spans):
                if not (len(n) == 3 and isinstance(n[0], int) and isinstance(n[1], int)):
                    bad["span_format"] += 1
                elif not (0 <= n[0] <= n[1] < len(toks)):
                    bad["span_range"] += 1
                if n[2] not in labels:
                    bad["unknown_label"] += 1
                if k and n[0] <= spans[k - 1][1]:
                    if m["source"] == "real" and s == "test":
                        real_overlap += 1  # adjacent-marker artifact of the original masking; reported, not failed
                    else:
                        bad["overlap_or_duplicate"] += 1
        check(not bad, f"{s}: structure ok {dict(bad) if bad else ''}")
        if real_overlap:
            print(f"WARN  {s}: {real_overlap} real windows have overlapping spans (adjacent RECON_ markers merged into one token)")

    # --- semantic / leakage ---
    syn_in_test = sum(m["source"] != "real" for m in man["test"])
    check(syn_in_test == 0, f"no synthetic or decoy windows in test ({syn_in_test} found)")
    leaked = Counter()
    for m in man["train"]:
        if m["source"] == "real":
            for lab in m["labels"]:
                if lab in cfg.SYNTHETIC_ONLY:
                    leaked[lab] += 1
    check(not leaked, f"no REAL train window contains a synthetic-only class {dict(leaked) if leaked else ''}")
    tf = {fp(w) for w in data["test"]}
    ov = sum(fp(w) in tf for w in data["train"])
    check(ov == 0, f"no exact (tokens+ner) overlap train/test ({ov} found)")
    tr_ids = {m["record_id"] for m in man["train"] if m["source"] == "real"}
    te_ids = {m["record_id"] for m in man["test"]}
    check(not (tr_ids & te_ids), f"no source flow in both splits ({len(tr_ids & te_ids)} found)")
    pos = [w for w in data["train"] if w["ner"]]
    check(len(pos) > 0, "train has entity windows")
    for s in ("train", "test"):
        dup = len(data[s]) - len({fp(w) for w in data[s]})
        print(f"INFO  {s}: {dup} exact duplicate windows inside the split")

    # --- coverage ---
    print(f"\n{'label':<24}{'train win':>10}{'train spans':>12}{'test win':>10}{'test spans':>11}")
    for lab in all_gliner_labels():
        row = []
        for s in ("train", "test"):
            row += [sum(any(n[2] == lab for n in w["ner"]) for w in data[s]), sum(n[2] == lab for w in data[s] for n in w["ner"])]
        print(f"{lab:<24}{row[0]:>10}{row[1]:>12}{row[2]:>10}{row[3]:>11}")
    print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
