"""
Audit the GLiNER train/val/test JSONL files for leakage and bias, and show raw entries.

Checks (each prints PASS / WARN / FAIL with counts and examples):
  1. RECON      any token containing "RECON" left in the text (the redaction sentinel spells the label)
  2. FIELDS     extra keys on each line (only tokenized_text + ner expected) and field-name tokens
                ("pii_types", "source", "label" as a key, ...) inside the text
  3. TYPE-NAME  entity VALUES that contain their own label name (e.g. a value like "..androidid..")
  4. DUPLICATES exact and digit-normalised duplicate windows inside each file, and shared between files
  5. VALUE OVERLAP  share of test entity values that also appear verbatim as train entity values, per label
  6. CONTEXT    the most common token just before an entity, per label (cue words the model can lean on)
  7. SAMPLES    N raw windows per file with entities marked [[value | label]]

Usage (PowerShell, repo root):
    python diagnostics\\data_audit.py --train data\\train_finetune.jsonl --val data\\val_finetune.jsonl `
        --test data\\test_finetune.jsonl --test-capped data\\test_capped_finetune.jsonl --samples 5 | Tee-Object data_audit.txt
    python diagnostics\\data_audit.py --selftest
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

FIELD_WORDS = {"pii_types", "pii_type", "source", "label", "labels", "is_pii", "has_pii", "ground_truth"}


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def spans(row):
    toks = row["tokenized_text"]
    for s, e, lab in row.get("ner", []):
        yield s, e, lab, " ".join(toks[s:e + 1])


def norm_label(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def digit_norm(tokens):
    return hashlib.md5(re.sub(r"[0-9]", "0", " ".join(tokens)).encode()).hexdigest()


def exact(tokens):
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()


def status(bad, warn=False):
    return ("WARN" if warn else "FAIL") if bad else "PASS"


def show(items, n=3):
    for it in items[:n]:
        print("      e.g.", it if len(str(it)) < 300 else str(it)[:300] + "...")


def audit(files, samples):
    data = {k: load(p) for k, p in files.items()}
    print("=" * 90)
    for k, rows in data.items():
        n_ent = sum(len(r.get("ner", [])) for r in rows)
        n_pos = sum(1 for r in rows if r.get("ner"))
        print(f"{k:<12} {files[k]}  windows={len(rows)}  with entities={n_pos}  entities={n_ent}")
    print("=" * 90)

    # 1. RECON
    print("\n[1] RECON sentinel left in text")
    for k, rows in data.items():
        hits = [" ".join(r["tokenized_text"]) for r in rows if any("recon" in t.lower() for t in r["tokenized_text"])]
        print(f"  {status(hits)}  {k}: {len(hits)} windows contain 'RECON'")
        show([h[:200] for h in hits])

    # 2. FIELDS
    print("\n[2] extra keys / field-name tokens")
    for k, rows in data.items():
        keys = Counter(tuple(sorted(r.keys())) for r in rows)
        extra = {ks: c for ks, c in keys.items() if set(ks) - {"tokenized_text", "ner"}}
        print(f"  {status(extra)}  {k}: key sets {dict(keys)}")
        tokc = Counter()
        for r in rows:
            for t in r["tokenized_text"]:
                if t.lower() in FIELD_WORDS:
                    tokc[t.lower()] += 1
        print(f"        {'WARN' if tokc else 'PASS'}  field-name tokens in text: {dict(tokc) if tokc else 'none'}"
              f"{'   (note: a word like source/label can be a normal word in a URL or header, inspect the examples)' if tokc else ''}")

    # 3. TYPE-NAME IN VALUE
    print("\n[3] entity values containing their own label name")
    for k, rows in data.items():
        bad = []
        for r in rows:
            for s, e, lab, val in spans(r):
                nl, nv = norm_label(lab), norm_label(val)
                if len(nl) >= 4 and nl in nv:
                    bad.append((lab, val))
        print(f"  {status(bad)}  {k}: {len(bad)} entities")
        show(bad)

    # 4. DUPLICATES
    print("\n[4] duplicate windows")
    ex_sets, dn_sets = {}, {}
    for k, rows in data.items():
        ex = Counter(exact(r["tokenized_text"]) for r in rows)
        dn = Counter(digit_norm(r["tokenized_text"]) for r in rows)
        ex_sets[k], dn_sets[k] = ex, dn
        dup_ex = sum(c - 1 for c in ex.values() if c > 1)
        dup_dn = sum(c - 1 for c in dn.values() if c > 1)
        print(f"  INFO  {k}: {len(rows)} windows | exact duplicates {dup_ex} ({dup_ex / max(1, len(rows)):.1%}) "
              f"| digit-normalised duplicates {dup_dn} ({dup_dn / max(1, len(rows)):.1%}) | distinct shapes {len(dn)}")
    names = list(data)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ea = set(ex_sets[a]); eb = set(ex_sets[b]); da = set(dn_sets[a]); db = set(dn_sets[b])
            share_b = sum(c for h, c in ex_sets[b].items() if h in ea)
            share_bn = sum(c for h, c in dn_sets[b].items() if h in da)
            tr_test = {a, b} & {"train"} and {a, b} - {"train"}
            flag = status(share_b > 0, warn=not tr_test)
            print(f"  {flag}  {b} windows also in {a}: exact {share_b}/{len(data[b])} ({share_b / max(1, len(data[b])):.1%}) "
                  f"| digit-normalised {share_bn}/{len(data[b])} ({share_bn / max(1, len(data[b])):.1%})")

    # 5. VALUE OVERLAP
    if "train" in data:
        print("\n[5] share of eval entity VALUES that also occur verbatim among train values (per label)")
        trv = defaultdict(set)
        for r in data["train"]:
            for s, e, lab, val in spans(r):
                trv[lab].add(val)
        for k, rows in data.items():
            if k == "train":
                continue
            tot, seen = Counter(), Counter()
            for r in rows:
                for s, e, lab, val in spans(r):
                    tot[lab] += 1
                    seen[lab] += val in trv[lab]
            allt, alls = sum(tot.values()), sum(seen.values())
            print(f"  INFO  {k}: {alls}/{allt} ({alls / max(1, allt):.1%}) entity values seen in train")
            for lab, c in sorted(tot.items(), key=lambda x: -x[1]):
                if seen[lab] / c > 0.2 and c >= 20:
                    print(f"        WARN  {lab}: {seen[lab]}/{c} ({seen[lab] / c:.0%}) values also in train")

    # 6. CONTEXT
    print("\n[6] most common token right before an entity (train; top 3 per label, labels with >=30 entities)")
    ref = data.get("train") or next(iter(data.values()))
    ctx = defaultdict(Counter)
    cnt = Counter()
    for r in ref:
        for s, e, lab, val in spans(r):
            cnt[lab] += 1
            ctx[lab][r["tokenized_text"][s - 1] if s > 0 else "<start>"] += 1
    for lab, c in sorted(cnt.items(), key=lambda x: -x[1]):
        if c < 30:
            continue
        top = ctx[lab].most_common(3)
        share = top[0][1] / c
        print(f"  {'WARN' if share > 0.5 else 'INFO'}  {lab:<24} n={c:<6} " + "  ".join(f"{t!r}:{v / c:.0%}" for t, v in top))

    # 7. SAMPLES
    print(f"\n[7] raw samples ({samples} per file, first windows with entities)")
    for k, rows in data.items():
        shown = 0
        print(f"  --- {k} ---")
        for r in rows:
            if not r.get("ner"):
                continue
            toks = list(r["tokenized_text"])
            for s, e, lab, val in sorted(r["ner"] and [(s, e, lab, " ".join(toks[s:e + 1])) for s, e, lab in r["ner"]], reverse=True):
                toks[s:e + 1] = [f"[[{val} | {lab}]]"]
            line = " ".join(toks)
            print("    ", line[:700] + ("..." if len(line) > 700 else ""))
            shown += 1
            if shown >= samples:
                break


def selftest():
    tr = [{"tokenized_text": ["uri", ":", "a", "id", "=", "123"], "ner": [[5, 5, "advertiser id"]]},
          {"tokenized_text": ["x", "RECON_Email", "y"], "ner": [[1, 1, "email"]], "source": "train"},
          {"tokenized_text": ["m", "=", "foo"], "ner": []}]
    te = [{"tokenized_text": ["uri", ":", "a", "id", "=", "123"], "ner": [[5, 5, "advertiser id"]]},
          {"tokenized_text": ["q", "pii_types", "z"], "ner": []}]
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = Path(d) / "tr.jsonl", Path(d) / "te.jsonl"
        p1.write_text("\n".join(json.dumps(x) for x in tr))
        p2.write_text("\n".join(json.dumps(x) for x in te))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            audit({"train": str(p1), "test": str(p2)}, 2)
        out = buf.getvalue()
    assert "FAIL  train: 1 windows contain 'RECON'" in out, out
    assert "FAIL  train: key sets" in out, out
    assert "pii_types" in out
    assert "FAIL  test windows also in train: exact 1/2" in out, out
    assert "values seen in train" in out
    print(out)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--test"); ap.add_argument("--test-capped")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    files = {k: v for k, v in (("train", a.train), ("val", a.val), ("test", a.test), ("test-capped", a.test_capped)) if v}
    if not files:
        ap.error("give at least one of --train/--val/--test/--test-capped")
    audit(files, a.samples)


if __name__ == "__main__":
    main()
