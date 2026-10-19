"""
audit_lengths.py - READ-ONLY audit of window lengths in subwords.
Changes nothing; writes diagnostics/length_audit.json.

Usage (from repo root):
  python tools/audit_lengths.py --data-dir data --tokenizer D:/models/antshield-gliner-bf16-1epoch --budget 384
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from length_utils import fit_dataset, per_word_subwords, print_fit_report  # noqa: E402


def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q / 100 * len(vals)))] if vals else 0


def audit_split(name, data, tok, thresholds):
    sub, words, maxchars, rows = [], [], [], []
    for i, ex in enumerate(data):
        toks = ex.get("tokenized_text", [])
        sw = sum(per_word_subwords(tok, toks))
        sub.append(sw)
        words.append(len(toks))
        mc = max((len(t) for t in toks), default=0)
        maxchars.append(mc)
        rows.append({"idx": i, "words": len(toks), "subwords": sw, "max_token_chars": mc,
                     "labels": sorted({str(s[2]) for s in ex.get("ner") or [] if len(s) >= 3})})
    print(f"\n=== {name}: {len(data)} examples ===")
    print(f"words     p50={pct(words,50)} p90={pct(words,90)} p99={pct(words,99)} max={max(words, default=0)}")
    print(f"subwords  p50={pct(sub,50)} p90={pct(sub,90)} p99={pct(sub,99)} max={max(sub, default=0)} "
          f"mean={statistics.mean(sub) if sub else 0:.0f}")
    print(f"longest single token: {max(maxchars, default=0)} chars")
    for t in thresholds:
        over = [r for r in rows if r["subwords"] > t]
        lab = Counter(l for r in over for l in r["labels"])
        print(f"  > {t:5d} subwords: {len(over):6d} examples ({100*len(over)/max(1,len(rows)):.1f}%)"
              + (f" | labels affected: {dict(lab.most_common(6))}" if over else ""))
    worst = sorted(rows, key=lambda r: r["subwords"], reverse=True)[:10]
    print("  10 longest:", [(r["idx"], r["subwords"]) for r in worst])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer", required=True, help="model dir (or name) holding the tokenizer files")
    ap.add_argument("--budget", type=int, default=384, help="text-subword budget to dry-run")
    ap.add_argument("--out", default="diagnostics/length_audit.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    train = load_jsonl(Path(args.data_dir) / "train_finetune.jsonl")
    test = load_jsonl(Path(args.data_dir) / "test_finetune.jsonl")

    labels = sorted({str(s[2]) for ex in train for s in ex.get("ner") or [] if len(s) >= 3})
    prompt = 1 + sum(1 + len(tok.tokenize(l)) for l in labels)   # <<ENT>>+label per type, <<SEP>>
    print(f"{len(labels)} labels; estimated prompt cost ~{prompt} subwords (added on top of text)")

    th = [256, 384, 512, 768, 1024, 2048]
    r_train = audit_split("train (raw)", train, tok, th)
    r_test = audit_split("test (raw)", test, tok, th)

    print(f"\n--- dry run: what --max-subwords {args.budget} would do ---")
    _, st = fit_dataset(train, tok, args.budget)
    print_fit_report("train", st)
    _, sv = fit_dataset(test, tok, args.budget)
    print_fit_report("test", sv)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"prompt_subwords": prompt, "train": r_train, "test": r_test}, f)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()