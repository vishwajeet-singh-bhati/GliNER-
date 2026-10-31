"""
Per-type confusion table (TP / FP / FN / TN) and boundary-error analysis (merged / split / shifted spans).

Runs ONE model over ONE JSONL file with the repo's own evaluator (same chunking, same exact-match scoring),
then reports, at a fixed threshold:

  [A] SPAN level, per PII type:  TP FP FN  (+ precision / recall / F1).  A TP is an exact (start, end, label)
      match.  TN is NOT defined at span level (there is no countable set of "spans that correctly were not
      predicted"), so the table also gives
  [B] WINDOW level, per PII type: TP FP FN TN, asking "does this window contain at least one <type> span?"
      gold-positive and predicted-positive per window.  TN = windows with no gold and no prediction of that type.
  [C] BOUNDARY errors, per type: every FP / FN that overlaps a gold / predicted span of the SAME label is
      classified as
        merged   one predicted span covers >= 2 gold spans of that label (e.g. "lat , lon" read as one value)
        split    one gold span is covered by >= 2 predicted spans of that label
        shifted  partial overlap, same label, neither contains the other
        wider    prediction strictly contains one gold span (extra tokens)
        narrower prediction strictly inside one gold span (missing tokens)
      and prints examples (text with [[gold]] and <<pred>> marked).

Usage (PowerShell, repo root). E8-QAT checkpoints and their PTQ variants MUST be run with --quantized
(they hold latent float weights that the quantized loader snaps to the E8 lattice; the ordinary loader gives ~0.29 F1):
    python diagnostics\\error_matrix.py --model D:\\...\\checkpoint --data-path data\\test_capped_finetune.jsonl `
        --labels-file <train_labels.json> --threshold 0.5 --out-dir eval_results\\matrix_tag
    python diagnostics\\error_matrix.py --selftest
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- pure functions (tested by --selftest)
def _ov(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def _contains(a, b):  # a contains b
    return a[0] <= b[0] and b[1] <= a[1]


def span_and_window_counts(store, labels, threshold):
    span = {l: Counter() for l in labels}
    win = {l: Counter() for l in labels}
    for rec in store:
        gold, scored = set(rec[0]), rec[1]
        pred = {(s, e, l) for s, e, l, sc in scored if sc >= threshold}
        for l in labels:
            g = {x for x in gold if x[2] == l}
            p = {x for x in pred if x[2] == l}
            span[l]["tp"] += len(g & p)
            span[l]["fp"] += len(p - g)
            span[l]["fn"] += len(g - p)
            gp, pp = bool(g), bool(p)
            win[l]["tp" if gp and pp else "fp" if pp else "fn" if gp else "tn"] += 1
    return span, win


def boundary_errors(store, labels, threshold, max_examples=6):
    """Classify same-label boundary errors; returns (counts[label][kind], examples[label][kind])."""
    counts = {l: Counter() for l in labels}
    ex = {l: defaultdict(list) for l in labels}
    for wi, rec in enumerate(store):
        gold, scored = set(rec[0]), rec[1]
        toks = rec[3] if len(rec) > 3 else None
        pred = {(s, e, l) for s, e, l, sc in scored if sc >= threshold}
        for l in labels:
            g = sorted(x for x in gold if x[2] == l)
            p = sorted(x for x in pred if x[2] == l)
            gset, pset = set(g), set(p)
            for q in p:
                if q in gset:
                    continue
                inside = [x for x in g if _contains(q, x)]
                if len(inside) >= 2:
                    kind = "merged"
                elif any(x for x in g if _ov(q, x)):
                    o = [x for x in g if _ov(q, x)][0]
                    kind = ("wider" if _contains(q, o) else "narrower" if _contains(o, q) else "shifted")
                    # a gold span covered by several preds is reported as split below, not here
                    if len([y for y in p if y not in gset and _contains(o, y)]) >= 2:
                        continue
                else:
                    continue
                counts[l][kind] += 1
                if len(ex[l][kind]) < max_examples:
                    ex[l][kind].append((wi, q, [x for x in g if _ov(q, x)], toks))
            for x in g:
                if x in pset:
                    continue
                inner = [y for y in p if y not in gset and _contains(x, y)]
                if len(inner) >= 2:
                    counts[l]["split"] += 1
                    if len(ex[l]["split"]) < max_examples:
                        ex[l]["split"].append((wi, x, inner, toks))
    return counts, ex


def mark(toks, gold_like, pred_like, ctx=6):
    """Text around the involved spans, [[gold]] and <<pred>> marked."""
    allsp = list(gold_like) + list(pred_like)
    lo = max(0, min(s[0] for s in allsp) - ctx)
    hi = min(len(toks), max(s[1] for s in allsp) + ctx + 1)
    out = []
    for i in range(lo, hi):
        t = toks[i]
        for s in gold_like:
            if i == s[0]:
                t = "[[" + t
            if i == s[1]:
                t = t + "]]"
        for s in pred_like:
            if i == s[0]:
                t = "<<" + t
            if i == s[1]:
                t = t + ">>"
        out.append(t)
    return " ".join(out)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- report
def report(store, labels, threshold, out_dir=None):
    span, win = span_and_window_counts(store, labels, threshold)
    counts, ex = boundary_errors(store, labels, threshold)
    lines = []
    w = max(len(l) for l in labels) + 2
    lines.append(f"threshold {threshold}   windows {len(store)}")
    lines.append("\n[A] SPAN level (exact start, end, label)   TN not defined at span level")
    lines.append(f"{'type':<{w}}{'TP':>8}{'FP':>8}{'FN':>8}{'support':>9}{'P':>8}{'R':>8}{'F1':>8}")
    T = Counter()
    for l in labels:
        c = span[l]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        lines.append(f"{l:<{w}}{c['tp']:>8}{c['fp']:>8}{c['fn']:>8}{c['tp'] + c['fn']:>9}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
        T.update(c)
    p, r, f = prf(T["tp"], T["fp"], T["fn"])
    lines.append(f"{'ALL (micro)':<{w}}{T['tp']:>8}{T['fp']:>8}{T['fn']:>8}{T['tp'] + T['fn']:>9}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
    lines.append("\n[B] WINDOW level: window has >=1 span of this type? (gold vs predicted)")
    lines.append(f"{'type':<{w}}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}")
    for l in labels:
        c = win[l]
        lines.append(f"{l:<{w}}{c['tp']:>8}{c['fp']:>8}{c['fn']:>8}{c['tn']:>8}")
    kinds = ["merged", "split", "shifted", "wider", "narrower"]
    lines.append("\n[C] same-label BOUNDARY errors (counted in FP/FN above)")
    lines.append(f"{'type':<{w}}" + "".join(f"{k:>10}" for k in kinds))
    for l in labels:
        if sum(counts[l].values()):
            lines.append(f"{l:<{w}}" + "".join(f"{counts[l][k]:>10}" for k in kinds))
    lines.append(f"{'ALL':<{w}}" + "".join(f"{sum(counts[l][k] for l in labels):>10}" for k in kinds))
    lines.append("\nexamples ([[gold]]  <<predicted>>)")
    for l in labels:
        for k in kinds:
            for wi, a, others, toks in ex[l][k]:
                if toks is None:
                    continue
                gold_like, pred_like = ([a], others) if k == "split" else (others, [a])
                lines.append(f"  {l} | {k} | window {wi}: {mark(toks, gold_like, pred_like)}")
    text = "\n".join(lines)
    print(text)
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "error_matrix.txt").write_text(text, encoding="utf-8")
        with open(d / "per_type_confusion.csv", "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["type", "span_TP", "span_FP", "span_FN", "win_TP", "win_FP", "win_FN", "win_TN"]
                        + [f"bnd_{k}" for k in kinds])
            for l in labels:
                cw.writerow([l, span[l]["tp"], span[l]["fp"], span[l]["fn"], win[l]["tp"], win[l]["fp"], win[l]["fn"],
                             win[l]["tn"]] + [counts[l][k] for k in kinds])
        print(f"\nwrote {d / 'error_matrix.txt'} and {d / 'per_type_confusion.csv'}")


# ---------------------------------------------------------------- self-test
def selftest():
    toks = "a b lat : 1.5 , 2.5 z q".split()
    L = ["coord", "email"]
    # gold: two coordinates (4,4) and (6,6); model predicted ONE merged span (4,6)  -> merged
    # gold (2,2)? use a split case in a second window; shifted in a third; clean TP in a fourth; FP in fifth
    store = [
        ([(4, 4, "coord"), (6, 6, "coord")], [(4, 6, "coord", 0.9)], False, toks),
        ([(4, 6, "coord")], [(4, 4, "coord", 0.9), (6, 6, "coord", 0.8)], False, toks),
        ([(4, 5, "coord")], [(5, 6, "coord", 0.9)], False, toks),
        ([(0, 0, "email")], [(0, 0, "email", 0.9)], False, toks),
        ([], [(1, 1, "email", 0.9), (3, 3, "coord", 0.2)], False, toks),
    ]
    span, win = span_and_window_counts(store, L, 0.5)
    assert (span["coord"]["tp"], span["coord"]["fp"], span["coord"]["fn"]) == (0, 4, 4), span["coord"]
    assert (span["email"]["tp"], span["email"]["fp"], span["email"]["fn"]) == (1, 1, 0)
    assert dict(win["email"]) == {"tp": 1, "fp": 1, "tn": 3}, dict(win["email"])
    assert dict(win["coord"]) == {"tp": 3, "tn": 2}, dict(win["coord"])
    c, ex = boundary_errors(store, L, 0.5)
    assert c["coord"]["merged"] == 1 and c["coord"]["split"] == 1 and c["coord"]["shifted"] == 1, dict(c["coord"])
    assert sum(c["coord"].values()) == 3, dict(c["coord"])
    assert "[[1.5]]" in mark(toks, [(4, 4)], [(4, 6)]) or "<<[[1.5]]" in mark(toks, [(4, 4)], [(4, 6)]) or "[[<<1.5" in mark(toks, [(4, 4)], [(4, 6)])
    report(store, L, 0.5)
    print("\nSELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--data-path")
    ap.add_argument("--labels-file", default=None)
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--scaling-mode", default="row_sigma_learnable", choices=["block_max", "row_sigma", "row_sigma_learnable"])
    ap.add_argument("--max-types", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.model and a.data_path):
        ap.error("--model and --data-path are required")
    import torch
    from eval import evaluate_gliner_audit as ev
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels, src = ev.resolve_labels(a.model, a.labels_file)
    print(f"labels ({len(labels)}) from {src}")
    _s, parsed = ev.parse_dataset(Path(a.data_path), set(labels), a.limit)
    max_types = max(a.max_types, len(labels))
    if a.quantized:
        model = ev.load_quantized_checkpoint(a.model, base_model=a.base_model, device=device,
                                             scaling_mode=a.scaling_mode, learnable_scaling=True, max_types=max_types)
    else:
        model = ev.load_ordinary_checkpoint(a.model, base_model=a.base_model, device=device, max_types=max_types)
    out = ev.evaluate_model_on_dataset(model=model, parsed_examples=parsed, labels=labels, threshold=a.threshold,
                                       flat_ner=True, multi_label=False, allow_enclosing_fallback=False,
                                       batch_size=a.batch_size, return_store=True)
    report(out["_store"], labels, a.threshold, a.out_dir)


if __name__ == "__main__":
    main()
