"""
Run every configuration from the experimental-results PDF (Tables A, B, C) and write, for each one,
the per-type confusion tables in the format the professor asked for:   type  TP  FP  FN  TN

  span level   : TP / FP / FN of exact (start, end, label) matches (TN is not defined for spans)
  window level : TP / FP / FN / TN of "does the window contain >=1 span of this type"
  boundary     : merged / split / shifted / wider / narrower same-label errors, with examples

Each config is evaluated on the test set named in the PDF with the same evaluator, chunking and scoring as
eval/evaluate_gliner_audit.py.  E8-QAT checkpoints and their PTQ variants are loaded with the quantized loader
(--quantized behaviour, scaling mode row_sigma_learnable).  A fixed threshold is used where the PDF states one
(FP/PTQ capped = 0.5; Table B = 0.98 / 0.05 / 0.70 / 0.05); otherwise the threshold is selected on the validation
file and applied once, as in the main evaluator.  After each run the micro-F1 is compared with the PDF value;
a deviation > 0.01 is flagged MISMATCH so a bad checkpoint load cannot slip through unnoticed.

Safe to re-run: configs that already finished are skipped (use --force to redo).  Unattended runtime ~3 h on a 16 GB GPU.

PowerShell, repo root:
    python diagnostics\\run_all_matrices.py --out-dir eval_results\\matrices | Tee-Object eval_results\\matrices_log.txt
    python diagnostics\\run_all_matrices.py --only B1,B2 ...        # subset
    python diagnostics\\run_all_matrices.py --selftest
"""
import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "diagnostics"))
import error_matrix as em  # noqa: E402
from data_audit import digit_norm  # noqa: E402

FP = r"fp-v2_6800steps_2026-10-24\checkpoint-6792"
S3 = r"clean\s3-joint-10ep"
GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

# id, PDF table, row name, model dir (relative to --models-root), quantized loader?, test set, fixed threshold (None = select on val), PDF micro-F1
CONFIGS = [
    ("A1", "A capped", "Full-precision FP checkpoint-6792", FP, False, "capped", 0.5, 0.9213),
    ("A3", "A capped", "1-bit E8 QAT enc + INT8 emb/heads (218 MB)", S3 + r"\ptq-e8enc-emb8", True, "capped", None, 0.8945),
    ("A4", "A capped", "1-bit E8 QAT enc + INT4 emb, INT8 heads (121 MB)", S3 + r"\ptq-e8enc-emb4", True, "capped", None, 0.9157),
    ("B1", "B full", "Full-precision FP checkpoint-6792", FP, False, "full", 0.98, 0.862),
    ("B2", "B full", "1-bit E8 QAT, epoch 7 (step 15848)", S3 + r"\checkpoint-15848", True, "full", 0.05, 0.805),
    ("B3", "B full", "1-bit E8 QAT, epoch 9 (step 20376)", S3 + r"\checkpoint-20376", True, "full", 0.70, 0.790),
    ("B4", "B full", "1-bit E8 QAT, epoch 10 (step 22640)", S3 + r"\checkpoint-22640", True, "full", 0.05, 0.800),
    ("C1", "C dedup", "Full-precision FP", FP, False, "dedup", None, 0.879),
    ("C2", "C dedup", "1-bit E8 QAT + FP embeddings", S3 + r"\checkpoint-22640", True, "dedup", None, 0.819),
    ("C3", "C dedup", "1-bit E8 QAT + INT8 embeddings", S3 + r"\ptq-e8enc-emb8", True, "dedup", None, 0.816),
    ("C4", "C dedup", "1-bit E8 QAT + INT6 embeddings", S3 + r"\ptq-emb6", True, "dedup", None, 0.832),
    ("C5", "C dedup", "1-bit E8 QAT + INT4 emb, group size 128", S3 + r"\ptq-emb4-g128", True, "dedup", None, 0.884),
    ("C6", "C dedup", "1-bit E8 QAT + INT4 emb, per-row", S3 + r"\ptq-e8enc-emb4", True, "dedup", None, 0.887),
]


def mp(root, rel):
    return Path(root, *rel.split("\\"))


def build_dedup_file(full_path, out_path):
    """One window per digit-normalised pattern (same rule as data_audit.py --write-dedup)."""
    seen, keep = set(), []
    for line in open(full_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        h = digit_norm(r["tokenized_text"])
        if h not in seen:
            seen.add(h)
            keep.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    return len(keep)


def micro_macro(span, labels):
    tp = sum(span[l]["tp"] for l in labels)
    fp = sum(span[l]["fp"] for l in labels)
    fn = sum(span[l]["fn"] for l in labels)
    p, r, f = em.prf(tp, fp, fn)
    f1s = [em.prf(span[l]["tp"], span[l]["fp"], span[l]["fn"])[2] for l in labels]
    sup = [em.prf(span[l]["tp"], span[l]["fp"], span[l]["fn"])[2] for l in labels if span[l]["tp"] + span[l]["fn"] > 0]
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, micro_f1=f,
                macro_f1_all16=sum(f1s) / len(f1s), macro_f1_supported=sum(sup) / max(1, len(sup)))


def md_section(cfg, thr, n_windows, labels, span, win, counts, m, note):
    cid, table, name, _p, _q, test, _t, expect = cfg
    w = max(len(l) for l in labels) + 2
    L = [f"## {cid} | Table {table} | {name}",
         f"test set: {test} ({n_windows} windows) | threshold {thr:g} | micro P {m['precision']:.3f}  R {m['recall']:.3f}  "
         f"F1 {m['micro_f1']:.4f} (PDF {expect}) | macro-F1 {m['macro_f1_all16']:.3f} (16 types) / "
         f"{m['macro_f1_supported']:.3f} (types with support) | {note}", "",
         "Window level (window has >=1 span of the type):", "```",
         f"{'type':<{w}}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}"]
    for l in labels:
        c = win[l]
        L.append(f"{l:<{w}}{c['tp']:>8}{c['fp']:>8}{c['fn']:>8}{c['tn']:>8}")
    L += ["```", "", "Span level (exact start, end, label; TN not defined):", "```",
          f"{'type':<{w}}{'TP':>8}{'FP':>8}{'FN':>8}{'support':>9}{'P':>8}{'R':>8}{'F1':>8}"]
    for l in labels:
        c = span[l]
        p, r, f = em.prf(c["tp"], c["fp"], c["fn"])
        L.append(f"{l:<{w}}{c['tp']:>8}{c['fp']:>8}{c['fn']:>8}{c['tp'] + c['fn']:>9}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
    L += [f"{'ALL (micro)':<{w}}{m['tp']:>8}{m['fp']:>8}{m['fn']:>8}{m['tp'] + m['fn']:>9}"
          f"{m['precision']:>8.3f}{m['recall']:>8.3f}{m['micro_f1']:>8.3f}", "```", ""]
    kinds = ["merged", "split", "shifted", "wider", "narrower"]
    tot = {k: sum(counts[l][k] for l in labels) for k in kinds}
    L.append("Same-label boundary errors (included in FP/FN above): " + ", ".join(f"{k} {v}" for k, v in tot.items()))
    per = [f"{l}: " + ", ".join(f"{k} {counts[l][k]}" for k in kinds if counts[l][k]) for l in labels if sum(counts[l].values())]
    if per:
        L.append("  by type: " + " | ".join(per))
    L.append("")
    return "\n".join(L)


def evaluate_one(cfg, ev, ctx, args):
    cid, table, name, rel, quant, test, fixed_thr, expect = cfg
    model_dir = str(mp(args.models_root, rel))
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    labels = ctx["labels"]
    if quant:
        model = ev.load_quantized_checkpoint(model_dir, base_model=args.base_model, device=ctx["device"],
                                             scaling_mode="row_sigma_learnable", learnable_scaling=True,
                                             max_types=max(25, len(labels)))
    else:
        model = ev.load_ordinary_checkpoint(model_dir, base_model=args.base_model, device=ctx["device"],
                                            max_types=max(25, len(labels)))
    common = dict(model=model, labels=labels, flat_ner=True, multi_label=False, allow_enclosing_fallback=False,
                  max_len_override=None, progress_every=3000, chunk_unit="words", batch_size=args.batch_size,
                  max_subwords=1024)
    if fixed_thr is not None:
        thr, note = fixed_thr, "threshold fixed (as in the PDF)"
    else:
        v = ev.evaluate_model_on_dataset(parsed_examples=ctx["val"], threshold=min(GRID), sweep=None,
                                         return_store=True, **common)
        thr, _tab, _w = ev.select_threshold(v["_store"], GRID, None)
        note = "threshold selected on the validation file"
    out = ev.evaluate_model_on_dataset(parsed_examples=ctx["tests"][test], threshold=thr, sweep=None,
                                       return_store=True, **common)
    store = out["_store"]
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    span, win = em.span_and_window_counts(store, labels, thr)
    counts, _ex = em.boundary_errors(store, labels, thr)
    m = micro_macro(span, labels)
    d = Path(args.out_dir) / cid
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        em.report(store, labels, thr, d)
    section = md_section(cfg, thr, len(store), labels, span, win, counts, m, note)
    flag = "OK" if abs(m["micro_f1"] - expect) <= 0.01 else "MISMATCH"
    (d / "section.md").write_text(section, encoding="utf-8")
    res = dict(id=cid, table=table, name=name, test=test, thr=thr, check=flag, expect=expect, m=m, section=section,
               span={l: {k: span[l][k] for k in ("tp", "fp", "fn")} for l in labels},
               win={l: {k: win[l][k] for k in ("tp", "fp", "fn", "tn")} for l in labels}, labels=labels)
    (d / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def write_combined(results, out_dir):
    out = Path(out_dir)
    head = ["# Per-type confusion tables for all configurations in the results PDF", "",
            "TP/FP/FN/TN at window level, TP/FP/FN at span level. 'check' compares micro-F1 with the PDF value.", "",
            "| id | table | configuration | test | thr | micro P | micro R | micro F1 | PDF F1 | check |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["m"]
        head.append(f"| {r['id']} | {r['table']} | {r['name']} | {r['test']} | {r['thr']:g} | {m['precision']:.3f} | "
                    f"{m['recall']:.3f} | {m['micro_f1']:.4f} | {r['expect']} | {r['check']} |")
    (out / "all_matrices.md").write_text("\n".join(head) + "\n\n" + "\n".join(r["section"] for r in results), encoding="utf-8")
    with open(out / "all_matrices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "table", "configuration", "test_set", "threshold", "type", "span_TP", "span_FP", "span_FN",
                    "window_TP", "window_FP", "window_FN", "window_TN", "check"])
        for r in results:
            for l in r["labels"]:
                s, wn = r["span"][l], r["win"][l]
                w.writerow([r["id"], r["table"], r["name"], r["test"], r["thr"], l, s["tp"], s["fp"], s["fn"],
                            wn["tp"], wn["fp"], wn["fn"], wn["tn"], r["check"]])


def run(args, ev):
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    labels = sorted(json.loads(Path(args.labels_file or mp(args.models_root, FP.split("\\")[0] + "\\train_labels.json"))
                               .read_text(encoding="utf-8")))
    print(f"labels ({len(labels)}): {labels}")
    paths = {"capped": data / "test_capped_finetune.jsonl", "full": data / "test_finetune.jsonl",
             "dedup": data / "test_dedup_finetune.jsonl"}
    if not paths["dedup"].exists():
        n = build_dedup_file(paths["full"], paths["dedup"])
        print(f"built {paths['dedup']} with {n} windows (PDF: 15,133)")
    tests = {}
    for k, p in paths.items():
        _s, parsed = ev.parse_dataset(p, set(labels), None)
        tests[k] = parsed
        print(f"test {k}: {len(parsed)} windows from {p}")
    _s, val = ev.parse_dataset(data / "val_finetune.jsonl", set(labels), None)
    ctx = dict(labels=labels, device=device, tests=tests, val=val)
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, failed = [], []
    for cfg in CONFIGS:
        if only and cfg[0] not in only:
            continue
        done = out / cfg[0] / "result.json"
        if done.exists() and not args.force:
            print(f"[{cfg[0]}] already done, skipping (use --force to redo)")
            results.append(json.loads(done.read_text(encoding="utf-8")))
            continue
        t0 = time.time()
        print(f"\n===== [{cfg[0]}] {cfg[1]} | {cfg[2]} | test={cfg[5]} =====", flush=True)
        try:
            r = evaluate_one(cfg, ev, ctx, args)
            results.append(r)
            m = r["m"]
            print(f"[{cfg[0]}] threshold {r['thr']:g}  P {m['precision']:.3f} R {m['recall']:.3f} micro-F1 {m['micro_f1']:.4f} "
                  f"(PDF {r['expect']}) -> {r['check']}   [{time.time() - t0:.0f}s]", flush=True)
        except Exception as exc:
            print(f"[{cfg[0]}] FAILED: {exc!r}", flush=True)
            failed.append((cfg[0], repr(exc)))
        if results:
            write_combined(results, out)
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"{r['id']}  {r['check']:<9} micro-F1 {r['m']['micro_f1']:.4f} (PDF {r['expect']})  {r['name']}")
    for cid, e in failed:
        print(f"{cid}  FAILED  {e}")
    if results:
        print(f"\nwrote {out / 'all_matrices.md'} and {out / 'all_matrices.csv'}; per-config folders in {out}")
    if any(r["check"] == "MISMATCH" for r in results):
        print("WARNING: at least one config deviates >0.01 from the PDF: do not send that table (see CHECK column).")


def selftest():
    import types
    L = ["coord", "email"]
    toks = "a b lat : 1.5 , 2.5 z q".split()
    store_by_test = {"capped": [([(4, 4, "coord")], [(4, 4, "coord", 0.9)], False, toks)],
                     "full": [([(4, 4, "coord"), (6, 6, "coord")], [(4, 6, "coord", 0.9)], False, toks)],
                     "dedup": [([(0, 0, "email")], [(0, 0, "email", 0.9)], False, toks)]}
    calls = []

    def parse_dataset(p, labels, limit):
        key = "capped" if "capped" in str(p) else "dedup" if "dedup" in str(p) else "val" if "val_" in str(p) else "full"
        return None, [("tag", key)]

    def load(model_dir, **kw):
        calls.append((Path(model_dir).name, kw.get("scaling_mode")))
        return object()

    def evaluate_model_on_dataset(parsed_examples, threshold, return_store=False, **kw):
        key = parsed_examples[0][1]
        return {"_store": store_by_test.get(key, store_by_test["full"])}

    ev = types.SimpleNamespace(parse_dataset=parse_dataset, load_quantized_checkpoint=load, load_ordinary_checkpoint=load,
                               evaluate_model_on_dataset=evaluate_model_on_dataset,
                               select_threshold=lambda store, grid, tpr: (0.5, {}, 1.0))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "data").mkdir()
        rows = [{"tokenized_text": ["x", "1"], "ner": []}, {"tokenized_text": ["x", "2"], "ner": []},
                {"tokenized_text": ["y", "3"], "ner": []}]
        (d / "data" / "test_finetune.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        for n in ["test_capped_finetune.jsonl", "val_finetune.jsonl"]:
            (d / "data" / n).write_text("")
        root = d / "models"
        for _id, _t, _n, rel, *_r in CONFIGS:
            mp(root, rel).mkdir(parents=True, exist_ok=True)
        (root / FP.split("\\")[0] / "train_labels.json").write_text(json.dumps(L))
        args = argparse.Namespace(models_root=str(root), data_dir=str(d / "data"), out_dir=str(d / "out"), only="A1,A3,B4,C5",
                                  force=False, device="cpu", batch_size=4, base_model="x", labels_file=None)
        run(args, ev)
        md = (d / "out" / "all_matrices.md").read_text(encoding="utf-8")
        csvt = (d / "out" / "all_matrices.csv").read_text(encoding="utf-8")
        assert "## A1" in md and "## B4" in md and "merged 1" in md, md
        assert "window_TN" in csvt and "MISMATCH" in md
        assert ("checkpoint-22640", "row_sigma_learnable") in calls and ("checkpoint-6792", None) in calls, calls
        assert build_dedup_file(d / "data" / "test_finetune.jsonl", d / "dd.jsonl") == 2   # x 1 / x 2 collapse
    print("\nSELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default=r"D:\gliner-antshield\models")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=r"eval_results\matrices")
    ap.add_argument("--labels-file", default=None, help="default: train_labels.json next to the FP model")
    ap.add_argument("--base-model", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--only", default=None, help="comma list of config ids, e.g. A1,B2")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    from eval import evaluate_gliner_audit as ev
    run(a, ev)


if __name__ == "__main__":
    main()
