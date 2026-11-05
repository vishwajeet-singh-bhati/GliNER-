"""Per-type span-level comparison of two configs from all_matrices.csv.
  python tools\\compare_configs.py eval_results\\matrices_fusion\\all_matrices.csv F4 F1
prints for each type: TP/FP/FN of A (baseline) and B (candidate), and the change in FP and FN, sorted by FP increase."""
import csv, sys
path, a, b = sys.argv[1:4]
rows = {}
for r in csv.DictReader(open(path, encoding="utf-8")):
    rows.setdefault(r["id"], {})[r["type"]] = {k: int(r["span_" + k]) for k in ("TP", "FP", "FN")}
A, B = rows[a], rows[b]
print(f"A={a}  B={b}   (span-level; dFP>0 means B has MORE false positives)")
print(f"{'type':<22}{'A TP':>7}{'A FP':>7}{'A FN':>7} |{'B TP':>7}{'B FP':>7}{'B FN':>7} |{'dFP':>7}{'dFN':>7}")
for t in sorted(A, key=lambda t: -(B[t]['FP'] - A[t]['FP'])):
    x, y = A[t], B[t]
    print(f"{t:<22}{x['TP']:>7}{x['FP']:>7}{x['FN']:>7} |{y['TP']:>7}{y['FP']:>7}{y['FN']:>7} |{y['FP']-x['FP']:>+7}{y['FN']-x['FN']:>+7}")
tot = lambda D, k: sum(v[k] for v in D.values())
print(f"{'TOTAL':<22}{tot(A,'TP'):>7}{tot(A,'FP'):>7}{tot(A,'FN'):>7} |{tot(B,'TP'):>7}{tot(B,'FP'):>7}{tot(B,'FN'):>7} |{tot(B,'FP')-tot(A,'FP'):>+7}{tot(B,'FN')-tot(A,'FN'):>+7}")
