"""
Build ONE forwardable Excel workbook from eval_results\\matrices\\all_matrices.csv (written by
diagnostics\\run_all_matrices.py).  Sheets: Summary, A capped, B full, C dedup, Notes.
Each config block uses the professor's layout:  type | TP | FP | FN | TN   (window level)
followed by the span-level TP / FP / FN, precision, recall, F1.

    pip install openpyxl            (once, if missing)
    python tools\\make_forward_xlsx.py --csv eval_results\\matrices\\all_matrices.csv --out eval_results\\gliner_pii_per_type_results.xlsx
    python tools\\make_forward_xlsx.py --selftest
"""
import argparse
import csv
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

PDF_F1 = {"A1": 0.9213, "A3": 0.8945, "A4": 0.9157, "B1": 0.862, "B2": 0.805, "B3": 0.790, "B4": 0.800,
          "C1": 0.879, "C2": 0.819, "C3": 0.816, "C4": 0.832, "C5": 0.884, "C6": 0.887}
SHEETS = OrderedDict([("A", "A capped"), ("B", "B full"), ("C", "C dedup")])
TEST_DESC = {"capped": "capped test set (<=5 windows per request template)",
             "full": "full test set (all windows, manual_anteater campaign)",
             "dedup": "de-duplicated test set (one window per digit-normalised pattern)"}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load(path):
    cfgs = OrderedDict()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = cfgs.setdefault(r["id"], dict(id=r["id"], table=r["table"], name=r["configuration"], test=r["test_set"],
                                              thr=float(r["threshold"]), rows=[]))
            c["rows"].append(dict(type=r["type"], **{k: int(r[k]) for k in
                                  ("span_TP", "span_FP", "span_FN", "window_TP", "window_FP", "window_FN", "window_TN")}))
    for c in cfgs.values():
        tp = sum(x["span_TP"] for x in c["rows"]); fp = sum(x["span_FP"] for x in c["rows"]); fn = sum(x["span_FN"] for x in c["rows"])
        c["P"], c["R"], c["F1"] = prf(tp, fp, fn)
        c["tp"], c["fp"], c["fn"] = tp, fp, fn
        c["macro"] = sum(prf(x["span_TP"], x["span_FP"], x["span_FN"])[2] for x in c["rows"]) / len(c["rows"])
        x0 = c["rows"][0]
        c["windows"] = x0["window_TP"] + x0["window_FP"] + x0["window_FN"] + x0["window_TN"]
    return cfgs


def build(cfgs, out):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    bold = Font(bold=True)
    head = PatternFill("solid", fgColor="DDE6F2")
    title = PatternFill("solid", fgColor="F3D9D5")
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Per-type results: 1-bit E8 QAT for GLiNER PII-type extraction"])
    ws["A1"].font = Font(bold=True, size=13)
    ws.append(["Span-level exact match (start, end, label). Thresholds as stated; see Notes."])
    ws.append([])
    hdr = ["ID", "Table", "Configuration", "Test windows", "Threshold", "Precision", "Recall", "Micro-F1", "Macro-F1 (16 types)",
           "Value in earlier PDF", "Difference"]
    ws.append(hdr)
    for c in ws[4]:
        c.font = bold; c.fill = head
    for c in cfgs.values():
        exp = PDF_F1.get(c["id"])
        ws.append([c["id"], c["table"], c["name"], c["windows"], c["thr"], round(c["P"], 3), round(c["R"], 3),
                   round(c["F1"], 4), round(c["macro"], 3), exp, round(c["F1"] - exp, 4) if exp else None])
    for col, w in zip("ABCDEFGHIJK", (6, 12, 46, 14, 11, 11, 9, 10, 18, 18, 11)):
        ws.column_dimensions[col].width = w
    for letter, sheet in SHEETS.items():
        s = wb.create_sheet(sheet)
        s.column_dimensions["A"].width = 26
        for col in "BCDEFGHIJK":
            s.column_dimensions[col].width = 10
        row = 1
        for c in [x for x in cfgs.values() if x["table"].startswith(letter)]:
            s.cell(row, 1, f"{c['id']} | {c['name']}").font = Font(bold=True, size=12)
            for j in range(1, 12):
                s.cell(row, j).fill = title
            row += 1
            s.cell(row, 1, f"Test set: {TEST_DESC.get(c['test'], c['test'])}, {c['windows']} windows | threshold {c['thr']:g} | "
                           f"micro P {c['P']:.3f}, R {c['R']:.3f}, F1 {c['F1']:.4f} | macro-F1 {c['macro']:.3f}")
            row += 1
            s.cell(row, 1, "Window level (>=1 span of the type in the window)").font = Font(italic=True)
            s.cell(row, 6, "Span level (exact match)").font = Font(italic=True)
            row += 1
            for j, h in enumerate(["type", "TP", "FP", "FN", "TN", "TP", "FP", "FN", "Precision", "Recall", "F1"], 1):
                cell = s.cell(row, j, h); cell.font = bold; cell.fill = head
                cell.alignment = Alignment(horizontal="center" if j > 1 else "left")
            row += 1
            for x in c["rows"]:
                p, r, f = prf(x["span_TP"], x["span_FP"], x["span_FN"])
                vals = [x["type"], x["window_TP"], x["window_FP"], x["window_FN"], x["window_TN"],
                        x["span_TP"], x["span_FP"], x["span_FN"], round(p, 3), round(r, 3), round(f, 3)]
                for j, v in enumerate(vals, 1):
                    s.cell(row, j, v)
                row += 1
            for j, v in enumerate(["ALL (micro, span level)", None, None, None, None, c["tp"], c["fp"], c["fn"],
                                   round(c["P"], 3), round(c["R"], 3), round(c["F1"], 3)], 1):
                cell = s.cell(row, j, v); cell.font = bold
            row += 3
    n = wb.create_sheet("Notes")
    notes = [
        "Definitions",
        "Span level: a prediction is a TP only if start token, end token and label all match a gold span exactly. TN is not defined at span level.",
        "Window level: for each PII type, a window is positive if it contains at least one span of that type (gold) or one predicted span (prediction). TN = windows with neither.",
        "TN values are large because most windows contain no PII; accuracy or specificity would therefore look near-perfect. Precision, recall and F1 are the informative measures.",
        "Types with fewer than ~30 test spans (device serial number, email address, gender, last name, password, phone number, sim card iccid) give unstable per-type scores.",
        "Thresholds: FP and PTQ capped rows use a fixed 0.5; Table B uses the thresholds listed in the Summary; other rows use a threshold selected on the validation file (grid 0.05-0.99) and applied once to the test set.",
        "The best threshold for most QAT rows is the lowest grid value (0.05), so the optimum may lie slightly lower.",
        "Differences from the earlier PDF (up to about 0.007 micro-F1) come from re-running the evaluation with this threshold grid.",
        "Data limitation: PII values in the dataset are synthetic replacements for redacted values placed in real traffic; repeated request templates influence the counts.",
    ]
    for i, t in enumerate(notes, 1):
        n.cell(i, 1, t)
    n["A1"].font = bold
    n.column_dimensions["A"].width = 160
    wb.save(out)


def selftest():
    labels = ["coord", "email"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "table", "configuration", "test_set", "threshold", "type", "span_TP", "span_FP", "span_FN",
                        "window_TP", "window_FP", "window_FN", "window_TN", "check"])
            for cid, tb, ts in [("A1", "A capped", "capped"), ("B2", "B full", "full"), ("C6", "C dedup", "dedup")]:
                for l, a in zip(labels, [(8, 2, 2, 6, 1, 1, 12), (3, 1, 0, 3, 1, 0, 16)]):
                    w.writerow([cid, tb, "cfg " + cid, ts, 0.5, l, *a, "OK"])
        cf = load(p)
        assert cf["A1"]["windows"] == 20 and abs(cf["A1"]["P"] - 11 / 14) < 1e-9
        out = Path(d) / "o.xlsx"
        build(cf, out)
        from openpyxl import load_workbook
        wb = load_workbook(out)
        assert wb.sheetnames == ["Summary", "A capped", "B full", "C dedup", "Notes"], wb.sheetnames
        assert wb["A capped"]["A1"].value.startswith("A1 |")
        assert wb["Summary"]["H5"].value == round(cf["A1"]["F1"], 4)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"eval_results\matrices\all_matrices.csv")
    ap.add_argument("--out", default=r"eval_results\gliner_pii_per_type_results.xlsx")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    cf = load(a.csv)
    build(cf, a.out)
    print(f"wrote {a.out}: {len(cf)} configurations")


if __name__ == "__main__":
    main()
