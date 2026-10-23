#!/usr/bin/env python3
"""
Rebuilds train/test fine-tuning data from the ORIGINAL antshield_public_dataset,
reusing the existing preprocessing unchanged (antshield_loader header whitelist,
RECON_ masking via recon_synthesizer, GLiNER tokenization, 300-token windows /
30 overlap / max 3 windows per record) and adding:

  * a real VALIDATION split (whole apps carved out of the campaign-routed train flows, natural class
    prior, exact-duplicate windows removed) used for early stopping / threshold selection.
  * per-class routing (see synth_config.py): classes in SYNTHETIC_ONLY have all
    their real flows in TEST; classes in SPLIT_20_80 get ~20% of real flows in
    TRAIN (grouped by app) and ~80% in TEST; all other classes keep the original
    campaign rule (auto_anteater -> train, manual_anteater -> test).
  * synthetic top-up in TRAIN for the first two groups (synth_inject/synth_values),
    placed in real negative contexts, plus non-PII decoys built the same way.
  * entity-free train windows capped at --neg-ratio x positive windows (seeded).
  * leakage guard on a CANONICAL fingerprint (entity values -> <LABEL>, digits -> 0, long ids -> X):
    train windows with a canonical twin in val/test, and val windows with a twin in test, are dropped.
  * exact-duplicate windows are removed from val/test (3k identical NUL-blob windows inflated the
    old test negatives) and entity-free train windows are de-duplicated before sampling.
  * line-aligned manifests: every output line is traceable (real flow / synthetic / decoy).

Usage (from repo root):
  python data/build_dataset.py --stats-only      # class x campaign table + routing, writes nothing
  python data/build_dataset.py                   # writes data/{train,val,test}_finetune.jsonl (+ manifests, report)
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antshield_loader import load_antshield_dataset  # noqa: E402
from pii_type_mapping import all_gliner_labels  # noqa: E402
from prepare_finetune_data import build_char_entities, chunk_and_tokenize  # noqa: E402
from gliner_tokenizer import tokenize_with_offsets  # noqa: E402
from splitting import assign_split_campaign, split_bucket  # noqa: E402
import synth_config as cfg  # noqa: E402
from synth_inject import inject, pick_key, spans_align_exactly  # noqa: E402
from synth_values import decoy_field, generate_value  # noqa: E402

Flow = namedtuple("Flow", "record_id campaign package label text entities")
SYNTH_CLASSES = cfg.SYNTHETIC_ONLY | cfg.SPLIT_20_80


def policy_of(label):
    if label in cfg.SYNTHETIC_ONLY:
        return "synthetic_only"
    if label in cfg.SPLIT_20_80:
        return "split_20_80"
    return "campaign"


def load_flows(args):
    records = load_antshield_dataset(args.dataset_root, include_metadata=args.include_metadata_fields)
    flows = []
    for r in records:
        # identical masking call to prepare_finetune_data.main (mask_recon_tokens=True)
        text, ents = build_char_entities(r.text, True, seed_prefix=r.record_id)
        campaign = "manual" if "manual_anteater" in r.record_id else "auto"
        flows.append(Flow(r.record_id, campaign, r.package_name, r.label, text, ents))
    flows.sort(key=lambda f: f.record_id)  # glob order is filesystem-dependent; sort for reproducibility
    return flows


def route_flows(flows, seed, val_frac=0.10):
    route, pool = {}, []
    for f in flows:
        pols = {policy_of(l) for _, _, l in f.entities}
        if "synthetic_only" in pols:
            route[f.record_id] = "test"
        elif "split_20_80" in pols:
            pool.append(f)
        else:  # original campaign rule; the old per-record val carve-out goes back to train
            route[f.record_id] = "test" if assign_split_campaign(f.record_id) == "test" else "train"
    # VAL: carve whole apps (packages) out of the campaign-routed train flows, so val shares no app
    # (hence no per-app traffic signature) with train.  Synthetic-policy classes keep all their train flows.
    camp = defaultdict(list)
    for f in flows:
        if route.get(f.record_id) == "train" and not any(policy_of(l) != "campaign" for _, _, l in f.entities):
            camp[f.package].append(f)
    n_camp = sum(len(v) for v in camp.values())
    vtarget, n_val = int(val_frac * n_camp), 0
    for p in sorted(camp, key=lambda p: split_bucket(f"{seed}:val:{p}")):
        if n_val + len(camp[p]) <= vtarget:
            for f in camp[p]:
                route[f.record_id] = "val"
            n_val += len(camp[p])
    groups = defaultdict(list)
    for f in pool:
        groups[f.package].append(f)
    order = sorted(groups, key=lambda p: split_bucket(f"{seed}:{p}"))
    target = int(cfg.REAL_TRAIN_FRACTION * len(pool))
    train_pk, n_train = set(), 0
    for p in order:  # greedy fill by whole app groups, never above the 20% target
        if n_train + len(groups[p]) <= target:
            train_pk.add(p)
            n_train += len(groups[p])
    if pool and not train_pk:
        train_pk.add(min(order, key=lambda p: len(groups[p])))
    for f in pool:
        route[f.record_id] = "train" if f.package in train_pk else "test"
    return route


def print_flow_table(flows, route):
    t = defaultdict(Counter)
    for f in flows:
        for lab in {l for _, _, l in f.entities}:
            t[lab]["flows"] += 1
            t[lab][f.campaign] += 1
            t[lab]["->" + route[f.record_id]] += 1
    print(f"\n{'label':<28}{'policy':<16}{'flows':>7}{'auto':>7}{'manual':>8}{'->train':>9}{'->test':>8}")
    for lab in all_gliner_labels():
        c = t[lab]
        print(f"{lab:<28}{policy_of(lab):<16}{c['flows']:>7}{c['auto']:>7}{c['manual']:>8}{c['->train']:>9}{c['->test']:>8}")
    n_neg = sum(1 for f in flows if not f.entities)
    print(f"\nflows total {len(flows)}; without entities {n_neg}; "
          f"routed train {sum(v == 'train' for v in route.values())} / val {sum(v == 'val' for v in route.values())}"
          f" / test {sum(v == 'test' for v in route.values())}")


def make_windows(flows, split, route, args):
    out, dropped_total = [], 0
    for f in flows:
        if route[f.record_id] != split:
            continue
        wins, dropped = chunk_and_tokenize(f.text, f.entities, args.max_tokens, args.overlap_tokens,
                                           args.max_windows_per_record)
        dropped_total += dropped
        for w in wins:
            out.append((w, {"source": "real", "record_id": f.record_id, "campaign": f.campaign}))
    return out, dropped_total


def build_carriers(flows, route, args, rng):
    """Real, label==0, entity-free TRAIN flows short enough to stay in one window after injection."""
    cands = [f.text for f in flows
             if route[f.record_id] == "train" and f.label == 0 and not f.entities and 40 <= len(f.text) <= 2500]
    rng.shuffle(cands)
    carriers = []
    for t in cands:
        if len(tokenize_with_offsets(t)) <= args.max_tokens - 40:
            carriers.append(t)
            if len(carriers) >= args.carrier_pool:
                break
    return carriers


def _one_window(text, spans, args):
    wins, dropped = chunk_and_tokenize(text, spans, args.max_tokens, args.overlap_tokens, args.max_windows_per_record)
    return wins[0] if (not dropped and len(wins) == 1) else None


def make_synthetic(primary, rng, carriers, args):
    for _ in range(30):
        labels = [primary]
        if rng.random() < args.cooccur_prob:
            for group in cfg.COOCCUR_GROUPS:
                if primary in group:
                    partners = [g for g in group if g != primary and g in SYNTH_CLASSES]
                    if partners:
                        labels.append(rng.choice(partners))
                    break
        fields = [(l, pick_key(l, rng), generate_value(l, rng)) for l in labels]
        carrier = rng.choice(carriers)
        res = inject(carrier, fields, rng)
        if res is None:
            continue
        text, spans = res
        w = _one_window(text, spans, args)
        if w is None or not spans_align_exactly(text, spans, w):
            continue
        meta = {"source": "synthetic", "labels": sorted(set(labels)),
                "carrier": hashlib.sha1(carrier.encode("utf-8")).hexdigest()[:12]}
        return w, meta, labels
    return None


def make_decoy(rng, carriers, args):
    for _ in range(30):
        fields = [(None, *decoy_field(rng)) for _ in range(rng.choice([1, 1, 2]))]
        carrier = rng.choice(carriers)
        res = inject(carrier, fields, rng)
        if res is None:
            continue
        w = _one_window(res[0], [], args)
        if w is not None and not w["ner"]:
            return w, {"source": "synthetic_decoy", "carrier": hashlib.sha1(carrier.encode("utf-8")).hexdigest()[:12]}
    return None


def has_overlap(w):
    sp = sorted(w["ner"])
    return any(sp[k][0] <= sp[k - 1][1] for k in range(1, len(sp)))


def fingerprint(w):
    return hashlib.md5(json.dumps([w["tokenized_text"], w["ner"]], separators=(",", ":")).encode("utf-8")).hexdigest()


_HEXY = re.compile(r"[0-9A-Za-z_-]{12,}")


def canonical_fingerprint(w):
    """Twin detector: windows that differ only in entity values / digits / long ids are the same
    template (the masker re-draws values per record, so exact hashes never match)."""
    lab_at = {}
    for s_, e_, l_ in w["ner"]:
        for i in range(s_, e_ + 1):
            lab_at[i] = l_
    out, prev = [], None
    for i, t in enumerate(w["tokenized_text"]):
        if i in lab_at:
            if prev != lab_at[i]:
                out.append("<" + lab_at[i] + ">")
            prev = lab_at[i]
            continue
        prev = None
        if _HEXY.fullmatch(t) and re.search(r"\d", t):
            out.append("X")
        else:
            out.append(re.sub(r"\d", "0", t))
    return hashlib.md5("\x1f".join(out).encode("utf-8")).hexdigest()


def dedupe_exact(items, entity_free_only=True):
    """Drop exact-duplicate windows (keeps the first).  By default only entity-free ones are
    collapsed; windows with entities are real evidence and are kept as they are."""
    seen, kept, dropped = set(), [], 0
    for w, m in items:
        if entity_free_only and w["ner"]:
            kept.append((w, m))
            continue
        fp = fingerprint(w)
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        kept.append((w, m))
    return kept, dropped


def labels_of(w):
    return sorted({n[2] for n in w["ner"]})


def write_split(out_dir, name, items):
    p, m = out_dir / f"{name}_finetune.jsonl", out_dir / f"{name}_manifest.jsonl"
    with open(p, "w", encoding="utf-8") as fp, open(m, "w", encoding="utf-8") as fm:
        for i, (w, meta) in enumerate(items):
            fp.write(json.dumps(w) + "\n")  # same serialisation as prepare_finetune_data
            fm.write(json.dumps({"i": i, **meta, "labels": labels_of(w)}) + "\n")
    return p, m


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_table(items):
    win, span = Counter(), Counter()
    for w, meta in items:
        key = meta["source"]
        for lab in labels_of(w):
            win[(lab, key)] += 1
        for n in w["ner"]:
            span[(n[2], key)] += 1
    return win, span


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="antshield_public_dataset")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--overlap-tokens", type=int, default=30)
    ap.add_argument("--max-windows-per-record", type=int, default=3)
    ap.add_argument("--include-metadata-fields", action="store_true")
    ap.add_argument("--target-windows", type=int, default=cfg.TARGET_TRAIN_WINDOWS,
                    help="train windows per synthetic-policy class (real + synthetic)")
    ap.add_argument("--neg-ratio", type=float, default=3.0,
                    help="train entity-free windows : positive windows (3.0 ~ the real 14.5%% test prior; the old 1.0 gave a 51%% positive train prior)")
    ap.add_argument("--test-cap-per-template", type=int, default=5,
                    help="also write test_capped_finetune.jsonl keeping at most N windows per canonical template "
                         "(one app can otherwise supply thousands of near-identical windows: 1,239 Roblox windows = "
                         "5.5%% of the test set and 53%% of its false positives)")
    ap.add_argument("--val-frac", type=float, default=0.10, help="fraction of campaign-routed train flows moved to val (whole apps)")
    ap.add_argument("--decoy-ratio", type=float, default=0.5, help="decoy windows per synthetic positive window")
    ap.add_argument("--cooccur-prob", type=float, default=0.2)
    ap.add_argument("--carrier-pool", type=int, default=3000)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    valid = set(all_gliner_labels())
    assert SYNTH_CLASSES <= valid, f"unknown labels in synth_config: {SYNTH_CLASSES - valid}"
    out_dir = Path(args.output_dir)
    outputs = [out_dir / n for n in ("train_finetune.jsonl", "val_finetune.jsonl", "test_finetune.jsonl",
                                     "train_manifest.jsonl", "val_manifest.jsonl", "test_manifest.jsonl",
                                     "build_report.json")]
    if not args.stats_only and not args.force and any(p.exists() for p in outputs):
        sys.exit("Output files already exist: " + ", ".join(str(p) for p in outputs if p.exists())
                 + "\nMove them away or pass --force.")

    rng = random.Random(f"{args.seed}:build")
    print("Loading + masking flows (same steps as prepare_finetune_data) ...")
    flows = load_flows(args)
    route = route_flows(flows, args.seed, args.val_frac)
    print_flow_table(flows, route)
    if args.stats_only:
        return

    # ---- real windows ----------------------------------------------------------
    train_real, d1 = make_windows(flows, "train", route, args)
    test_items, d2 = make_windows(flows, "test", route, args)
    val_items, d3 = make_windows(flows, "val", route, args)
    test_items, test_dups = dedupe_exact(test_items)
    val_items, val_dups = dedupe_exact(val_items)
    # adjacent markers (e.g. RECON_SerialNumberRECON_AndroidId) merge into ONE token after masking, so two
    # entities share a token: corrupt supervision. Dropped from train only; test is left exactly as built.
    n_overlap = sum(has_overlap(w) for w, _ in train_real)
    train_real = [x for x in train_real if not has_overlap(x[0])]
    real_pos = [x for x in train_real if x[0]["ner"]]
    real_neg_all = [x for x in train_real if not x[0]["ner"]]
    seen_c, real_neg = Counter(), []
    for x in real_neg_all:                      # cap template twins (<=8 per template) so negatives stay diverse
        c = canonical_fingerprint(x[0])
        if seen_c[c] < 8:
            seen_c[c] += 1
            real_neg.append(x)
    print(f"\nreal train windows {len(train_real)} (with entities {len(real_pos)}); "
          f"real val windows {len(val_items)} (exact-duplicate entity-free removed: {val_dups}); "
          f"real test windows {len(test_items)} (exact-duplicate entity-free removed: {test_dups}); entities dropped by chunking {d1 + d2 + d3}; "
          f"real train windows dropped for overlapping spans {n_overlap}; "
          f"real test windows with overlapping spans (kept) {sum(has_overlap(w) for w, _ in test_items)}")

    # ---- synthetic top-up ------------------------------------------------------
    carriers = build_carriers(flows, route, args, rng)
    if not carriers:
        sys.exit("No usable carrier flows (real, label 0, short, in train).")
    print(f"carrier pool: {len(carriers)} real negative train flows")
    have = Counter()
    for w, _ in real_pos:
        for lab in labels_of(w):
            have[lab] += 1
    flows_per_label = Counter(l for f in flows for l in {e[2] for e in f.entities})
    syn_pos = []
    for lab in sorted(SYNTH_CLASSES, key=lambda c: flows_per_label[c]):  # rarest first
        fails = 0
        while have[lab] < args.target_windows:
            res = make_synthetic(lab, rng, carriers, args)
            if res is None:
                fails += 1
                if fails > 50:
                    sys.exit(f"could not generate synthetic data for '{lab}' (carriers too long?)")
                continue
            w, meta, labels = res
            syn_pos.append((w, meta))
            for l in set(labels):
                have[l] += 1

    n_decoy = round(args.decoy_ratio * len(syn_pos))
    decoys = []
    while len(decoys) < n_decoy:
        res = make_decoy(rng, carriers, args)
        if res is None:
            break
        decoys.append(res)

    budget = round(args.neg_ratio * (len(real_pos) + len(syn_pos)))
    keep_neg = max(0, budget - len(decoys))
    sampled_neg = rng.sample(real_neg, min(keep_neg, len(real_neg)))

    train_items = real_pos + syn_pos + decoys + sampled_neg
    rng.shuffle(train_items)

    # ---- leakage guard on canonical twins -------------------------------------------------
    heldout_c = {canonical_fingerprint(w) for w, _ in test_items} | {canonical_fingerprint(w) for w, _ in val_items}
    before = len(train_items)
    train_items = [(w, m) for w, m in train_items
                   if m["source"] != "real" or canonical_fingerprint(w) not in heldout_c]
    n_train_twins = before - len(train_items)
    train_c = {canonical_fingerprint(w) for w, _ in train_items}
    test_c = {canonical_fingerprint(w) for w, _ in test_items}
    before_v = len(val_items)
    val_items = [(w, m) for w, m in val_items
                 if canonical_fingerprint(w) not in train_c and canonical_fingerprint(w) not in test_c]
    n_val_twins = before_v - len(val_items)
    print(f"synthetic positives {len(syn_pos)}, decoys {len(decoys)}, real negatives kept "
          f"{len(sampled_neg)}/{len(real_neg)} (after capping template twins at 8, from {len(real_neg_all)}); "
          f"train windows with a canonical twin in val/test dropped: {n_train_twins}; "
          f"val windows with a twin in train/test dropped: {n_val_twins}")

    # ---- write ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    tp, tm = write_split(out_dir, "train", train_items)
    vp, vm = write_split(out_dir, "val", val_items)
    cap_seen, capped_items = Counter(), []
    for w, m in test_items:
        c = canonical_fingerprint(w)
        if cap_seen[c] < args.test_cap_per_template:
            cap_seen[c] += 1
            capped_items.append((w, m))
    cp, cm = write_split(out_dir, "test_capped", capped_items)
    sp, sm = write_split(out_dir, "test", test_items)

    tr_win, tr_span = class_table(train_items)
    te_win, te_span = class_table(test_items)
    print(f"\n{'label':<24}{'train real':>11}{'train synth':>12}{'train spans':>12}{'test win':>10}{'test spans':>11}")
    report_classes = {}
    for lab in all_gliner_labels():
        r, s = tr_win[(lab, "real")], tr_win[(lab, "synthetic")]
        sp_tr = tr_span[(lab, "real")] + tr_span[(lab, "synthetic")]
        tw, ts = te_win[(lab, "real")], te_span[(lab, "real")]
        flag = "   <-- low train" if sp_tr < 50 and lab != "other personal information" else ""
        print(f"{lab:<24}{r:>11}{s:>12}{sp_tr:>12}{tw:>10}{ts:>11}{flag}")
        report_classes[lab] = {"train_windows_real": r, "train_windows_synthetic": s, "train_spans": sp_tr,
                               "test_windows": tw, "test_spans": ts, "policy": policy_of(lab)}
    print(f"\ntrain: {len(train_items)} windows ({sum(1 for w, _ in train_items if w['ner'])} with entities) -> {tp}")
    print(f"val  : {len(val_items)} windows ({sum(1 for w, _ in val_items if w['ner'])} with entities) -> {vp}")
    print(f"test_capped: {len(capped_items)} windows ({sum(1 for w, _ in capped_items if w['ner'])} with entities, "
          f"<= {args.test_cap_per_template} per template) -> {cp}")
    print(f"test : {len(test_items)} windows ({sum(1 for w, _ in test_items if w['ner'])} with entities) -> {sp}")

    report = {
        "args": vars(args),
        "policy": {"synthetic_only": sorted(cfg.SYNTHETIC_ONLY), "split_20_80": sorted(cfg.SPLIT_20_80),
                   "real_train_fraction": cfg.REAL_TRAIN_FRACTION},
        "counts": {"test_capped_windows": len(capped_items), "train_windows": len(train_items), "val_windows": len(val_items), "test_windows": len(test_items),
                   "val_windows_with_entities": sum(1 for w, _ in val_items if w["ner"]),
                   "test_windows_with_entities": sum(1 for w, _ in test_items if w["ner"]),
                   "train_windows_with_entities": sum(1 for w, _ in train_items if w["ner"]),
                   "test_exact_duplicate_negatives_removed": test_dups,
                   "val_exact_duplicate_negatives_removed": val_dups,
                   "val_dropped_as_canonical_twins": n_val_twins,
                   "synthetic_positive": len(syn_pos), "decoys": len(decoys),
                   "real_negative_kept": len(sampled_neg), "real_negative_available": len(real_neg),
                   "train_dropped_as_canonical_twins": n_train_twins,
                   "train_dropped_overlapping_spans": n_overlap},
        "classes": report_classes,
        "sha256": {p.name: sha256_file(p) for p in (tp, vp, sp, cp, tm, vm, sm, cm)},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"report -> {out_dir / 'build_report.json'}")


if __name__ == "__main__":
    main()
