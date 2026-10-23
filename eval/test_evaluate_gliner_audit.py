#!/usr/bin/env python3
"""
Unit tests for eval/evaluate_gliner_audit.py
Covers:
1. Exact 1-to-1 matching (partial overlaps & label mismatches rejected)
2. Duplicate gold and predicted spans
3. Zero-support labels (all 17 labels preserved; macro_f1_all vs macro_f1_supported)
4. Invalid, out-of-range, and overlapping gold spans
5. Subword-aware chunk boundaries and cross-chunk prediction deduplication
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_gliner_audit as audit


class TestEvaluateGlinerAudit(unittest.TestCase):
    def setUp(self):
        self.labels = audit.get_configured_labels()
        self.allowed_set = set(self.labels)

    def test_17_labels_present(self):
        self.assertEqual(len(self.labels), 17)
        self.assertIn("other personal information", self.labels)
        self.assertIn("advertiser id", self.labels)

    def test_exact_one_to_one_matching(self):
        gold = [(0, 1, "first name"), (3, 3, "city"), (5, 6, "email address")]
        preds = [
            (0, 1, "first name"),     # Exact match -> TP for first name
            (3, 4, "city"),           # Boundary mismatch -> FP + FN for city
            (5, 6, "username"),       # Label mismatch -> FP for username, FN for email address
        ]
        stats = audit.match_spans_one_to_one(gold, preds, self.labels)
        self.assertEqual(stats["first name"]["tp"], 1)
        self.assertEqual(stats["first name"]["fp"], 0)
        self.assertEqual(stats["first name"]["fn"], 0)

        self.assertEqual(stats["city"]["tp"], 0)
        self.assertEqual(stats["city"]["fp"], 1)
        self.assertEqual(stats["city"]["fn"], 1)

        self.assertEqual(stats["email address"]["tp"], 0)
        self.assertEqual(stats["email address"]["fn"], 1)
        self.assertEqual(stats["username"]["fp"], 1)

    def test_duplicate_spans_do_not_inflate_counts(self):
        raw_record = {
            "tokenized_text": ["GET", "/api", "user", "john", "PASS"],
            "ner": [
                [3, 3, "first name"],
                [3, 3, "first name"],  # duplicate gold annotation
            ],
        }
        summary = audit.ValidationSummary()
        res = audit.validate_and_parse_example(raw_record, 0, self.allowed_set, summary)
        self.assertIsNotNone(res)
        _, dedup_gold, overlaps = res
        self.assertFalse(overlaps)
        self.assertEqual(len(dedup_gold), 1)
        self.assertEqual(summary.duplicate_gold_spans, 1)

        # Duplicate predictions across overlapping chunks
        raw_preds = [
            (3, 3, "first name", 0.72),
            (3, 3, "first name", 0.91),
        ]
        dedup_preds = audit.deduplicate_predictions(raw_preds)
        self.assertEqual(len(dedup_preds), 1)
        self.assertAlmostEqual(dedup_preds[0][3], 0.91)

        stats = audit.match_spans_one_to_one(dedup_gold, [(s, e, l) for s, e, l, _ in dedup_preds], self.labels)
        self.assertEqual(stats["first name"]["tp"], 1)
        self.assertEqual(stats["first name"]["fp"], 0)
        self.assertEqual(stats["first name"]["support"], 1)

    def test_zero_support_labels_and_macro_f1(self):
        # Only 1 supported class ("password") with perfect F1 = 1.0; 16 classes have 0 support
        gold = [(1, 1, "password")]
        preds = [(1, 1, "password")]
        stats = audit.match_spans_one_to_one(gold, preds, self.labels)
        table = audit.compute_metrics_table(stats, self.labels)

        self.assertEqual(len(table["per_class"]), 17)
        self.assertEqual(table["per_class"]["other personal information"]["support"], 0)
        self.assertEqual(table["per_class"]["other personal information"]["f1"], 0.0)
        self.assertFalse(table["per_class"]["other personal information"]["has_support"])

        self.assertAlmostEqual(table["micro"]["f1"], 1.0)
        self.assertAlmostEqual(table["macro_f1_supported_only"], 1.0)
        self.assertAlmostEqual(table["macro_f1_all_configured"], round(1.0 / 17.0, 6))
        self.assertEqual(table["num_supported_labels"], 1)
        self.assertEqual(table["num_configured_labels"], 17)

    def test_invalid_and_overlapping_gold_spans_reported(self):
        raw_record = {
            "tokenized_text": ["a", "b", "c", "d"],
            "ner": [
                [-1, 1, "city"],            # invalid: negative start
                [2, 1, "city"],             # invalid: end < start
                [0, 10, "city"],            # invalid: out of range
                [0, 0, "not_a_pii_label"],  # invalid: unknown label
                [1, 2, "first name"],       # valid, overlaps with [2, 3, "last name"] at token 2
                [2, 3, "last name"],        # valid, overlaps at token 2
            ],
        }
        summary = audit.ValidationSummary()
        res = audit.validate_and_parse_example(raw_record, 42, self.allowed_set, summary)
        self.assertIsNotNone(res)
        _, dedup_gold, has_overlap = res
        self.assertEqual(summary.invalid_gold_spans, 4)
        self.assertEqual(summary.unknown_label_spans, 1)
        self.assertEqual(len(dedup_gold), 2)
        self.assertTrue(has_overlap)
        self.assertEqual(summary.examples_with_overlapping_gold, 1)
        self.assertEqual(summary.overlapping_gold_span_count, 2)
        self.assertIn(42, summary.invalid_example_indices)
        self.assertIn(42, summary.overlapping_example_indices)

    def test_chunk_boundary_coverage_and_giant_token(self):
        # Fake subword counter: 1 subword per 4 chars
        counter_fn = lambda w: max(1, (len(w) + 3) // 4)
        chunker = audit.SubwordChunker(counter_fn, labels=self.labels, max_len=120, safety_margin=4, overlap_subwords=12)

        # Create 60 tokens plus 1 giant 500-char token in the middle
        tokens = [f"tok{i}" for i in range(25)] + ["X" * 500] + [f"tok{i}" for i in range(26, 60)]
        chunks, _ = chunker.chunk_tokens(tokens)

        self.assertGreater(len(chunks), 2)
        # Verify 100% token index coverage
        covered = set()
        for ch in chunks:
            for t_idx in range(ch.start_tok, ch.end_tok + 1):
                covered.add(t_idx)
        self.assertEqual(covered, set(range(len(tokens))))
        # Verify the giant token at index 25 was isolated and flagged
        giant_chunks = [ch for ch in chunks if ch.exceeded_single_token_budget]
        self.assertEqual(len(giant_chunks), 1)
        self.assertEqual(giant_chunks[0].start_tok, 25)
        self.assertEqual(giant_chunks[0].end_tok, 25)

    # ---- tests for the fixes -------------------------------------------------------------
    def test_word_chunker_single_pass_for_300_word_window(self):
        counter_fn = lambda w: 3          # pessimistic: 3 subwords per word -> 900 subwords
        ch = audit.WordChunker(counter_fn, labels=self.labels, max_len=384)
        chunks, _ = ch.chunk_tokens([f"w{i}" for i in range(300)])
        self.assertEqual(len(chunks), 1)  # the legacy subword chunker would split this
        legacy = audit.SubwordChunker(counter_fn, labels=self.labels, max_len=384)
        self.assertGreater(len(legacy.chunk_tokens([f"w{i}" for i in range(300)])[0]), 1)

    def test_word_chunker_covers_long_windows(self):
        ch = audit.WordChunker(lambda w: 1, labels=self.labels, max_len=100, overlap_words=10)
        toks = [f"w{i}" for i in range(450)]
        chunks, _ = ch.chunk_tokens(toks)
        covered = set()
        for c in chunks:
            self.assertLessEqual(c.end_tok - c.start_tok + 1, 100)
            covered.update(range(c.start_tok, c.end_tok + 1))
        self.assertEqual(covered, set(range(450)))

    def test_error_taxonomy(self):
        gold = {(0, 1, "first name"), (5, 5, "city"), (9, 9, "zip code"), (12, 12, "gender")}
        pred = {(0, 1, "last name"),      # same span, wrong label  -> fp+fn label confusion
                (4, 5, "city"),           # boundary, same label    -> fp boundary + fn boundary
                (20, 20, "password")}     # spurious in positive window; zip/gender missed
        errs, _ = audit.classify_errors(gold, pred)
        self.assertEqual(errs["fp_label_confusion"], 1)
        self.assertEqual(errs["fn_label_confusion"], 1)
        self.assertEqual(errs["fp_boundary_same_label"], 1)
        self.assertEqual(errs["fn_boundary"], 1)
        self.assertEqual(errs["fp_spurious_in_positive_window"], 1)
        self.assertEqual(errs["fn_missed"], 2)
        errs2, by_label = audit.classify_errors(set(), {(3, 3, "advertiser id")})
        self.assertEqual(errs2["fp_in_entity_free_window"], 1)
        self.assertEqual(by_label["advertiser id"], 1)

    def test_score_store_threshold_filtering(self):
        labels = ["city", "gender"]
        store = [
            ([(0, 0, "city")], [(0, 0, "city", 0.9), (4, 4, "gender", 0.4)], False),
            ([], [(2, 2, "city", 0.6)], False),
        ]
        full, _c, errs, ef, n_ef, n_ef_pred = audit.score_store(store, labels, 0.5)
        m = audit.compute_metrics_table(full, labels)["micro"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))   # 0.4 dropped, 0.6 kept
        self.assertEqual((n_ef, n_ef_pred), (1, 1))
        full, *_ = audit.score_store(store, labels, 0.7)
        m = audit.compute_metrics_table(full, labels)["micro"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))
        full, *_ = audit.score_store(store, labels, 0.3)
        m = audit.compute_metrics_table(full, labels)["micro"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 2, 0))

    def test_resolve_labels_prefers_train_labels(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "train_labels.json").write_text(json.dumps(["city", "gender"]), encoding="utf-8")
            (Path(d) / "checkpoint-5").mkdir()
            labels, src = audit.resolve_labels(str(Path(d) / "checkpoint-5"), None)
            self.assertEqual(labels, ["city", "gender"])
            self.assertIn("train_labels.json", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
