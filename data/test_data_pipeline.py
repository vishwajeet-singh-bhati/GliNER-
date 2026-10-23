#!/usr/bin/env python3
"""Stdlib-only regression tests for the data pipeline fixes.  Run:  python data/test_data_pipeline.py"""
import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gliner_tokenizer import char_span_to_token_span, tokenize_with_offsets  # noqa: E402
from prepare_finetune_data import build_char_entities, chunk_and_tokenize  # noqa: E402
import pii_shapes  # noqa: E402
from build_dataset import canonical_fingerprint, dedupe_exact  # noqa: E402


def luhn_ok(num):
    t = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            d -= 9 if d > 9 else 0
        t += d
    return t % 10 == 0


class TestTokenizer(unittest.TestCase):
    def test_strict_rejects_mid_token_span(self):
        toks = tokenize_with_offsets("xx12ab34cd rest")
        self.assertIsNone(char_span_to_token_span(toks, 2, 8))          # inside one token
        self.assertEqual(char_span_to_token_span(toks, 0, 10), (0, 0))  # whole token
        self.assertEqual(char_span_to_token_span(toks, 2, 8, strict=False), (0, 0))  # legacy


class TestMasking(unittest.TestCase):
    CASES = [
        "Content-Type: text/plain<RECON_AdvertisingId>",
        "goid=1489461607753RECON_ImeiNumber&x=1",
        "a%7E%7ERECON_FirstName%7E%7ERECON_LastName%7E%7Eb",
        "RECON_AndroidIdRECON_SerialNumber",
    ]

    def test_every_span_lands_on_token_boundaries(self):
        from pii_type_mapping import all_gliner_labels  # noqa: F401
        for raw in ("text/plainRECON_AdvertisingId", "goid=1489461607753RECON_Imei",
                    "%7E%7ERECON_FirstName%7E%7E", "RECON_AndroidIdRECON_SerialNumber x RECON_Email"):
            text, ents = build_char_entities(raw, True, seed_prefix="t")
            toks = tokenize_with_offsets(text)
            starts, ends = {t[1] for t in toks}, {t[2] for t in toks}
            for s, e, _ in ents:
                self.assertIn(s, starts, (raw, text))
                self.assertIn(e, ends, (raw, text))
            wins, dropped = chunk_and_tokenize(text, ents, 300, 30)
            self.assertEqual(dropped, 0)

    def test_deterministic_and_path_independent(self):
        a = build_char_entities("x RECON_AdvertisingId y", True, seed_prefix="raw_data/auto_anteater/b1/HTTP/a.json::k")
        b = build_char_entities("x RECON_AdvertisingId y", True, seed_prefix="raw_data/auto_anteater/b1/HTTP/a.json::k")
        self.assertEqual(a, b)


class TestShapes(unittest.TestCase):
    def test_shapes(self):
        rng = random.Random(1)
        v4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
        for _ in range(200):
            self.assertRegex(pii_shapes.uuid4(rng), v4)
            im = pii_shapes.imei(rng)
            self.assertEqual(len(im), 15)
            self.assertTrue(luhn_ok(im))
            self.assertEqual(len(pii_shapes.imsi(rng)), 15)
            ic = pii_shapes.iccid(rng)
            self.assertEqual(len(ic), 19)
            self.assertTrue(ic.startswith("89") and luhn_ok(ic))

    def test_location_is_key_aware(self):
        rng = random.Random(2)
        self.assertNotIn(",", pii_shapes.location(rng, "&lat="))
        self.assertNotIn(",", pii_shapes.location(rng, '"longitude":'))
        self.assertIn(",", pii_shapes.location(rng, "&ll="))


class TestLeakageGuards(unittest.TestCase):
    def _w(self, toks, ner):
        return {"tokenized_text": toks, "ner": ner}

    def test_canonical_twins_match_across_values(self):
        a = self._w(["id", "=", "3f1a2b4c-1111-4222-8333-444455556666", "x", "17"], [[2, 2, "advertiser id"]])
        b = self._w(["id", "=", "9a8b7c6d-5555-4666-9777-888899990000", "x", "23"], [[2, 2, "advertiser id"]])
        c = self._w(["id", "=", "9a8b7c6d-5555-4666-9777-888899990000", "x", "23"], [])
        self.assertEqual(canonical_fingerprint(a), canonical_fingerprint(b))
        self.assertNotEqual(canonical_fingerprint(a), canonical_fingerprint(c))

    def test_dedupe_keeps_positives_and_collapses_empty_duplicates(self):
        neg = self._w(["\x00"] * 5, [])
        pos = self._w(["a", "b"], [[0, 0, "city"]])
        items = [(neg, {}), (neg, {}), (neg, {}), (pos, {}), (pos, {})]
        kept, dropped = dedupe_exact(items)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(kept), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
