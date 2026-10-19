"""
Injects synthetic fields into a REAL negative flow text, so synthetic examples sit
in real traffic context (uri / post_body / header pieces, exactly as the loader
builds them) instead of in a fixed template the model could memorise.

Entity character spans are tracked with private-use marker characters that are
inserted around each value while the text is assembled and stripped at the end,
so offsets are exact however many fields are injected.

Styles (chosen at random among those that fit the carrier):
  query       append &key=value to the existing "uri: ..." piece (value URL-encoded)
  json_merge  add "key":"value" into an existing JSON post_body object
  form_merge  append &key=value to an existing form-encoded post_body
  json_new    new "post_body: {...}" piece (sometimes nested: {"user":{...}})
  form_new    new "post_body: k=v" piece
  header      new "header_X-<Key>: value" piece
A field with label None is a decoy: injected the same way, no entity span.
"""
import json
import random
import re
from typing import List, Optional, Tuple
from urllib.parse import quote

from gliner_tokenizer import tokenize_with_offsets
from synth_vocab import KEYS, NEUTRAL_KEYS, NESTING_KEYS

_START = 0xE100          # entity i starts with chr(_START + i)
_END = "\ue001"
_MARKER_RE = re.compile("[\ue000-\ue1ff]")

Field = Tuple[Optional[str], str, str]     # (label or None, key, value)
Span = Tuple[int, int, str]                # (char_start, char_end_exclusive, label)


def pick_key(label: str, rng: random.Random) -> str:
    if KEYS.get(label) and rng.random() < 0.7:
        return rng.choice(KEYS[label])
    return rng.choice(NEUTRAL_KEYS)


def _resolve(text: str, labels: List[str]):
    out, n, cur, spans = [], 0, None, []
    for ch in text:
        o = ord(ch)
        if _START <= o < _START + 256:
            cur = (o - _START, n)
        elif ch == _END:
            idx, start = cur
            spans.append((start, n, labels[idx]))
            cur = None
        else:
            out.append(ch)
            n += 1
    return "".join(out), sorted(spans)


def inject(carrier: str, fields: List[Field], rng: random.Random):
    """Returns (text, spans) or None if the carrier can't be used."""
    if _MARKER_RE.search(carrier):
        return None
    pieces = carrier.split(" | ")
    uri_i = next((i for i, p in enumerate(pieces) if p.startswith("uri: ") and len(p) > 5), None)
    body_i = next((i for i, p in enumerate(pieces) if p.startswith("post_body: ")), None)
    body = pieces[body_i][len("post_body: "):] if body_i is not None else None

    styles = [("header", 1)]
    if uri_i is not None:
        styles.append(("query", 3))
    if body_i is not None:
        b = body.strip()
        if b.startswith("{") and b.endswith("}"):
            styles.append(("json_merge", 3))
        elif b and not b.startswith("[") and "=" in b:
            styles.append(("form_merge", 2))
    else:
        styles += [("json_new", 2), ("form_new", 1)]
    style = rng.choices([s for s, _ in styles], weights=[w for _, w in styles])[0]

    labels: List[str] = []

    def enc(label, value, mode):
        v = quote(value, safe="") if mode == "url" else json.dumps(value)[1:-1] if mode == "json" else value
        if label is None:
            return v
        labels.append(label)
        return chr(_START + len(labels) - 1) + v + _END

    def kv_url():
        return "&".join(f"{k}={enc(l, v, 'url')}" for l, k, v in fields)

    def json_obj():
        colon = ": " if rng.random() < 0.3 else ":"
        inner = ",".join(f'"{k}"{colon}"{enc(l, v, "json")}"' for l, k, v in fields)
        if rng.random() < 0.3:
            return f'"{rng.choice(NESTING_KEYS)}"{colon}{{{inner}}}'
        return inner

    def new_piece_pos():
        natural = (uri_i + 1) if uri_i is not None else 0
        return natural if rng.random() < 0.7 else rng.randint(0, len(pieces))

    if style == "query":
        p = pieces[uri_i]
        sep = "" if p.endswith(("?", "&")) else ("&" if "?" in p else "?")
        pieces[uri_i] = p + sep + kv_url()
    elif style == "json_merge":
        rest = body.strip()[1:].lstrip()
        inner = json_obj()
        pieces[body_i] = "post_body: {" + inner + ("}" if rest == "}" else "," + rest)
    elif style == "form_merge":
        pieces[body_i] = "post_body: " + body.rstrip().rstrip("&") + "&" + kv_url()
    elif style == "json_new":
        pieces.insert(new_piece_pos(), "post_body: {" + json_obj() + "}")
    elif style == "form_new":
        pieces.insert(new_piece_pos(), "post_body: " + kv_url())
    else:  # header
        for l, k, v in fields:
            name = "X-" + "-".join(w.capitalize() for w in re.split(r"[_.\-]", k) if w)
            pieces.insert(rng.randint(len(pieces) // 2, len(pieces)), f"header_{name}: {enc(l, v, 'raw')}")

    text, spans = _resolve(" | ".join(pieces), labels)
    if len(spans) != len(labels):
        return None
    return text, spans


def spans_align_exactly(text: str, spans: List[Span], window: dict) -> bool:
    """True iff the window's token spans cover EXACTLY the injected char spans
    (same labels, same order; no span merged with a neighbouring character)."""
    toks = tokenize_with_offsets(text)
    if [t[0] for t in toks] != window["tokenized_text"] or len(window["ner"]) != len(spans):
        return False
    return all(
        lab == l2 and toks[s][1] == cs and toks[e][2] == ce
        for (cs, ce, lab), (s, e, l2) in zip(spans, window["ner"])
    )
