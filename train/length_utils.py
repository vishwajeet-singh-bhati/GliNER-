"""
length_utils.py - subword-aware length control for GLiNER training windows.

Why this exists: finetune_gliner.py caps windows at 256 *words*, but memory
scales with *subword* count. Network-traffic text (hashes, base64, URLs,
RECON_ markers) splits into many subwords per word, so a "256-word" window can
become thousands of subwords and OOM a 16 GB GPU even at batch size 1.

What this does, per example:
  1. Clip very long words (plain words to 32 chars, entity words to 64 chars).
  2. Count real subwords with the model's own tokenizer.
  3. If over budget, slide a window that never cuts through an entity span and
     keep the window containing the most entities (keeps rare-class examples).
  4. Drop an example only if no window can hold even one of its entities.
"""
from collections import Counter


def per_word_subwords(tokenizer, words):
    """Subword count for each word (same word-split scheme GLiNER feeds the encoder)."""
    if not words:
        return []
    try:
        enc = tokenizer(words, is_split_into_words=True,
                        add_special_tokens=False, verbose=False)
        counts = [0] * len(words)
        for w in enc.word_ids():          # raises on slow tokenizers -> fallback below
            if w is not None:
                counts[w] += 1
        return counts
    except Exception:
        return [len(tokenizer.tokenize(w)) for w in words]


def _spans(ex):
    n = len(ex.get("tokenized_text", []))
    out = []
    for sp in ex.get("ner") or []:
        if len(sp) >= 3 and 0 <= int(sp[0]) <= int(sp[1]) < n:
            out.append((int(sp[0]), int(sp[1]), str(sp[2])))
    return out


def fit_example(ex, tokenizer, budget, max_plain=32, max_entity=64):
    """Return (new_example_or_None, info). `budget` = max text subwords."""
    tokens = list(ex.get("tokenized_text", []))
    spans = _spans(ex)
    ent_idx = {i for s, e, _ in spans for i in range(s, e + 1)}

    clipped = False
    for i, t in enumerate(tokens):
        lim = max_entity if i in ent_idx else max_plain
        if len(t) > lim:
            tokens[i] = t[:lim]
            clipped = True

    counts = per_word_subwords(tokenizer, tokens)
    info = {"clipped": clipped, "trimmed": False, "dropped": False,
            "lost": [], "subwords": sum(counts)}

    if sum(counts) <= budget:
        return {**ex, "tokenized_text": tokens, "ner": [list(s) for s in spans]}, info

    n = len(tokens)
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)

    best = None                                   # ((score, -start), start, end)
    for a in range(n):
        if any(s < a <= e for s, e, _ in spans):  # start would cut an entity
            continue
        b = a
        while b < n and prefix[b + 1] - prefix[a] <= budget:
            b += 1
        changed = True
        while changed:                            # pull end back before a cut entity
            changed = False
            for s, e, _ in spans:
                if s < b <= e:
                    b = s
                    changed = True
        if b <= a:
            continue
        score = sum(1 for s, e, _ in spans if a <= s and e < b)
        key = (score, -a)
        if best is None or key > best[0]:
            best = (key, a, b)

    if best is None or (spans and best[0][0] == 0):
        info["dropped"] = True
        info["lost"] = [l for _, _, l in spans]
        return None, info

    _, a, b = best
    kept = [[s - a, e - a, l] for s, e, l in spans if a <= s and e < b]
    info["trimmed"] = True
    info["lost"] = [l for s, e, l in spans if not (a <= s and e < b)]
    info["subwords"] = prefix[b] - prefix[a]
    return {**ex, "tokenized_text": tokens[a:b], "ner": kept}, info


def fit_dataset(data, tokenizer, budget, max_plain=32, max_entity=64):
    out, stats = [], {"total": len(data), "kept": 0, "clipped": 0, "trimmed": 0,
                      "dropped": 0, "lost_by_label": Counter(), "max_subwords": 0}
    for ex in data:
        new_ex, info = fit_example(ex, tokenizer, budget, max_plain, max_entity)
        stats["clipped"] += info["clipped"]
        stats["trimmed"] += info["trimmed"]
        stats["lost_by_label"].update(info["lost"])
        if new_ex is None:
            stats["dropped"] += 1
            continue
        stats["kept"] += 1
        stats["max_subwords"] = max(stats["max_subwords"], info["subwords"])
        out.append(new_ex)
    return out, stats


def print_fit_report(name, st):
    print(f"[length-fit:{name}] total={st['total']} kept={st['kept']} "
          f"clipped={st['clipped']} trimmed={st['trimmed']} dropped={st['dropped']} "
          f"max_subwords_after={st['max_subwords']}", flush=True)
    if st["lost_by_label"]:
        lost = ", ".join(f"{k}:{v}" for k, v in st["lost_by_label"].most_common())
        print(f"[length-fit:{name}] entity spans lost by label -> {lost}", flush=True)