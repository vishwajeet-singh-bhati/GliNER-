"""
long_filter.py - drop training windows that are too long in SUBWORDS.
Nothing is clipped or trimmed: a window is either kept exactly as-is or removed.
"""
from collections import Counter


def count_subwords(tokenizer, words):
    if not words:
        return 0
    try:
        enc = tokenizer(words, is_split_into_words=True,
                        add_special_tokens=False, verbose=False)
        return len(enc["input_ids"])
    except Exception:
        return sum(len(tokenizer.tokenize(w)) for w in words)


def drop_long_examples(data, tokenizer, max_subwords):
    counts = [count_subwords(tokenizer, ex.get("tokenized_text", [])) for ex in data]
    over = {t: sum(c > t for c in counts) for t in (512, 768, 1024, 2048)}
    kept, lost = [], Counter()
    for ex, c in zip(data, counts):
        if c > max_subwords:
            lost.update(str(s[2]) for s in (ex.get("ner") or []) if len(s) >= 3)
        else:
            kept.append(ex)
    print(f"[long-filter] subword counts over 512/768/1024/2048: "
          f"{over[512]}/{over[768]}/{over[1024]}/{over[2048]} of {len(data)}", flush=True)
    print(f"[long-filter] max_subwords={max_subwords}: kept {len(kept)}, "
          f"dropped {len(data) - len(kept)}", flush=True)
    if lost:
        print("[long-filter] entity spans in dropped windows -> "
              + ", ".join(f"{k}:{v}" for k, v in lost.most_common()), flush=True)
    return kept