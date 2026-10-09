"""
check_seq_lengths.py  -  how long are your AntShield training samples in SUBWORD tokens?

GLiNER's max_len counts WORDS, but DeBERTa attention cost grows with the square of SUBWORD tokens.
Traffic payloads (hex, base64, URLs, JWTs) shatter into many subwords per word, so a "384-word" sample
can be thousands of subword tokens. That would explain the 3.88 GiB single allocations, the 58 GiB
"allocated" on a 16 GiB card, and 4 s -> 120 s per step.

Usage (PowerShell, from the project folder, venv active):
  python check_seq_lengths.py --data-dir data
  python check_seq_lengths.py --file data\\train.json --tokenizer microsoft/mdeberta-v3-base

Accepts GLiNER-style json / jsonl:  {"tokenized_text": [...words...], "ner": [...]}  or  {"text": "..."}.
"""
import argparse, glob, json, os, re, statistics


def load_samples(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            raw = f.read().strip()
        items = []
        if raw.startswith("["):
            items = json.loads(raw)
        else:
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    try: items.append(json.loads(line))
                    except Exception: pass
        for it in items:
            if isinstance(it, dict):
                if "tokenized_text" in it: yield it["tokenized_text"]
                elif "text" in it: yield re.findall(r"\w+(?:[-_]\w+)*|\S", it["text"])


class DummyTok:   # only for offline self-test: ~4 characters per subword
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(max(1, (len(text) + 3) // 4)))}


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir"); ap.add_argument("--file")
    ap.add_argument("--tokenizer", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()

    paths = [a.file] if a.file else sorted(glob.glob(os.path.join(a.data_dir, "**", "*.json*"), recursive=True))
    if not paths: raise SystemExit("no .json/.jsonl files found")
    if a.dummy: tok = DummyTok()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)

    words_per, sub_per, worst_words = [], [], []
    for i, words in enumerate(load_samples(paths)):
        if i >= a.limit: break
        n_sub = 0
        for w in words:
            k = len(tok(w, add_special_tokens=False)["input_ids"])
            n_sub += k
            if k >= 40: worst_words.append((k, w[:60]))
        words_per.append(len(words)); sub_per.append(n_sub)

    print(f"files: {len(paths)}   samples analysed: {len(sub_per)}")
    print(f"words / sample      p50={pct(words_per,.5)}  p90={pct(words_per,.9)}  p99={pct(words_per,.99)}  max={max(words_per)}")
    print(f"subwords / sample   p50={pct(sub_per,.5)}  p90={pct(sub_per,.9)}  p99={pct(sub_per,.99)}  max={max(sub_per)}")
    print(f"subwords per word   mean={sum(sub_per)/max(1,sum(words_per)):.2f}   (natural prose is ~1.2-1.5)")
    for t in (512, 1024, 2048, 4096):
        print(f"samples with > {t:4d} subwords: {100*sum(s > t for s in sub_per)/len(sub_per):5.1f}%")
    est = lambda n: 12 * (68 * n * 768 + 32 * 12 * n * n) / 2**30
    print(f"\nrough activation memory for ONE sample (12-layer base, fp32, no checkpointing): "
          f"n=512 -> {est(512):.1f} GB   n=1024 -> {est(1024):.1f} GB   n=2048 -> {est(2048):.1f} GB   n=4096 -> {est(4096):.1f} GB")
    worst_words.sort(reverse=True)
    if worst_words:
        print("\nlongest single 'words' (subwords, text):")
        for k, w in worst_words[:8]: print(f"  {k:5d}  {w!r}")
    print("\nIf p99 or max is far above ~512, that is your OOM + slowness: cap it (see the recipe) before anything else.")


if __name__ == "__main__":
    main()
