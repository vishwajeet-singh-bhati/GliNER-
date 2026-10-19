import argparse
import json
import shutil

def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=False)

def subword_lengths(tokenizer, windows, batch=128):
    lengths = []
    for i in range(0, len(windows), batch):
        chunk = [w["tokenized_text"] or ["."] for w in windows[i:i + batch]]
        enc = tokenizer(chunk, is_split_into_words=True, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths

def process(path, tokenizer, max_subwords, write):
    windows = [json.loads(line) for line in open(path, encoding="utf-8")]
    lengths = subword_lengths(tokenizer, windows)
    order = sorted(range(len(windows)), key=lambda i: -lengths[i])
    srt = sorted(lengths)
    pct = lambda q: srt[min(len(srt) - 1, int(q * len(srt)))]
    over = [i for i, n in enumerate(lengths) if n > max_subwords]
    
    print(f"\n=== {path} ({len(windows)} windows) ===")
    print(f"subwords per window: p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={srt[-1]}")
    for thr in (512, 768, 1024, 2048):
        print(f"  windows over {thr:>4} subwords: {sum(1 for n in lengths if n > thr)}")
    
    print("10 longest windows:")
    for i in order[:10]:
        words = windows[i]["tokenized_text"]
        longest = max(words, key=len) if words else ""
        labels = sorted({e[2] for e in windows[i]["ner"]})
        print(f"  line {i + 1:>6}: {lengths[i]:>5} subwords, {len(words):>4} words, "
              f"longest word {len(longest)} chars {longest[:40]!r}, labels={labels}")
    
    lost = {}
    for i in over:
        for lab in {e[2] for e in windows[i]["ner"]}:
            lost[lab] = lost.get(lab, 0) + 1
    
    print(f"over budget ({max_subwords}): {len(over)} windows; windows with entities that would be dropped: {lost or 'none'}")
    
    if write:
        keep = [w for w, n in zip(windows, lengths) if n <= max_subwords]
        shutil.copyfile(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            for w in keep:
                f.write(json.dumps(w) + "\n")
        print(f"wrote {len(keep)} windows (dropped {len(windows) - len(keep)}); original saved as {path}.bak")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-subwords", type=int, default=768)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    
    tok = load_tokenizer(args.tokenizer)
    for path in args.files:
        process(path, tok, args.max_subwords, args.write)
