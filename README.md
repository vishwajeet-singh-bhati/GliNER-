# AntShield PII-Type Extraction with GLiNER

Follow-on to the E8-quantized BERT binary PII classifier (see `llm-quantization`
project). That model answered "does this packet leak PII?" This project answers
"which PII type(s) does it leak?", using GLiNER (github.com/urchade/GLiNER) for
zero-shot / fine-tuned span-level NER, evaluated on the same AntShield data,
and eventually run on-device via the same ARM NEON C++ kernel approach.

## Phases (tagged as git milestones)

| Tag | Phase | Status |
|---|---|---|
| `v0-baseline-zeroshot` | Convert AntShield -> GLiNER format, run pretrained model, no fine-tuning | this scaffold |
| `v1-finetuned` | Fine-tune GLiNER encoder on AntShield train split | not started |
| `v2-quantized` | Export + quantize encoder for mobile (reuse NEON kernel work) | not started |
| `v3-mobile-bench` | On-device benchmark via wireless ADB, physical device | not started |

## Real dataset layout (confirmed by direct inspection, not assumed)

```
antshield_public_dataset/
  exposures.csv        # aggregate counts per (pii type, package, dest) -- not per-packet, not used
  apps_info.csv         # app metadata, not used
  raw_data/
    auto_anteater/batch1..6/HTTP|HTTPS/<package_name>.json
    manual_anteater/...

47,076 total packets, 8,579 positive (label=1), across 711 JSON files.
```

Each JSON file is a dict keyed by `"<ts>,<uuid>"` -> packet, with fields
including `post_body`, `headers` (dict), `uri`, `user_agent`, `referrer`,
`label` (0/1), `pii_types` (list or null), `package_name`.

pii_types found in the real data (counts): AdvertiserId(6217) Location(3380)
AndroidId(3314) IMEI(424) City(348) Zipcode(313) Username(227)
SerialNumber(116) FirstName(101) IMSI(36) Email(31) ICCID(5) Gender(4)
LastName(4) Password(2) PhoneNumber(2) Other(1)

## Important finding: the dataset is redacted with self-labeling tokens

Every real PII value in this public release has been replaced with a literal
token of the form `RECON_<Type>` (e.g. `RECON_AndroidId`, `RECON_Email`),
confirmed present in `uri`, `post_body`, `headers`, `user_agent`, or
`referrer` for **100% (8579/8579)** of positive packets.

This means the token *spells out its own label*. A model can solve "what
type is this" (or even "is this PII at all", via the bare `RECON_` prefix)
by substring-matching the token, without learning anything about what real
PII looks like. This affects your earlier binary BERT result too, since
`RECON_` is present in 100% of positives and 0% of negatives in this same
release -- worth a quick sanity check on what that model actually learned to
key off of.

`data/convert_to_gliner_format.py` supports `--mask-recon-tokens`, which
replaces each token with a synthetic realistic-looking value of the correct
shape (`data/recon_synthesizer.py`) before spans are finalized. **Use this
flag** unless you specifically want a literal-replay comparison against the
(likely similarly leaky) BERT baseline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run baseline (Phase 0)

```bash
python data/convert_to_gliner_format.py \
    --dataset-root /path/to/antshield_public_dataset \
    --output data/test_gliner.jsonl \
    --mask-recon-tokens

python eval/zero_shot_eval.py \
    --data data/test_gliner.jsonl \
    --model urchade/gliner_multi_pii-v1
```

## Data cleaning (confirmed, data-driven)

Out of ~400 distinct header keys in the real dataset, only 31 ever carry a
RECON_ token (see `data/analyze_pii_bearing_fields.py`, output committed at
`configs/productive_header_keys.json`). `antshield_loader.py` now only
includes those 31 header keys plus the always-productive top-level fields
(`uri`, `post_body`, `user_agent`, `referrer`). This alone drops the median
record from bloated boilerplate down to a fraction of GLiNER's 384-token
limit -- but see the next section, word count is not token count here.

Regenerate the whitelist if you get a newer dataset dump:
```bash
python data/analyze_pii_bearing_fields.py --dataset-root /path/to/antshield_public_dataset
```

## Truncation: the real cause and the fix

Naive word-count is misleading for this data: `post_body` fields are dense
JSON, and GLiNER's tokenizer treats every `{`, `"`, `:`, `,` as its own
token. A packet with 15 words can tokenize to 150+ tokens. Measured on the
real dataset (after header cleaning): median 74 tokens, p90 314, but **5.5%
of records exceed GLiNER's 384-token limit** -- including ~1,050
crash-report/stack-trace-style packets that run 2,000-16,000+ tokens.

Two separate fixes, for the two separate places truncation happens:

- **Eval** (`eval/chunking.py`): character-based sliding window over
  `predict_entities`, merges overlapping predictions back to global offsets.
  Used automatically by `eval/zero_shot_eval.py`.
- **Fine-tuning data** (`data/prepare_finetune_data.py`): token-based
  windowing using GLiNER's own tokenization regex (verified from source, not
  guessed), so char-span entities convert to token-span entities correctly.
  Caps windows-per-record at 3 (prioritizing windows that contain an entity)
  so the ~1,050 stack-trace outliers don't balloon into 50+ near-duplicate
  windows each and dominate the training set.

## Phase 1: fine-tuning

```bash
python data/prepare_finetune_data.py \
    --dataset-root /path/to/antshield_public_dataset \
    --output-dir data \
    --mask-recon-tokens
# writes data/train_finetune.jsonl, val_finetune.jsonl, test_finetune.jsonl
# (token-indexed format -- NOT the same schema as test_gliner.jsonl above,
#  which is char-offset and used only by zero_shot_eval.py)

pip install "gliner[training]"

python train/finetune_gliner.py \
    --data-dir data \
    --base-model urchade/gliner_multi_pii-v1 \
    --output-dir models/antshield-gliner-v1
```

To evaluate any checkpoint (baseline or fine-tuned) on the held-out test
split, generate the char-offset eval file with `--split test` (uses the same
hash split as `*_finetune.jsonl`, so no train/test leakage):
```bash
python data/convert_to_gliner_format.py \
    --dataset-root /path/to/antshield_public_dataset \
    --output data/test_gliner.jsonl --mask-recon-tokens --split test

python eval/zero_shot_eval.py --data data/test_gliner.jsonl --model models/antshield-gliner-v1
```
(`--model` accepts a local path -- GLiNER's `from_pretrained` handles both.)

## Incorporated from the reference BERT preprocessing script

Cross-checked `preprocess_antshield.py` (the script used for the binary BERT
model) against the real dataset. Three real fixes came out of this:

1. **Split now matches the BERT model's test set exactly, by default.**
   `--split-strategy campaign` (now the default in both converters):
   `auto_anteater` -> train/val, `manual_anteater` -> test. Verified against
   the BERT script's own reported counts: test = {0: 21370, 1: 3819} (manual_anteater
   exactly), train+val = {0: 17127, 1: 4760} (auto_anteater exactly). This is
   what makes a GLiNER-vs-BERT comparison meaningful -- my earlier hash-based
   split was scoring GLiNER against a different test set than BERT was scored on.

   Known caveat, carried over verbatim because it applies identically here
   (documented in `splitting.py`): this is a split by collection campaign,
   not by app -- 73 package names appear in both campaigns (~71.6% of test
   packages also in train), so test metrics are likely somewhat optimistic
   relative to a genuinely unseen-app scenario. Use `--split-strategy hash`
   for a non-campaign-correlated split if that matters for your purposes.

2. **RECON_ token matching now absorbs trailing suffixes correctly.**
   ~5,300 redaction tokens have junk after the type name (`RECON_Location841329`,
   base64-ish device strings). My regex was previously stopping right after the
   type name and leaving that suffix glued onto the synthetic replacement value
   in masked mode (verified and fixed -- 0 residual `RECON_` substrings, 0
   glued-garbage entities after the fix, checked across the full test split).
   `data/pii_type_mapping.py`'s `build_recon_pattern()` is now the single
   source of truth for this regex, used by both converters.

3. **package_name normalization + optional metadata fields.**
   `"No mapping_ entry not found"` / `"System"` -> `unknown_package`,
   matching the BERT script's bucketing. Available via
   `--include-metadata-fields` (off by default -- these fields never carry a
   RECON_ token, so they're pure context for GLiNER's span task, not
   something worth spending token budget on by default; turn on for closer
   feature-parity with the BERT experiment).

One finding surfaced but NOT fixed here since it's in the other repo: the
reference script's `RECON_[A-Za-z0-9]*` regex has no `_` in its character
class, so on the ~57 records with concatenated tokens
(`RECON_SerialNumberRECON_AndroidId`) it merges into the second token's
`RECON_` prefix and leaves a bare `_AndroidId` in the BERT training text --
a residual version of the exact leakage problem that script's own docstring
warns about, just for the binary task instead of the type task.

Also confirmed, not changed: the BERT script's 2000-char post_body truncation
drops ~476 RECON_ tokens across train+test (213 auto, 263 manual) as false
negatives. This pipeline's sliding-window chunking (see above) doesn't have
that failure mode.
