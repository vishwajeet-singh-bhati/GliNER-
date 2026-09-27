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
