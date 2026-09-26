# AntShield PII-Type Extraction with GLiNER

Follow-on to the E8-quantized BERT binary PII classifier (see `llm-quantization`
project). That model answered "does this packet leak PII?" This project answers
"which PII type(s) does it leak?", using GLiNER (github.com/urchade/GLiNER) for
zero-shot / fine-tuned span-level NER, evaluated on the same AntShield split,
and eventually run on-device via the same ARM NEON C++ kernel approach.

## Phases (tagged as git milestones)

| Tag | Phase | Status |
|---|---|---|
| `v0-baseline-zeroshot` | Convert AntShield -> GLiNER format, run pretrained model, no fine-tuning | this scaffold |
| `v1-finetuned` | Fine-tune GLiNER encoder on AntShield train split | not started |
| `v2-quantized` | Export + quantize encoder for mobile (reuse NEON kernel work) | not started |
| `v3-mobile-bench` | On-device benchmark via wireless ADB, physical device | not started |

## Why this isn't a drop-in reuse of the BERT C++ kernel

The binary classifier was: encoder -> [CLS] -> linear head. GLiNER additionally
needs label-prompt encoding and a span-scoring head (bilinear/dot-product
between span representations and label embeddings). The NEON kernel work you
already have covers the *encoder* forward pass; the span-scoring head is new
and gets implemented in `mobile/` once `v2` starts.

## Data assumption

AntShield packets look like:

```json
{
  "post_body": "appflag=...&androidid=8f2e1...",
  "headers": {...},
  "label": 1,
  "pii_types": ["AndroidId"]
}
```

`pii_types` is treated as ground truth for span type; span *offsets* are
recovered by string-matching the known PII value against the extracted text
(same approach AntShield itself uses for predefined PII, see
`data/antshield_loader.py`).

**Action needed from you**: drop a small sample (5-10 records) of your actual
downloaded AntShield JSON file into `data/sample_raw.json` — the loader below
is written against the schema documented in the AntShield paper/site, but if
your local copy differs (field names, nesting) I'll adjust the parser to match
exactly rather than guessing further.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run baseline (Phase 0)

```bash
python data/convert_to_gliner_format.py \
    --input /path/to/antshield_test_split.json \
    --output data/test_gliner.jsonl

python eval/zero_shot_eval.py \
    --data data/test_gliner.jsonl \
    --model urchade/gliner_multi_pii-v1
```
