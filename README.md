# gliner2-candle

Rust + Candle port of **GLiNER2** entity and relation extraction (not the original GLiNER).

## Quick Start

Run inference — the model downloads automatically on first use (no manual
`hf download` needed):

```
cargo run --release -- \
  --model-id fastino/gliner2.5-small-v1 \
  --text "Apple was founded by Steve Jobs in Cupertino." \
  --entities "person,organization,location"
```

`--model-id` fetches `config.json`, `tokenizer.json`,
`encoder_config/config.json`, and `model.safetensors` into a model-specific
directory (`./models/gliner2.5-small-v1/`) and reuses the verified cache on
later runs. Alternatively point `--model-dir` at a directory with those four
files (it takes precedence and never downloads).

## CLI

Requests are entity-only, relation-only, or combined (relations need a
boundary checkpoint such as `fastino/gliner2.5-small-v1`; span checkpoints
like `fastino/gliner2-large-v1` fail clearly).

```
# Entity-only
gliner2-candle --model-id fastino/gliner2.5-small-v1 \
  --text "Apple was founded by Steve Jobs in Cupertino." \
  --entities "person,organization,location"

# Relation-only (directed head -> tail)
gliner2-candle --model-id fastino/gliner2.5-small-v1 \
  --text "Steve Jobs founded Apple." --relations "founder"

# Combined entities + relations
gliner2-candle --model-id fastino/gliner2.5-small-v1 \
  --text "Steve Jobs founded Apple in Cupertino." \
  --entities "person,organization,location" --relations "founder"

# Machine-readable JSON (stable field names/ordering, agents can diff output)
gliner2-candle --model-id fastino/gliner2.5-small-v1 --json \
  --text "Steve Jobs founded Apple." --entities "person,organization"

# Agent verification case file (per-case pass/fail, nonzero exit on failure)
gliner2-candle --model-id fastino/gliner2.5-small-v1 --cases cases/cli-cases.json
```

Flags:

- `--model-dir` Local model directory (`config.json`, `model.safetensors`,
  `tokenizer.json`, `encoder_config/config.json`). Takes precedence over
  `--model-id`; never triggers a download.
- `--model-id` HuggingFace model ID to download/reuse
  (default `fastino/gliner2-large-v1`).
- `--revision` Hub revision (commit sha, tag, or branch) for `--model-id`.
  Defaults to the pinned revisions documented in `DEVELOP.md`
  (`fastino/gliner2.5-small-v1` → `7132dc4561c3…`, `fastino/gliner2-large-v1`
  → `5312584a6fd5…`).
- `--models-root` Where model cache directories live (default `./models`);
  each model ID maps to `<models-root>/<repo-name>/`.
- `--text` Input text.
- `--entities` Comma-separated entity types.
- `--relations` Comma-separated relation types.
- `--threshold` Score threshold in `[0.0, 1.0]` (default `0.5`).
- `--cases` Versioned JSON case file (see below).
- `--json` Machine-readable JSON on stdout (one-off results or case report).
- `--list-weights [N]` Lists weight keys with dtype and shape (debug).
- `--device` `cpu` (default) or `cuda:N`.

Readable output prints the detected architecture and active tasks, then every
entity (type, confidence, exact text, half-open byte offsets) and every
relation (type, confidence, directed head/tail text and offsets), or an
explicit "No … found above threshold" line. All offsets are half-open UTF-8
**byte** offsets into `--text`.

## Case files (agent verification)

`--cases FILE` runs each case sequentially through the same production
inference path as one-off runs and reports per-case PASS/FAIL with the exact
mismatches; the exit status is nonzero if any case fails. The format is
versioned JSON (`"version": 1`) — see `cases/cli-cases.json` for a committed
example covering positive, negative, and near-threshold (ambiguous) cases:

```json
{
  "version": 1,
  "model_id": "fastino/gliner2.5-small-v1",
  "cases": [
    {
      "name": "readme-entities",
      "text": "Apple was founded by Steve Jobs in Cupertino.",
      "entities": ["person", "organization", "location"],
      "threshold": 0.5,
      "expect_entities": [
        {"type": "person", "text": "Steve Jobs", "byte_start": 21, "byte_end": 31,
         "confidence": {"min": 0.9, "max": 1.0}}
      ]
    }
  ]
}
```

Each case names its entity/relation types and (optionally) its own threshold;
expectations give the label/type, exact surface, and exact byte offsets.
Optional `confidence: {min, max}` bounds catch large numerical drift without
bitwise equality. Matching is strict set equality — a missing extraction, an
unexpected extra extraction, or a confidence outside its bounds fails the
case.

## Notes

- This project targets the `fastino/gliner2-*` model family
  (`fastino/gliner2.5-*` boundary checkpoints extract entities *and*
  relations; `fastino/gliner2-*` span checkpoints are entity-only).
- CPU performance improves significantly with `--release`.
- GPU/accelerated backends are supported by Candle, but not configured here by default.

## Source Layout

- `src/model.rs` Architecture dispatch (`Model::{Span,Boundary}`), load, predict
- `src/boundary*.rs` Boundary architecture (preprocess, encode, pool, decode, relations)
- `src/processor.rs` Span-path tokenization and input building
- `src/span_rep.rs` Span representation (span path)
- `src/count_lstm.rs` Count-aware structure embedding (span path)
- `src/inference.rs` Span-path post-processing and span extraction
- `src/hub.rs` `--model-id` download & cache contract
- `src/cases.rs` Case-file schema, validation, matching
- `src/main.rs` CLI

## License

MIT
