# gliner2-candle

Rust + Candle port of **GLiNER2** entity extraction (not the original GLiNER).

## Quick Start

1. Download a model (example):
```
hf download fastino/gliner2-large-v1 --local-dir ./model
```

2. Run inference:
```
cargo run --release -- \
  --model-dir ./model \
  --text "Apple was founded by Steve Jobs in Cupertino." \
  --entities "person,organization,location"
```

## Notes

- This project targets the `fastino/gliner2-*` model family.
- CPU performance improves significantly with `--release`.
- GPU/accelerated backends are supported by Candle, but not configured here by default.

## CLI

- `--model-dir` Path to a directory containing `config.json`, `encoder_config/config.json`, and `model.safetensors`.
- `--text` Input text.
- `--entities` Comma-separated list of entity types.
- `--threshold` Optional score threshold (default in code).
- `--list-weights N` Lists the first N weight keys (debug).

## Source Layout

- `src/model.rs` Model load, forward, predict
- `src/processor.rs` Tokenization and input building
- `src/span_rep.rs` Span representation
- `src/count_lstm.rs` Count-aware structure embedding
- `src/inference.rs` Post-processing and span extraction
- `src/main.rs` CLI

## License

MIT
