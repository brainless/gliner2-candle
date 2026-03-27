# gliner2-candle — Developer Notes

GLiNER2 entity extraction in pure Rust using [Candle](https://github.com/huggingface/candle).
Targets the **fastino/gliner2-*** model family (not the original GLiNER / urchade models).

## Quick start

```bash
# Download model (one-time)
hf download fastino/gliner2-large-v1 --local-dir ./model

# Inspect weight keys (verify prefixes after model updates)
cargo run -- --model-dir ./model --list-weights 50

# Run inference
cargo run --release -- \
  --model-dir ./model \
  --text "Apple was founded by Steve Jobs in Cupertino." \
  --entities "person,organization,location"
```

## Architecture

```
input_ids (schema tokens + [SEP_TEXT] + word subtokens)
    │
    ▼
DeBERTa V2 encoder          candle_transformers::models::debertav2
    │
    ├─► word embeddings      index_select at word_token_indices (first-subtoken pooling)
    │
    ├─► entity embeddings    index_select at [P] token positions
    │
    ├─► [SEP_TEXT] embedding → count_pred MLP → predicted extraction count
    │
    ├─► CountLSTM            pos_embedding + GRU(h0=entity_embs) + projector MLP
    │                        output: (gold_count, n_types, hidden)
    │
    ├─► SpanRepLayer         project_start + project_end MLPs → out_project MLP
    │                        output: (text_len, max_width, hidden)
    │
    └─► Scoring              matmul replacing einsum('lkd,bpd->bplk')
                             → sigmoid → threshold → greedy overlap removal
```

## Source layout

| File | Purpose |
|---|---|
| `src/config.rs` | `Gliner2Config` (config.json) + `EncoderConfig` loader |
| `src/model.rs` | `GLiNER2::load()` + `forward()` + `predict()` |
| `src/span_rep.rs` | `SpanRepLayer`: three 2-layer MLPs (project_start, project_end, out_project) |
| `src/count_lstm.rs` | `CountLSTM`: positional embedding + GRU + projector MLP |
| `src/processor.rs` | Schema → `input_ids` for entity extraction task only |
| `src/inference.rs` | Sigmoid, threshold, greedy overlap removal, char-offset mapping |
| `src/main.rs` | CLI (`--model-dir`, `--text`, `--entities`, `--threshold`, `--list-weights`) |

## Weight key prefixes (fastino/gliner2-large-v1)

```
encoder.embeddings.*                 DeBERTa V2 embeddings
encoder.encoder.layer.N.*            DeBERTa V2 transformer layers
span_rep.span_rep_layer.project_start.{0,3}.*
span_rep.span_rep_layer.project_end.{0,3}.*
span_rep.span_rep_layer.out_project.{0,3}.*
classifier.{0,2}.{weight,bias}       binary span scorer MLP (loaded, not yet wired)
count_pred.{0,2}.{weight,bias}       extraction count predictor MLP
count_embed.pos_embedding.weight
count_embed.gru.*
count_embed.projector.{0,2}.*
```

MLP Sequential layout (PyTorch): `0=Linear, 1=GELU, 2=Dropout (skipped at inference), 3=Linear`
→ only indices **0** and **3** (or **0** and **2** for 2-element MLPs) have weights.

## Known limitations / next steps

- Entity extraction only — relations, classifications, and JSON structures not yet implemented
- `clf1`/`clf2` (binary span classifier) loaded but not wired into the scoring path
- Batch inference not implemented (single text at a time)
- No Metal / CUDA device tested yet (`--device cuda:0` flag exists)
- SpanRepLayer assumes markerV0 boundary-pair strategy; verify against gliner source if results look off
