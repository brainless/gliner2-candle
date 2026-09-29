# gliner2-candle — Developer Notes

GLiNER2 entity extraction in pure Rust using [Candle](https://github.com/huggingface/candle).
Targets the **fastino/gliner2-*** model family (not the original GLiNER / urchade models).

## Quick start

```bash
# Download is automatic with --model-id (first run; cached under ./models/ after)
cargo run --release -- --model-id fastino/gliner2.5-small-v1 \
  --text "Apple was founded by Steve Jobs in Cupertino." \
  --entities "person,organization,location"

# Or point --model-dir at an existing directory (no download):
hf download fastino/gliner2-large-v1 --local-dir ./models/gliner2-large-v1

# Inspect weight keys (verify prefixes after model updates)
cargo run -- --model-dir ./models/gliner2-large-v1 --list-weights 50   # first 50
cargo run -- --model-dir ./models/gliner2.5-small-v1 --list-weights    # complete inventory

# Run inference
cargo run --release -- \
  --model-dir ./models/gliner2-large-v1 \
  --text "Apple was founded by Steve Jobs in Cupertino." \
  --entities "person,organization,location"

# Agent verification case file (see "User CLI & agent verification cases")
cargo run --release -- --model-id fastino/gliner2.5-small-v1 --cases cases/cli-cases.json
```

## Architecture

Span (`architecture` missing or `"span"`) pipeline — boundary checkpoints are
dispatched before any of this loads (see "Architecture dispatch & config
parsing" below):

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

Boundary (`architecture: "boundary"`, `fastino/gliner2.5-*`) pipeline
(`src/boundary*.rs`; the span modules above never run):

```
schema groups (entities [E] markers, relations [R] head/tail roles)
    + [SEP_TEXT] + words                     BoundaryPrepared (boundary_pre.rs)
    │
    ▼
DeBERTa V2 encoder          candle_transformers::models::debertav2
    │
    ├─► text states          first-subtoken gathering (word rows)
    ├─► query states         contextual state at each [E]/[R] marker
    │                        relation query = concat(head,tail) if
    │                        directional_relation_states else mean
    │
    ▼
Boundary encoder            left token/BOS + right token/EOS → project →
(boundary_enc.rs)           concat → project → LayerNorm → pre-norm local
                            self-attention + residual SwiGLU refinement
    │
    ▼
Marginal heads              start/end logits [B,Q,L+1], inside [B,Q,L]
(heads.py port)             scaled dot products, MASK_LOGIT=-1e4, inside
                            prefix fp32 + per-query mean centering
    │
    ▼
Shared candidate pool       query-agnostic pool builder + pooled pair scorer
(boundary_pool.rs)          (candidate_pool: "shared" for gliner2.5-small-v1;
                            per_query proposer/scorer NOT ported)
    │                        pair logits = reranker compat + start + end
    │                        marginals (once) + content/length terms
    ▼
Entities                    pair_temperature → sigmoid → threshold →
(boundary_decode.rs)        abstention / adaptive fill → flat overlap
                            (weighted interval scheduling per type) →
                            exact slice of the caller's input (byte offsets)
    │
    ▼
Relations                   typed/capped head-tail pair generation from raw
(boundary_rel.rs)           candidate pair logits (no pair_temperature) →
                            SparseRelationScorer (endpoint gather + relative
                            position MLP + biaffine) → relation_temperature
                            → threshold → edge dedup → directed edges
```

## Source layout

| File | Purpose |
|---|---|
| `src/config.rs` | `Gliner2Config` (config.json) + `Architecture` enum + `BoundaryHeadConfig` (parse/migrate/validate) + `EncoderConfig` loader |
| `src/model.rs` | `Model` dispatch enum (`Span`/`Boundary`); `SpanModel::load()` + `forward()` + `predict()`; shared `predict_entities()` / `predict_relations()` |
| `src/boundary.rs` | `BoundaryModel` load + validated head weights + `encode`/`forward_head`/`forward` (Task 5 marginals + Task 6 shared pool/pair scoring + Task 7 null/count heads) + `decode_entities`/`predict_entities` + `propose_relations`/`decode_relations`/`predict_relations` |
| `src/boundary_enc.rs` | Boundary encoding (`encoding.py`) + marginal heads (`heads.py`) — epic Task 5 |
| `src/boundary_pool.rs` | Shared candidate pool + pooled pair scorer (`pool.py`) — epic Task 6 |
| `src/boundary_rel.rs` | Relation pair generator + `SparseRelationScorer` + relation decode/dedup (`relations.py` / `_decode_relations`) — epic Task 8 |
| `src/span_rep.rs` | `SpanRepLayer`: three 2-layer MLPs (project_start, project_end, out_project) |
| `src/count_lstm.rs` | `CountLSTM`: positional embedding + GRU + projector MLP |
| `src/processor.rs` | Schema → `input_ids` for entity extraction task only (span path; untouched by boundary work) |
| `src/boundary_pre.rs` | Boundary-path preprocessing & query routing (epic Task 4) + oracle parity tests |
| `src/boundary_decode.rs` | Boundary entity decoder (epic Task 7): threshold / abstention / adaptive fill / `flat` overlap resolution / caller-text slicing |
| `src/inference.rs` | Sigmoid, threshold, greedy overlap removal, char-offset mapping (span path only — untouched by boundary work) |
| `src/hub.rs` | `--model-id` download & cache contract (model-specific dirs, revision pinning, atomic fetch, identity-verified reuse) — epic Task 9 |
| `src/cases.rs` | Versioned agent case file: schema, validation, strict matching — epic Task 9 |
| `src/main.rs` | CLI (`--model-dir`, `--model-id`, `--revision`, `--models-root`, `--text`, `--entities`, `--relations`, `--threshold`, `--cases`, `--json`, `--list-weights`) |

## Weight key prefixes (fastino/gliner2-large-v1, span checkpoint)

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

MLP Sequential layouts (PyTorch): `Linear,GELU,Dropout,Linear` has weights at
indices **0** and **3** (span_rep projections, boundary `film_output`/`mlp`);
`Linear,ReLU,Linear` has weights at indices **0** and **2** (large-v1
`classifier`, `count_pred`, `count_embed.projector`). The right indices depend
on the module composition — check the safetensors keys rather than assuming
(this is a documented past bug class).

## Checkpoint inventory (epic `gliner-2.5-boundary-support`, Task 1)

Verified 2026-09-29 against the downloaded artifacts and the pinned Python
reference. Everything below was read from the real files, not from defaults.

### Checkpoints and revisions

**Supported checkpoints (exhaustive):** `fastino/gliner2-large-v1` (span,
entity-only) and `fastino/gliner2.5-small-v1` (boundary, entities + relations)
— the two rows below. **`fastino/gliner2.5-base-v1` and
`fastino/gliner2.5-multi-v1` (and any other base/multi checkpoints) are NOT
claimed supported**: per the epic they must be inventoried and validated
separately (including the multilingual splitter/pooling path for multi)
before support is claimed, and that has not been done.

| Local dir | Hub ID | Architecture | Hub revision (pinned) | Weights |
|---|---|---|---|---|
| `./models/gliner2-large-v1/` | `fastino/gliner2-large-v1` | span (legacy) | `5312584a6fd5543e457ba5f309ac5db226431d1a` | `model.safetensors` (single file, 419 tensors) |
| `./models/gliner2.5-small-v1/` | `fastino/gliner2.5-small-v1` | **boundary** | `7132dc4561c3f94563c6147e75ffa8ef34c4964a` | `model.safetensors` (single file, 334 tensors, F32, 295,567,700 bytes, sha256 etag `4ee98278…55de2b`) |

- Revisions are recorded in each dir's `.cache/huggingface/download/*.metadata`
  (line 1 = commit sha). Hub HEAD of `fastino/gliner2-large-v1` has since moved
  to `d7aa8a2850d175a9227ff80358ea380723d03ad2`; the local copy is the pinned
  older revision and is what regression checks must use.
- **Shards: none.** Both checkpoints are one `model.safetensors`; there is no
  `model.safetensors.index.json`. The single-file loader assumption holds.
- Python behavioral reference: `~/Projects/GLiNER2` @
  `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` (pinned). Verified this commit
  **loads and executes** the 2.5 checkpoint: `AutoExtractor.from_pretrained`
  strict-loads all 334/334 tensors (`BoundaryExtractor`, config_version 3),
  and entity, relation-only, and combined entity+relation extraction all run
  (README Apple sentence → Apple/Steve Jobs/Cupertino; `Steve Jobs founded
  Apple in Cupertino.` + `founder` → directed edge Steve Jobs → Apple).

### fastino/gliner2.5-small-v1 file layout

```
config.json                  extractor config (architecture "boundary")
encoder_config/config.json   DeBERTa encoder config (model_type deberta-v2)
tokenizer.json               full SPM tokenizer (self-sufficient)
tokenizer_config.json        metadata (transformers-5-style)
model.safetensors            all weights
README.md, SKILL.md, *.png   docs/banners only
```

Unlike large-v1 there is **no** `spm.model`, `added_tokens.json`, or
`special_tokens_map.json` — `tokenizer.json` alone defines everything.
`tokenizer_config.json` stores `extra_special_tokens` as a **list**
(transformers 5.x format; checkpoint was saved with `transformers_version`
5.8.0). On transformers 4.x this trips `AutoTokenizer` before GLiNER2's
`load_extractor_tokenizer` compatibility retry can see it unless `protobuf` is
installed; the `tokenizers` crate used by Rust reads `tokenizer.json` directly
and is unaffected.

Tokenizer facts (SPM, `add_prefix_space: true`, vocab 128011):
`[PAD]=0 [CLS]=1 [SEP]=2 [UNK]=3 [MASK]=128000`,
`[SEP_STRUCT]=128001 [SEP_TEXT]=128002 [P]=128003 [C]=128004 [E]=128005
[R]=128006 [L]=128007 [EXAMPLE]=128008 [OUTPUT]=128009 [DESCRIPTION]=128010`.
Note this vocab size (128011) differs from stock `microsoft/deberta-v3-xsmall`
(128100) — the encoder was resized for GLiNER2's added special tokens.

### Weight keys (334 tensors, all F32)

Grouped summary (full listing: `--list-weights` with no value):

```
encoder.embeddings.word_embeddings.weight        [128011, 384]
encoder.embeddings.LayerNorm.{weight,bias}       [384]
encoder.encoder.layer.N.*                        12 layers:
    attention.self.{query_proj,key_proj,value_proj}.{weight,bias}  [384,384]
    attention.output.{dense.{weight,bias},LayerNorm.{weight,bias}} [384,384]
    intermediate.dense.{weight,bias}             [1536,384]
    output.{dense.{weight,bias},LayerNorm.{weight,bias}}          [384,384]
encoder.encoder.rel_embeddings.weight            [512, 384]   (= 2×position_buckets)
encoder.encoder.LayerNorm.{weight,bias}          [384]        (norm_rel_ebd=layer_norm)

boundary_head.boundary_encoder.*                 boundary encoding:
    left_projection / right_projection           [128, 384]
    bos_state / eos_state                        [384]
    output_projection                            [128, 256]   (concat 2×128)
    layer_norm                                   [128]
    attention_blocks.{0,1}.*                     2 layers, qkv [384,128], out [128,128], norm [128]
    refinement_blocks.0.*                        1 SwiGLU block: input [512,128], out [128,256], norm
boundary_head.boundary_query_head.*              start/end/inside marginals:
    {start,end,inside}_query_projection          [128, 384]
    {start,end}_boundary_projection              [128, 128]
    inside_text_projection                       [128, 384]
boundary_head.null_projection.{weight,bias}      [1, 384]    (abstention null logit)
boundary_head.count_head.{weight,bias}           [1, 384]    (count log rate; NOT the span CountLSTM)
boundary_head.candidate_encoder.{weight,bias}    [384, 256]  (records-only candidate states)
boundary_head.shared_pool_builder.*              ACTIVE shared pool:
    {start,end}_projection                       [128, 128]
boundary_head.shared_pool_scorer.*               ACTIVE shared scorer:
    query_projection                             [128, 384]
    {start,end}_projection                       [128, 128]
    length_projection                            [128, 3]    (log1p(len), len/|text|, 1/sqrt(len))
    prior_projection                             [128, 1]
    content_pooler.{value_projection,layer_norm} [64, 384] / [64]
    content_projection                           [128, 64]
    candidate_norm                               [128]
    film                                         [256, 128]
    film_output.{0,3}                            [64,128] / [1,64]  (Linear,GELU,Dropout,Linear)
boundary_head.boundary_proposer.*                INACTIVE (per_query path):
    start_query_projection [64,384], start_pair_projection [128,128], end_key_projection [128,128]
boundary_head.pair_scorer.*                      INACTIVE (per_query path):
    {start,end}_endpoint_projection [128,128]    (reranker_endpoint_compat)
    endpoint_difference_projection [1,256]       (endpoint_difference_features)
    compat_mix [1,8]                             (multihead_pair_compat_heads=8)
    inside_weight [1,384]                        (query_conditioned_inside_weight)
    length_query_projection [3,384], content_* (content_dim=64), query_gate [64,384]

relation_scorer.*                                SparseRelationScorer (12 tensors):
    mlp.{0,3}                                    [384,2306] / [1,384]
                                                 2306 = 4×384 endpoint states + 768 relation query + order + dist
    head_content_projection / tail_content_projection  [384,384]
    relation_content_gate                        [384, 768] (directional relation query)
    content_linear                               [1, 1536]  (biaffine content: 2×384 + 768)
record_decoder.*                                 RecordHead (18 tensors) — records task, out of epic scope
classifier.{0,3}                                 Sequential(Linear(384,768),ReLU,Dropout,Linear(768,1))
                                                 — classification-task head, not entity/relation inference
```

### Active inference flags (boundary_head, config_version 3)

Python `migrate_config_dict` at the pinned commit: config_version 3 is the
current version, so **all serialized values are taken as-is** (only
config_version < 3 would default `enable_records`/`enable_relations` to false).
Values below are the migrated truth for `fastino/gliner2.5-small-v1`.

**Candidate pool verdict: `candidate_pool: "shared"`** — not `per_query`.
`BoundaryHead.forward` runs `shared_pool_builder` + `shared_pool_scorer`
(+ `candidate_encoder` for records states). `boundary_proposer` and
`pair_scorer` are constructed and present in the state dict but are **not
executed** at inference (only a diagnostics dry-run during training). Never run
the per_query math for this checkpoint.

| Group | Flags |
|---|---|
| Task/top-level | `architecture=boundary`, `architecture_version=1`, `token_pooling=first`, `max_len=4096`, `attn_implementation=sdpa`, `model_name=microsoft/deberta-v3-xsmall` |
| Shared path (**active**) | `pool_size=192`, `pool_boundary_top_k=32`, `min_pool_per_query=8`, `candidate_attention_layers=0`, `candidate_attention_heads=4`, `query_attention_layers=0` (0 → no attention weights in either) |
| Per_query only (**inactive here**) | `start_top_k=24`, `end_top_k=24`, `starts_per_end=12`, `ends_per_start=12`, `candidate_budget=192`, `training_candidate_budget=192`, `end_block_size=256`, `bidirectional_proposals=true`, `boundary_top_k_alpha=0.08`, `boundary_top_k_max=128`, `boundary_top_k_bucket=8`, `export_mode=auto`, `vectorized_pair_elements=16777216`, `enable_rotary_endpoints=true`, `rotary_base=10000.0`, `reranker_endpoint_compat=true`, `endpoint_difference_features=true`, `multihead_pair_compat_heads=8`, `query_conditioned_inside_weight=true` (rotary & those four scoring flags only feed `SparseBoundaryProposer`/`SparseBoundaryPairScorer`, so they do not affect this checkpoint's math) |
| Boundary encoding (both paths) | `boundary_dim=128`, `boundary_attention_layers=2`, `boundary_attention_heads=4`, `boundary_attention_window=128`, `boundary_refinement_layers=1`, `boundary_ffn_multiplier=2.0`, `use_inside_evidence=true` |
| Scoring flags shared by both pools | `enable_span_content=true`, `content_dim=64`, `content_soft_max_pool=false`, `pair_dim=128` |
| Entity decode | `pair_temperature=1.0` (divide before sigmoid), `adaptive_threshold=false`, `enable_abstention=true` + `abstention_threshold=0.5`, `enable_count_head=true` (unused while `adaptive_threshold=false`), `overlap_policy=flat` (exact-span dedup + maximum-total-score weighted interval scheduling per entity type) |
| Relations (**enabled**) | `enable_relations=true`, `directional_relation_states=true` (relation query = concat of head/tail role states, dim 768), `relation_biaffine_content=true`, `relation_heads_per_type=32`, `relation_tails_per_type=32`, `relation_pair_cap=64`, `relation_argument_proposal_threshold=0.2`, `relation_temperature=1.0` |
| Records (**enabled but out of scope**) | `enable_records=true`, `record_dim=128`, `record_instance_queries=32`, `record_anchor_threshold=0.5`, `record_anchor_proposal_threshold=0.2`, `record_field_threshold=0.5`, `record_temperature=1.0` — `record_decoder.*` weights exist; do not run or expose this task |

Relation availability: `enable_relations=true` **and** `relation_scorer.*`
weights are present **and** the path executes (verified in Python: directed
`founder` edge extracted). Relation support for this checkpoint can be claimed
once ported. Argument selection uses `sigmoid(pair_logits)` **without**
`pair_temperature`, gated by `relation_argument_proposal_threshold`.

Optional modules present in the state dict vs executed by config — for this
checkpoint: `boundary_proposer`/`pair_scorer` are present but inactive (shared
pool selected); `record_decoder` is present and enabled (`enable_records=true`)
and `classifier` is present as the classification-task head — both outside this
epic's tasks; `count_head` is
configured but unused while `adaptive_threshold=false`. Weight presence alone
never selects a path — always read `boundary_head.candidate_pool` after
Python's config migration.

### DeBERTa-v3 encoder compatibility (Candle 0.9.2)

`encoder_config/config.json` says `model_type: deberta-v2` with DeBERTa-v3-xsmall
geometry: `hidden=384`, `layers=12`, `heads=6`, `intermediate=1536`,
`vocab_size=128011`, `max_position_embeddings=512`, `position_buckets=256`,
`max_relative_positions=-1`, `relative_attention=true`, `norm_rel_ebd=layer_norm`,
`pos_att_type=[p2c,c2p]`, `share_att_key=true`, `type_vocab_size=0`,
`position_biased_input=false`, `hidden_act=gelu`, `layer_norm_eps=1e-7`,
`legacy=true`, `dtype=float16` (metadata only; saved tensors are F32).

Verdict: **compatible** — `candle_transformers::models::debertav2` (resolved
0.9.2) loads this checkpoint as-is; no encoder prerequisite is missing.
Checked against the resolved 0.9.2 source in `~/.cargo/registry`, not the local
0.11 checkout:

- Vocabulary: `word_embeddings` [128011, 384]; size taken from config
  (`embedding_size` absent → equals `hidden_size`, so no `embed_proj`).
- Embeddings: `position_biased_input=false` → no position embeddings loaded;
  `type_vocab_size=0` → no token-type embeddings; checkpoint contains neither
  key. `embeddings.LayerNorm` present as expected.
- Relative positions: `rel_embeddings` [512, 384] matches Candle's
  `position_buckets(256) × 2` sizing rule; `norm_rel_ebd=layer_norm` matches
  `encoder.LayerNorm.{weight,bias}`; `share_att_key=true` matches the absence
  of `pos_key_proj`/`pos_query_proj` (attention `query_proj`/`key_proj` are
  reused for relative positions).
- Unknown config fields (`legacy`, `dtype`, `transformers_version`, …) are
  ignored by serde; `conv_kernel_size` absent → no ConvLayer (checkpoint has
  none).
- Numerical check (throwaway harness, since removed): Candle
  `DebertaV2Model::load` + forward on fixed token ids vs the pinned Python
  HF encoder on the same ids — max abs diff **4.5e-6** over a 6×384 output
  (fp32 noise). The `gelu` activation maps to erf-gelu on both sides.
- All 12 encoder layers use `query_proj`/`key_proj`/`value_proj` names exactly
  as Candle expects under `vb.pp("encoder")` (full key prefix `encoder.` in the
  safetensors, same as the span checkpoint).

## Architecture dispatch & config parsing (epic Task 3)

`src/config.rs` mirrors Python's flow in `gliner2/configuration.py` (pinned
commit): `migrate_config_dict` → typed parse → `validate_boundary_head`.

1. **Migration** (`migrate_config_dict`): resolves `architecture` (missing /
   null / `""` → `span`; unknown → error mirroring Python's
   `Unknown extractor architecture …`), and for boundary configs with
   `config_version < 3` defaults **missing** `enable_records` /
   `enable_relations` to `false` (old checkpoints must not silently gain
   tasks). `config_version` ≥ 3 (or explicit values) are taken as-is.
2. **Typed parse**: `Gliner2Config` with the `Architecture` enum
   (`Span` | `Boundary`, missing → `Span`) and `BoundaryHeadConfig` —
   all inference-active `boundary_head` fields with Python's
   `BoundaryHeadSettings` defaults for anything missing.
3. **Validation**: dimension/range/enum rules ported from Python's
   `validate_boundary_head` — positive dims (`boundary_dim`, `pair_dim`,
   `start_top_k`, `end_top_k`, `ends_per_start`, `starts_per_end`,
   `candidate_budget`, `end_block_size`, `content_dim`, …), even-dim rule for
   `enable_rotary_endpoints`, `pair_dim`/`boundary_dim` divisibility by
   `multihead_pair_compat_heads` / attention heads, `candidate_pool` /
   `export_mode` / `overlap_policy` membership, pool sizes and
   `min_pool_per_query ≤ pool_size`, `boundary_top_k_max ≥ max(start,end)_top_k`,
   adaptive-budget vs `export_mode='vectorized'`, temperature ranges, relation
   caps/thresholds. Training-only fields (loss weights, focal loss, hard
   negatives, `training_candidate_budget`, `max_gold_per_query`, `dropout`,
   record/classification loss knobs) are **not parsed** — they never change
   inference math. Record/classification *tasks* stay out of epic scope:
   `enable_records` is parsed but never exposed or run.

**Unsupported *active* modes error at load** (silently ignoring a checkpoint
flag is a documented bug class):

| Condition | Error |
|---|---|
| `architecture` not `span`/`boundary` | `Unknown extractor architecture …` |
| `token_pooling != "first"` (both archs) | unsupported — only first-subtoken pooling is ported |
| `counting_layer != "count_lstm"` (span) | unsupported — `count_lstm_moe`/`_v2` not ported |
| `boundary_head.overlap_policy != "flat"` | unsupported — `nested`/`longest` not ported (epic Task 7 may relax) |
| shared pool with `candidate_attention_layers > 0` or `query_attention_layers > 0` | unsupported — those attention blocks not ported (epic Task 6) |
| `enable_span_content` + `content_soft_max_pool` | unsupported — `SpanContentPooler`'s smooth-maximum (LSE) branch not ported; only the mean-pooling path runs (epic Task 6) |

Invalid enum values (`candidate_pool`, `export_mode`, `overlap_policy`) error
Python-style regardless of support. Inactive flags (e.g. the per_query
scoring/rotary flags while `candidate_pool == "shared"`) are parsed and
validated but not gated — they do not affect the selected path's math.

`src/model.rs::Model` is the dispatch enum (`Span(SpanModel)` /
`Boundary(BoundaryModel)`); `Model::load` selects the variant from
`config.json` **before** any head weight lookups, so a boundary checkpoint
never touches span-head keys and vice versa. The CLI calls the shared
`Model::predict_extractions(text, entity_types, relation_types, threshold)`
(epic Task 9's combined operation over `predict_entities` /
`predict_relations`). The span forward
path (`SpanModel::forward`) is unchanged; span preprocessing moved into
`SpanModel::predict_text`. `SpanRepLayer`/`CountLSTM` stay span-only —
boundary math lives in `src/boundary.rs` + `src/boundary_enc.rs` (Tasks 5–8),
never inside the span modules.

Boundary loading at this task = config parse + **real weight
presence/shape validation** + dispatch scaffolding; boundary *math* is not
implemented at this task (epic Task 5 later added the encoding/marginals
forward — see "Boundary encoder & marginals" above). `src/boundary.rs::expected_boundary_weights` derives every
`(tensor name, shape)` the **active** path reads from the config (dims,
`candidate_pool`, flag-gated modules) and `validate_boundary_weights` loads
each tensor through the VarBuilder — existence + exact shape, per the
"Checkpoint inventory" above. For `candidate_pool: "shared"` that covers
`boundary_encoder`, `boundary_query_head`, `null_projection`/`count_head`
(flag-gated), `shared_pool_builder`, `shared_pool_scorer` (content modules
gated by `enable_span_content`) and `relation_scorer` (gated by
`enable_relations`, shapes depend on `directional_relation_states` /
`relation_biaffine_content`); the inactive `boundary_proposer`/`pair_scorer`
weights are optional and untouched (and the reverse for a `per_query`
checkpoint). `candidate_encoder` (records-only), `record_decoder` and
`classifier` are never required. `Model::predict_entities` on a boundary
checkpoint now runs the full entity pipeline (Tasks 5–7: marginals, shared
candidate pool/pair scoring, and the entity decoder — see "Boundary encoder &
marginals", "Shared candidate pool & pair scoring", and "Boundary entity
decoder" below); relations are accepted by the shared
`Model::predict_relations` (epic Task 8 — see "Boundary relation extraction"
below).

Verified 2026-09-29 against `oracle/manifest.json`'s
`config.migrated_boundary_settings`: all 51 parsed `boundary_head` fields
match exactly; the 32 unparsed manifest fields are training-only/record-task
knobs. `RUST_LOG=debug` prints the migrated config as JSON for re-checking.

## Boundary preprocessing & query routing (epic Task 4)

`src/boundary_pre.rs` is the boundary-path preprocessor. It is deliberately
**boundary-specific**: the span path (`src/processor.rs`) keeps its own splitter
(byte offsets, no URL/email/`@mention` recognition) and its behavior is
unchanged. Entry point: `BoundaryModel::preprocess(text, entity_types,
relation_types, max_len)` → `BoundaryPrepared`. Ported at the pinned commit
from `SchemaTransformer`'s entity/relation path (`gliner2/processor.py`), the
query enumeration of `build_boundary_batch_metadata`
(`gliner2/processing/boundary_preprocessing.py`), and the relation role
routing of `_encode_core` (`gliner2/models/boundary/model.py`).

### Schema construction (transformed schema order)

- The **entities** group comes first — `( [P] entities ( [E] t1 [E] t2 … ) )`,
  one `[E]` marker per type in declared order — then one **relations** group
  per relation type in declared order — `( [P] name ( [R] head [R] tail ) )`.
  Duplicate type names collapse to first appearance (Python dict / group-key
  semantics of `Schema.entities`/`Schema.relations` +
  `_process_entities`/`_process_relations`). `json_structures`/`classifications`
  are out of epic scope and are not accepted by this path.
- Query layout (`QueryEntry`, aligned with the oracle `query_layout`): every
  extractive schema child (`[E]`/`[R]` marker) becomes one boundary query in
  group/marker order; the output type (`field_name`) is the token after the
  marker, so output types stay aligned with marker order.
- Relation role routing (`RelationRoute`, aligned with the oracle
  `relation_role_routing`): per relations group, `head_query_id`/`tail_query_id`
  are the group's first two role queries (single ids, as Python's
  `RelationTypeSpec`); the relation query state is `concat(head,tail)` when
  `boundary_head.directional_relation_states` (this checkpoint: on, dim
  768 = 2 × 384) else `mean(head,tail)` — both modes implemented.
- Input assembly mirrors `_format_input_with_mapping`: schema groups joined by
  `[SEP_STRUCT]`, then `[SEP_TEXT]`, then the words; every combined token is
  tokenized **in isolation** (identical to Python's per-token `tokenize()`
  through the shared `tokenizers` backend). Word routing rows are 1:1 with
  words even when a word tokenizes to nothing (placeholder row, as Python).
  Only structural marker slots (`[P]` at index 1 and child markers at 4, 6, …
  per schema) are routed — prompt/label text never is.

### Text normalization and offset conventions

- Python `_collate_batch` normalizes **before** splitting: `""` → `"."`; input
  without terminal `.`/`!`/`?` gets a synthetic `"."` appended. The normalized
  text is what the model sees and what **Python** decodes against. Rust matches
  that normalized token sequence exactly for model parity (oracle
  `normalized_text`) but retains the caller's exact input separately
  (`TextMap::caller_text`).
- Python offsets are Unicode code points; Rust converts to UTF-8 byte offsets
  immediately. `BoundaryPrepared.word_char_spans` = byte offsets into
  `TextMap::normalized_text()`; `TextMap::start_mapping`/`end_mapping` expose
  the code-point values (Python `start_mappings`/`end_mappings`).
- Candidates map via `TextMap::candidate_bytes(start, end)` using the
  candidate-decoder convention `start_mappings[start]` .. `end_mappings[end-1]`
  on a half-open word interval. A candidate whose mapped character range
  reaches into the synthetic suffix is **rejected** (`None`): Python decodes
  against the normalized text and can surface the synthetic `"."`, while Rust
  must never return a surface or byte range outside the caller's input. This is
  the documented user-facing difference from Python's normalized output — e.g.
  for input `"Alice greeted Bob"` a passing Python candidate `[2,4)` would print
  `"Bob."`; Rust drops it entirely. The rule is inert for the oracle corpus at
  threshold 0.5 but is tested (`suffix_crossing_candidates_are_rejected`,
  including a URL word that swallows the synthetic dot into one split word and
  the empty-text case where every candidate is rejected). Surviving ranges
  slice the caller's input via `TextMap::surface` and are byte-exact for
  Unicode (`surfaces_slice_caller_input`, `unicode_byte_offsets_slice_exactly`).
- Word splitting ports `WhitespaceTokenSplitter` exactly (URL / email /
  `@mention` recognition, Python's `str` word class, `\x1c`–`\x1f` whitespace
  quirks included; case-fold only the token value so offsets stay valid).
- Request `max_len` truncates **words** before schema encoding (Python
  `collate_fn_inference(max_len=…)`; `None` = no truncation). The config's
  `max_len` is *not* applied at inference — matching Python's `extract()`,
  which passes `max_len=None`. `token_pooling` must be `"first"` (rejected at
  config load; re-checked in `preprocess` with a clear error).

### Oracle comparison (Task 4 acceptance)

`cargo test` runs the `#[cfg(test)]` harness at the bottom of
`src/boundary_pre.rs` against all 18 `oracle/cases/NN_*.json` (requires
`./models/gliner2.5-small-v1`; fixture code-point spans are converted to UTF-8
bytes before comparison). Exact equality is asserted on `token_ids`,
`first_subtoken_positions`, `query_marker_positions`, `schema_marker_positions`,
`schema_tokens`, `task_types`, `words`, `word_char_spans`, `normalized_text`,
`query_layout`, and `relation_role_routing` (mode string + `query_state_dim`).
Verified 2026-09-29: **all 18 cases exact match** — including `07_url_email`
(splitter), `08_unicode` (code-point→byte offsets), and the relation routing of
`12`–`18` (concat query state, dim 768). Focused checks cover suffix
rejection (05/06/10 + URL-swallow), surface slicing of every mapped word over
all cases, and `max_len` truncation. Span regression on
`./models/gliner2-large-v1` (README example) is byte-identical before/after —
the span preprocessing path is untouched.

## Boundary encoder & marginals (epic Task 5)

`src/boundary_enc.rs` ports `gliner2/models/boundary/encoding.py`
(`BoundaryEncoder`, `BoundaryAttentionBlock`, `ResidualSwiGLU`) and
`gliner2/models/boundary/heads.py` (`BoundaryQueryHead` → `BoundaryMarginals`)
at the pinned commit. `src/boundary.rs` wires them into `BoundaryModel::encode`
(DeBERTa forward + first-subtoken/`[E]`-marker gathers = `_encode_core`'s
gather stage → `EncodedStates`) and `BoundaryModel::forward_head` / `forward`,
which **stop at marginals + states**: `BoundaryOutputs` exposes
text/query states and masks, boundary states + mask, and the
start/end/inside marginals with `inside_prefix` / `inside_prefix_mean` —
exactly what epic Task 6's shared-pool builder/scorer (`pool.py`) consumes.
Tensors keep the Python batch dimension (`B=1` today).

### Active vs inactive features for `fastino/gliner2.5-small-v1`

| Feature (source) | Config | Status |
|---|---|---|
| left/right projection, concat, `output_projection`, `layer_norm` (encoding.py) | — | **active**, ported |
| pre-norm local self-attention blocks | `boundary_attention_layers=2`, `boundary_attention_heads=4`, `boundary_attention_window=128` | **active**, ported (manual SDPA: `candle_nn::ops::sdpa` has no CPU path) |
| residual SwiGLU refinement | `boundary_refinement_layers=1`, `boundary_ffn_multiplier=2.0` | **active**, ported |
| dropout (all modules) | `dropout=0.1` | **off at inference** (eval mode) — not ported, identity |
| start/end/inside scaled dot-product heads (heads.py) | `boundary_dim=128` | **active**, ported |
| fp32 mean-centered inside prefix + mean restore | `use_inside_evidence=true` | **active**, ported |
| `query_conditioned_inside_weight` | `true` | **inactive** for this checkpoint — feeds only the per_query `pair_scorer`; `heads.py` always uses `inside_query_projection` (port uses it) |
| padding/mask behavior (boundary/token/query masks) | — | ported; trivial at `B=1` inference (all valid), verified vs Python on a synthetic padded batch (below) |

No active encoding/heads feature is unsupported; nothing fails at load for
this checkpoint. `null_projection`/`count_head` (abstention/count) are simple
query-state linears consumed by the Task 7 decode contract (see "Boundary
entity decoder" below).

### Parity-critical math (heads.py / encoding.py conventions)

- Boundary `i` sits between token `i-1` (left) and token `i` (right):
  boundary 0 uses the learned `bos_state`, each sample's final boundary `n_b`
  uses `eos_state` (also written at `L`, masked out). The **final valid
  boundary is EOS, including after truncation**.
- Attention: pre-norm qkv → heads (`head_dim = 32`, scale `1/sqrt(32)`);
  `allowed = key_mask && |i-j| <= window`, OR-ed with the **diagonal** (so
  padding query rows keep one legal key — no NaN softmax rows); masked scores
  filled with `-inf` before softmax. Each attention block zeroes padding
  boundary rows after its residual; refinement is unmasked; the encoder's
  final output is zeroed at padding boundaries. LayerNorm eps `1e-5` (PyTorch
  default) matches `nn.LayerNorm`.
- Marginal masking uses the finite sentinel `MASK_LOGIT = -1e4`
  (`constants.py`) at invalid boundary/token/**query** positions — exact on
  both sides, finite so sums stay safe.
- **Inside prefix and the mean restore** (the easily-omitted term): invalid
  inside positions become 0 for prefix purposes; each query's valid inside
  logits are centered by their mean (`valid_count` clamped ≥ 1) and cumsum'd
  **in fp32** with a leading zero → `inside_prefix [B,Q,L+1]`; the mean is
  carried as `inside_prefix_mean [B,Q,1]`. Interval scoring is
  `prefix[end] - prefix[start] + mean * (end - start)`
  (`BoundaryMarginals::inside_interval_sum`, mirroring
  `scoring.interval_prefix_score` including its index clamp). The identity
  `= sum(inside_logits[start..end])` holds exactly (fp32 cumsum rounding only)
  for intervals inside a sample's valid text; **omitting the restore changes
  every interval score by `mean * (end-start)`**. Rust tests assert both the
  Python parity and the identity.

### Oracle comparison and observed tolerances (Task 5 acceptance)

`oracle/capture_oracle.py` was extended minimally (pin checks untouched):
cases `01`/`05`/`10` carry `raw_boundary` (boundary states + mask,
start/end/inside logits, `inside_prefix`, `inside_prefix_mean`,
`interval_prefix_score` rows `[batch,query,start,end,value]`); case `01`
additionally carries `raw_core` (gathered `text_states`/`query_states`) and
the `after_layer_norm` / `after_attention_0` stages; and
`oracle/synthetic_masks.json` holds a synthetic padded `B=2, L=3, Q=2` batch
(inputs from an exact fp32 lattice formula recorded in the file) exercising
what the `B=1` corpus cannot: EOS at each sample's own `n_b`, boundary/token/
query masks, `MASK_LOGIT` fills, padding zeroing, centered prefix over masked
tokens, and the attention window (`window=2` re-run of attention block 0 — the
production `window=128` is inert at `N≤4`). The capture also fixes an earlier
bug where `shapes.boundary.inside_prefix` recorded `[1,Q,L]` instead of
`[1,Q,L+1]`.

**Tolerance: assert `< 1e-4` abs (fp32 CPU vs Python fp32)**. Observed max
abs diffs (`cargo test --release -- --nocapture`):

| Comparison | max abs diff |
|---|---|
| synthetic stages (layer_norm / attention / refinement / final states) | ≤ 1.5e-6 |
| synthetic marginals + prefix + mean | ≤ 1.2e-6 |
| synthetic `window=2` attention block | 2.4e-7 |
| 01 `text_states` / `query_states` (encoder + gather) | 6.2e-6 / 3.8e-6 |
| 01 boundary states (row 0/BOS 2.6e-6, row 9/EOS 1.8e-6) | 3.4e-6 |
| 01 marginals / prefix / mean | ≤ 7.2e-6 |
| 01 interval rows (`interval_prefix_score`) | **2.3e-5** (largest) |
| 05 / 10 all raw tensors | ≤ 4.1e-6 |

The 2.3e-5 outlier is interval reconstruction (fp32 cumsum + gather-diff vs
Python's gather-diff of its own cumsum) and is well inside the stated
tolerance. Boundary 0 (BOS) and L (EOS) rows are asserted explicitly in
addition to full-tensor comparison.

Focused Rust checks (`cargo test`, requires `./models/gliner2.5-small-v1`):
`src/boundary_enc.rs` — synthetic parity incl. stage order, `window=2` vs
full-attention divergence (the window must actually bind), `MASK_LOGIT` exact
at a padded query / padded boundary / padded token, padding row zeroing,
prefix zero origin, one-token `[s,s+1)` and edge `[0,1)/[n-1,n)/[0,n)`
interval identities (valid queries only — masked rows are pure sentinel and
never interval-scored); `src/boundary.rs` — shape asserts against the fixture
widths, boundary 0/L rows, raw parity for 01/05/10 (with 01 stages),
interval-row parity + identity on the real forward.

## Shared candidate pool & pair scoring (epic Task 6)

`src/boundary_pool.rs` ports `gliner2/models/boundary/pool.py` at the pinned
commit for the checkpoint-selected **`candidate_pool == "shared"`** path (this
checkpoint; the `per_query` `proposal.py`/`scoring.py` math is NOT ported and
never runs — `BoundaryModel::score_candidates` errors clearly for it).
`BoundaryModel::forward` = preprocess → encode → marginals (Task 5) →
`DocumentCandidatePool` + `SharedPoolScorer` → [`BoundaryForward`] with
query-agnostic pool rows (`[1,C,2]` half-open word intervals) and per-query
raw pair logits `[1,Q,C]` (pre `pair_temperature`). `C = pool_size = 192`.

### What executes for `shared` (pool.py)

**Selection — `DocumentCandidatePool.forward`** (inference: no gold
injection, `gold_pairs is None`):

1. Query-max endpoint union: `union_start[i] = amax_q start_logits[q,i]`
   (same for end) with the `MASK_LOGIT` floor at invalid query/boundary
   positions; `union_valid = boundary_mask & query_mask.any(-1)`.
2. `select_top_boundaries` on each side (`pool_boundary_top_k = 32`,
   `k = min(k, N)`): floor-fill invalid at `MASK_LOGIT`, **stable descending
   sort (index tie-break)**, invalid slots carry index 0.
3. One query-agnostic Cartesian pairing pass (`ks × ke ≤ 1024` rows here):
   `pair_valid = starts_valid & ends_valid & (end > start)`; per-pair
   `compat = (start_proj·end_proj)/sqrt(boundary_dim)` and the selection
   score `union_pair_score = compat + union_start[s] + union_end[e]`.
4. Quota band (`min_pool_per_query = 8`): every active query's strongest
   `quota = min(8, ks*ke)` pairs (ranked by `start_logits + end_logits +
   compat`, stable) are reserved at synthetic priority
   `-MASK_LOGIT * 0.5 + rank_bonus` (`5000 + [quota..1]`) — above ordinary
   global scores, below training gold injection (never at inference).
5. `_deduplicate_pool(keys = s*N + e, capacity = pool_size)`: invalid rows →
   sentinel key `N*N`/`MASK_LOGIT`; stable score-descending sort then stable
   key-ascending sort (each key's first occurrence is its highest-priority
   copy); `keep = valid & first`; final order = stable descending kept score
   over the key-ascending sequence = **descending score, ties (start,end)
   ascending**; capped at `pool_size` (padded `(0,false)`).
6. Retained rows rescored: `proposal_logits = compat + union_start[s] +
   union_end[e]` (invalid → `MASK_LOGIT`), `compat_logits` marginal-free
   (invalid → 0).

**Scoring — `SharedPoolScorer.forward`** (candidate features computed once,
then scored against all queries):

```
candidate = start_proj(boundary[s]) + end_proj(boundary[e])
          + length_proj(log1p(len), len/text_len, 1/sqrt(len))   # len = max(e-s, 1)
          + prior_proj(compat)                                    # marginal-free prior
          + content_proj(mean-pooled text span)                   # enable_span_content
candidate = LayerNorm(candidate) * valid_mask
score = candidate · query_proj(query) / sqrt(pair_dim)
      + film_output( candidate * (1 + gamma) + beta )             # gamma,beta = film(query)
      + start_marginal[q, s] + end_marginal[q, e]                 # added ONCE
      + (prefix[e] - prefix[s] + mean*(e-s)) / sqrt(max(e-s,1))   # inside evidence
```

Active optional terms for `fastino/gliner2.5-small-v1`: **span content**
(`enable_span_content=true`, `SpanContentPooler` mean path — fp32 cumsum
prefix, `sum/max(len,1)`, LayerNorm 1e-5; `content_soft_max_pool=false`, and
`true` is rejected at load), **inside interval evidence**
(`use_inside_evidence=true`, Task 5's centered fp32 prefix + `mean*(end-start)`
restore), and the **length features** `log1p(len)` / `len/text_length` /
`1/sqrt(len)` through `length_projection`. The `prior_projection` input is
`compat_logits` — **not** `proposal_logits` (which already carry the
marginals). Inactive here and not executed: `OverlapBiasedCandidateAttention`
(`candidate_attention_layers=0`), `EvidenceConditionedQueryAttention`
(`query_attention_layers=0`), and everything in `scoring.py`'s per_query
scorer (`endpoint_difference_features`, `reranker_endpoint_compat`,
`multihead_pair_compat_heads`, `query_conditioned_inside_weight`,
`enable_rotary_endpoints`) — those flags feed only the inactive
`boundary_proposer`/`pair_scorer`. FiLM + `film_output`
(`Linear,GELU,Dropout,Linear` = indices 0/3) is always active.

**Memory bound:** no `L x L` span grid is ever built. Selection materializes
at most `pool_boundary_top_k² = 1024` pair rows + `Q * min_pool_per_query`
quota rows (plain `f32` slices, stable Rust sorts mirroring
`torch.argsort(..., stable=True)`); the scorer works on `C = pool_size = 192`
rows with a `C x Q x pair_dim` FiLM tensor (`192 x Q x 128`). The
`export_mode`/`vectorized_pair_elements`/`end_block_size` streaming knobs are
`per_query`-only and inert for the shared pool (the Cartesian universe is
already bounded by the top-k).

**Marginals added once** (`task6_marginals_are_added_exactly_once`, real
forward + synthetic scorer in `src/boundary_pool.rs`): the pair score
decomposes as `pre_marginal + start + end + inside` with the gathered
start/end marginals at coefficient exactly 1 (`assert_eq!` against the raw
marginal value); the double-added variant differs on every non-degenerate
row; `proposal = compat + union marginals` while the scorer prior is `compat`
(Feeding `proposal` as prior changes the score — the prior-channel guard in
the synthetic test — so marginals cannot enter twice via prior + additive
terms).

### Oracle comparison and observed tolerances (Task 6 acceptance)

`cargo test --release` compares all 18 `oracle/cases/NN_*.json` against the
captured Python fields (`selected_candidate_indices`, `candidate_valid_mask`,
`candidate_proposal_logits`, `candidate_compat_logits`,
`candidate_pair_logits` — no capture extension needed). **Indices and valid
masks match exactly on every case** (including the query-agnostic shared rows
and the `10_empty_text` edge); logits tolerance is the stated **`< 1e-4` abs
(fp32 CPU vs Python fp32)**, observed aggregate max abs diffs:

| Comparison | max abs diff |
|---|---|
| `candidate_proposal_logits` (all 18 cases) | 1.7e-5 |
| `candidate_compat_logits` (all 18 cases) | 1.0e-5 |
| `candidate_pair_logits` (all 18 cases) | 3.8e-5 |

Per-case pair logits stay ≤ 2.2e-5 except `18_multi_type_combined` (3.8e-5,
the largest query set). The residual is fp32 noise from the Task 5 marginals
(≤ 2.3e-5 there) propagating through the scorer — no structural error.

Focused Rust checks (`cargo test`): `src/boundary_pool.rs` — dedup collapses
duplicate keys keeping the highest-priority copy, invalid rows never win,
budget cap holds exactly `pool_size` rows, `end > start` on every valid row
and invalid rows zeroed, tie-order determinism (equal logits → boundary-index
ascending; equal pool scores → (start,end) ascending; reruns bit-identical),
and the synthetic marginals-added-once decomposition; `src/boundary.rs` —
exact indices/masks + `< 1e-4` logits over all 18 cases, and the real-forward
marginals-added-once assertions on cases 01–03.

## Boundary entity decoder (epic Task 7)

`src/boundary_decode.rs` ports the entity decode contract (epic
"Source-checked inference contract" item 7): `gliner2/models/boundary/engine.py`
(`_extract_from_batch` → `_group_scored_candidates` → `_decode_entities`) and
`gliner2/inference/overlap.py` (default `flat` policy). The span path's
decoder (`src/inference.rs`) is untouched. Entry points: pure slice-based
helpers (`threshold_query_candidates`, `resolve_flat`) plus
`BoundaryDecoder::decode_entities`, and the tensor adapter
`BoundaryModel::decode_entities` / `predict_entities` that the CLI reaches via
`Model::predict_entities`. This API is entity-only; relation requests go
through `BoundaryModel::decode_relations` / `Model::predict_relations`
(epic Task 8 — see "Boundary relation extraction" below), which never filter
arguments through this decoder's threshold, abstention, or overlap policy.

### Decode pipeline (as implemented)

1. **Probabilities** — `p = sigmoid(pair_logit / pair_temperature)`: the
   division happens **before** the sigmoid
   (`torch.sigmoid(pair_logits / pair_temperature)`, f32 like the tensor).
   `pair_temperature` is `1.0` on this checkpoint (so inert at runtime) and
   the ordering is asserted synthetically
   (`pair_temperature_is_applied_before_sigmoid`).
2. **Threshold** (per query, emitted in pool-row order like Python's
   `keep.nonzero()`): `eligible = valid_mask & query_mask` (query mask all
   valid at `B=1`); `keep = eligible & (p >= threshold)` — **inclusive** at
   the boundary (asserted). The CLI's `--threshold` is the single score
   threshold; Python's per-type schema thresholds
   (`entity_metadata.threshold`) are not exposed by the Rust API.
3. **Adaptive fill** (`adaptive_threshold`) — **inactive** for this
   checkpoint but implemented and unit-checked from Python source:
   `predicted = torch.round(exp(count_log_rate)).long().clamp(0, C)`
   (round-half-to-even), and the kept set is **unioned** with the
   `predicted` top-ranked eligible rows (stable descending `p`, ineligible
   rows ranked at `MASK_LOGIT`): the fill adds below-threshold rows when
   fewer than `predicted` survived, never removes threshold hits, and never
   adds ineligible rows. `adaptive_threshold=true` with
   `enable_count_head=false` fails clearly ("adaptive threshold decoding
   requires count_log_rates"), mirroring Python's `ValueError`.
4. **Abstention** (`enable_abstention`) — `sigmoid(null_logit) >
   abstention_threshold` suppresses the **whole query** (strict `>`; equality
   does not abstain — asserted). `null_logits`/`count_log_rates` come from
   `boundary_head.null_projection`/`count_head` (`Linear(hidden,1)` on the
   query states, wired into `BoundaryModel::forward`; `None` when the flag is
   off, matching Python's `null_logits is None` gating).
5. **Overlap resolution — per query** (never across types; epic constraint)
   with `overlap_policy`. This checkpoint selects `flat`; `nested`/`longest`
   still fail at config load (`check_supported`, the Task 3 gate kept — the
   checkpoint uses `flat`, so porting the other policies was not required).
6. **Coordinate mapping** — half-open `[start,end)` word interval →
   caller-text **byte** range via `TextMap::candidate_bytes`
   (`start_mappings[start]` .. `end_mappings[end-1]`, Task 4); the span
   path's `(start,width)`/inclusive-end decode is never used. Candidates with
   `end <= start`, `end > n_words`, or whose mapped code-point range reaches
   into the synthetic `'.'` suffix are rejected **in full** (see "Suffix
   divergence" below). Surviving surfaces are sliced from the **caller's
   exact input** (not the normalized text) and trimmed; empty surfaces are
   dropped. Every returned `ExtractedEntity` carries byte offsets into the
   caller's input (`char_start` inclusive, `char_end` exclusive), re-verified
   by slicing the caller's string in the acceptance tests.

`ExtractedEntity` output order matches Python's `final_output`: entity types
in **declared** (query/marker) order, and within a type the overlap
resolver's rank order — descending confidence, then ascending start/end
(e.g. oracle 05: `Bob` before `Alice`; oracle 02: `Jane Doe` before
`John Smith`).

### `flat` overlap policy (as implemented)

`resolve_flat` = `overlap.py::resolve_overlaps` canonical `flat` (= the
`disallow` alias), run independently per query over the thresholded
`(p, start, end)` list:

1. rank key `(-p, start, end, input_index)` — a total order, fully
   deterministic;
2. **exact-span dedup**: each distinct `(start,end)` collapses to its
   highest-ranked copy;
3. **maximum-total-score weighted interval scheduling** (dynamic program over
   `by_end` — ascending `end`, `start`, `-p`, input index; predecessor is the
   last span with `end_j <= start_i`; totals accumulate in f64 exactly like
   Python's float sums of the f32 scores);
4. ties (equal totals) prefer the larger set, then the lexicographically
   better selection under the rank key — Python's exact branches, so
   equal-score crossings resolve to the lower start deterministically
   (asserted across reruns);
5. the selected set is re-ranked by the rank key (descending confidence, then
   ascending start/end) — this is the emitted item order.

This is per-type maximum-total-score selection and deliberately differs from
the span path's **global greedy** highest-confidence suppression
(`src/inference.rs`, unchanged): boundary types resolve independently, span
types suppress across all types.

### Suffix divergence (user-facing difference from Python, epic §1)

Python decodes against the **normalized** text (caller input + synthetic
`'.'`) and can surface that punctuation: for `"Alice greeted Bob"` a winning
candidate covering the synthetic word would print `"Bob."`. Rust must never
return a surface or byte range outside the caller's input:

- a candidate whose mapped range reaches into the synthetic suffix is dropped
  in full (not trimmed) — tested on oracle case 06
  (`suffix_rejection_case_06`) and empty-text case 10 (normalized `"."`, so
  every candidate is rejected);
- because overlap resolution runs **before** mapping (as in Python), a
  suffix-crossing span can also *shadow* a contained clean span — the WIS
  keeps the crossing span, then the mapping rejects it, and Rust emits
  nothing where Python emits `"Cupertino."`. The ordering is intentional
  (identical selection math to Python) and this is the documented user-facing
  difference from Python's normalized output.

The rule is inert for the oracle corpus at threshold 0.5 (no fixture entity
touches the suffix — asserted in `task7_final_entities_match_python`), so
final entities match Python exactly there.

### Oracle comparison and observed tolerances (Task 7 acceptance)

`cargo test --release` compares against the Python oracle: intermediates over
all 18 cases, final entities over the 13 entity cases (01–11, 13, 18).
**Final entities match exactly** on labels, surfaces, item order, and byte
offsets (fixture code-point offsets converted to UTF-8 bytes on the caller's
text); confidence stays within the stated **`< 1e-4` abs** tolerance:

| Comparison | max abs diff |
|---|---|
| `final_output` confidences (32 entities, 13 cases) | 5.4e-7 |
| `thresholded_candidates` probabilities (all 18 cases; span indices + pool-row order exact) | 1.7e-6 |
| `null_logits` (all 18 cases) | 8.1e-6 |
| `count_log_rates` (all 18 cases) | 3.5e-6 |

Observed 2026-09-29 (`cargo test --release task7 -- --nocapture`); residual is
the Task 5/6 fp32 logit noise propagating through the sigmoid. Threshold
decisions are unaffected on this corpus (nearest kept probability to 0.5 is
0.503, oracle 02).

Focused Rust checks (`cargo test`): `src/boundary_decode.rs` (no model
needed) — pair_temperature before sigmoid (asserts against both wrong
orders), threshold inclusive at equality, abstention (strict `>`; disabled
flag never suppresses), adaptive fill (union semantics, clamp to `C`,
ineligible never added, ties-to-even `round(exp(rate))`, clear error without
count head), WIS optimality on synthetic overlap cases (adjacent pair beats
one larger span; nested outer wins when it is the best total; crossing ties
resolve deterministically; disjoint chains kept), exact-span dedup to the
highest-ranked copy, suffix rejection case 06 (drop + clean mapping +
shadowing divergence), empty-text case 10; `src/boundary.rs` —
`task7_null_count_and_thresholded_candidates_match_python`,
`task7_final_entities_match_python`, and
`task7_predict_entities_matches_decode` (also exercises the
`Model::predict_entities` dispatch).

Span regression: the README example on `./models/gliner2-large-v1` is
byte-identical before/after (the span path, including `src/inference.rs`, is
untouched by Task 7).

## Boundary relation extraction (epic Task 8)

`src/boundary_rel.rs` ports the relation path at the pinned commit:
`gliner2/models/boundary/relations.py` (`TypedRelationPairGenerator` +
`SparseRelationScorer`) and `gliner2/models/boundary/engine.py`
(`_decode_relations` + `_deduplicate_relation_edges`). Entry points: pure
slice-based helpers (`generate_relation_pairs`, `deduplicate_relation_edges`,
`decode_relations_from_pairs`), the tensor scorer
(`SparseRelationScorer::load`/`forward`), and the model adapters
`BoundaryModel::propose_relations` / `decode_relations` / `predict_relations`
reached through the shared dispatch `Model::predict_relations` (Task 3's
prediction operation). `gliner2/inference/schema.py::Schema.relations` schema
construction and the role query-state routing (`concat(head,tail)` when
`directional_relation_states`) were ported in Task 4 and are reused unchanged
(`BoundaryPrepared::relation_role_routing`, oracle-checked on cases 12–18).

### Pipeline (as implemented)

1. **Typed/capped pair generation** (`TypedRelationPairGenerator.generate`,
   decode path) over the Task 6 shared candidate-pool rows — never an all-pairs
   entity grid. Per relation type, head mentions come from the head role
   query's pair logits and tail mentions from the tail role query's (the same
   pool row under a role query is one mention). Argument probabilities are
   `sigmoid(candidates.pair_logits)` — **without** `pair_temperature` (that
   temperature is entity-decode only) — gated by
   `relation_argument_proposal_threshold` (`>=`, inclusive). Capped ranking at
   `relation_heads_per_type` / `relation_tails_per_type`: probability
   descending, ties by `(span_start, span_end, flat_index)` ascending (Python's
   stable secondary sorts). Pairs are the capped cross product ranked by
   `head_prob × tail_prob` descending with `(head_slot, tail_slot)` ascending
   ties, capped at `relation_pair_cap`; identical head/tail spans are excluded
   (`allow_self=false` — Python `RelationTypeSpec` default for every
   schema-built spec). Output is relation-major (relation-index order), then
   pair-score order within a type — Python's `[B, R, pair_cap]` batch order.
2. **Relation scoring** (`SparseRelationScorer`): text states
   (`core["text_states"]`) gathered at each argument's start and `end-1`; the
   feature vector is the four endpoint states + the relation query state
   (`concat(head,tail)` when `directional_relation_states`, else `mean`) +
   `sign(tail_start - head_start)` + `|tail_start - head_start| / L`, through
   `mlp = (Linear, GELU, Dropout, Linear)` (Sequential indices 0/3; dropout
   inert at inference). With `relation_biaffine_content` (this checkpoint: on)
   the score additionally gets the biaffine content term: fp32 mean-pooled
   argument contents (`prefix[end] - prefix[start]) / max(len, 1)`), projected
   by `head_/tail_content_projection`, gated by `sigmoid(relation_content_gate
   (rel))`, `(head · gate · tail) / sqrt(H)`, plus `content_linear(2H + rq, 1)`.
3. **Decode** (`_decode_relations`): raw logits are divided by
   `relation_temperature` **before** the sigmoid (like `pair_temperature` for
   entities), `score < threshold` drops (inclusive at equality), both half-open
   argument spans map to caller-text byte offsets through
   `TextMap::candidate_bytes` (`start_mappings[start]`..`end_mappings[end-1]`,
   suffix-reaching candidates rejected in full — same rules as entities), and
   per-relation-type edges go through `_deduplicate_relation_edges`:
   contained-mention upgrade to the largest containing mention, exact
   `(head, tail)` dedup keeping the best score, semantic collapse on
   case-folded surfaces ranked by `(mention_distance, -score, head_start,
   tail_start)` — mention distance **before** score (Task 2's surprise) — and
   strict-subset dominance. Output order: relation types in first-surviving-pair
   order (= declared order for the relation-major proposals), then
   `(head_start, tail_start, -score)` within a type. Direction is preserved:
   the head is always the head-role argument even when its mention follows the
   tail's in the text (case 15). Relation arguments are **never** filtered
   through the entity decoder's threshold, abstention, or overlap policy.

### Result shape

`boundary_rel::ExtractedRelation` = `{ relation_type, head, tail, confidence }`
with `RelationArgument { text, char_start, char_end }` per argument: directed
head/tail trimmed surfaces plus half-open **byte** offsets into the caller's
exact input (untrimmed mapped ranges, same convention as `ExtractedEntity`),
and the shared edge confidence `sigmoid(raw_logit / relation_temperature)`
(Python attaches the same value to both arguments). Surfaces are re-verified by
slicing the caller's string in the acceptance tests.

### Config flags respected

`enable_relations`, `relation_heads_per_type`, `relation_tails_per_type`,
`relation_pair_cap`, `relation_argument_proposal_threshold`,
`directional_relation_states` (concat vs mean relation query), and
`relation_biaffine_content` all change the math and are implemented;
`relation_temperature` calibrates the decode. All enabled relation modes of the
flag matrix are ported — nothing is silently ignored. Relation extraction
requires the **shared** candidate pool (the pair generator consumes its rows +
per-role-query pair logits); a `per_query` checkpoint errors clearly at request
time, as does a checkpoint with `enable_relations=false` / missing
`relation_scorer` weights (see fail-clearly below).

### Fail-clearly behavior (no fake negatives)

Requesting relations on a checkpoint that cannot produce them errors instead of
returning an empty result that would look like a valid negative prediction:

- span architecture (`Model::predict_relations` on `Model::Span`) →
  "relation extraction is not supported by the span architecture …";
- boundary checkpoint with `enable_relations=false` (or no `relation_scorer`) →
  "relation extraction is not available on this checkpoint …"
  (`BoundaryModel::check_relation_support`, asserted by flipping the config
  flag on the real checkpoint);
- `per_query` candidate pool + relations → clear "only implemented for
  candidate_pool=\"shared\"" error;
- empty relation-type list → clear "requires at least one relation type" error.

Missing `relation_scorer` tensors with `enable_relations=true` already fail at
load (`expected_boundary_weights` / `validate_boundary_weights`).

### Oracle comparison and observed tolerances (Task 8 acceptance)

Relation cases 12–18 (relation-only, combined, direction fwd/rev, duplicate
mentions, no-relation, multi-type). Tolerance **`< 1e-4` abs (fp32 CPU vs
Python fp32)**; observed (`cargo test --release task8 -- --nocapture`,
2026-09-29):

| Comparison | result / max abs diff |
|---|---|
| `proposed_argument_pairs` (cases 12–18) | exact: head/tail word indices, `head_query_id`/`tail_query_id`, relation types; arg probs **1.9e-6** |
| `relation_logits` (raw scorer, 12 proposed pairs) | **3.8e-6** |
| `final_output.relation_extraction` (8 edges) | exact: types, direction, surfaces, code-point→byte offsets; confidence **8.9e-7** |

Case 16's contained-mention upgrade matches Python exactly (the emitted head is
the containing "Steve Jobs returned" at the upgraded mention's own score); case
17 is deliberately near-threshold (conf ≈ 0.5106 vs 0.5) and is asserted to
flip under a 0.52 threshold; cases 14 vs 15 assert semantic direction
preservation. The relation query-state vectors are not captured by the oracle
(shapes only), so the `concat(head,tail)` construction is validated end-to-end
through the raw relation logits above.

Focused Rust checks (`cargo test`): `src/boundary_rel.rs` — pair generator caps
(pair/head/tail), tie order (argument `(start,end,flat)`, pair
`(head_slot,tail_slot)` grid), identical head/tail exclusion (+ `allow_self`),
inclusive proposal threshold, raw-logit argument selection (asserts a
`pair_temperature`-scaled gate would drop the pair that raw sigmoid keeps),
scorer endpoint gathers (`start`/`end-1`), biaffine term presence/additivity
(hand-built weights) and `relation_temperature` before sigmoid (both wrong
orders rejected), edge dedup (contained-mention upgrade case-16 pattern, exact
dedup to best score, semantic distance-before-score ranking); `src/boundary.rs`
— `task8_proposed_pairs_and_relation_logits_match_python`,
`task8_final_relations_match_python`, `task8_direction_is_preserved`,
`task8_no_relation_case_is_near_threshold_sensitive`,
`task8_predict_relations_and_fail_clearly` (relation-only end-to-end incl. the
`Model::predict_relations` dispatch + all fail-clearly paths).

Span regression: the README example on `./models/gliner2-large-v1` is
byte-identical before/after (the span path is untouched by Task 8).

## User CLI & agent verification cases (epic Task 9)

`src/main.rs` is the user-facing CLI **and** the agent verification harness:
one-off extraction, machine-readable JSON, and a versioned case-file mode all
run through the **same** production inference path
(`Model::predict_extractions` = `BoundaryModel::preprocess` → `forward` →
`decode_entities` + `decode_relations`, or `SpanModel::predict_text`).
Combined entity+relation requests decode both from **one** forward over the
full schema (Python's single `extract()` encoding), so combined results match
the Task 7/8 oracle exactly instead of encoding entities and relations in
separate passes.

### CLI surface

| Flag | Behavior |
|---|---|
| `--model-dir DIR` | Local checkpoint dir (the four required files). Takes precedence over `--model-id`, never downloads |
| `--model-id ID` | Hub model ID to download/reuse (default `fastino/gliner2-large-v1`) |
| `--revision REV` | Hub revision (sha/tag/branch) for `--model-id`; defaults to the pinned revision below |
| `--models-root DIR` | Root of the model cache (default `./models`) |
| `--text TEXT` | One-off input text (required in one-off mode; the empty string is a valid input — zero words, empty extraction — matching case-file semantics and oracle case 10) |
| `--entities A,B` | Comma-separated entity types |
| `--relations A,B` | Comma-separated relation types (boundary checkpoints) |
| `--threshold F` | Score threshold `[0.0, 1.0]` (default 0.5; cases carry their own) |
| `--cases FILE` | Versioned case-file mode (mutually exclusive with `--text`/`--entities`/`--relations`/`--threshold`) |
| `--json` | Stable machine-readable JSON on stdout (one-off results or case report) |
| `--list-weights [N]` | Debug weight inventory (no text needed) |
| `--device` | `cpu` / `cuda:N` |

Entity-only, relation-only, and combined requests are all valid. Readable
output prints the detected architecture and active tasks, then each entity
(type, confidence, exact text, half-open byte offsets) and each relation
(type, confidence, directed head/tail text and offsets) — or an explicit
`No entities/relations found above threshold X.` line. All offsets are
half-open UTF-8 **byte** offsets into the caller's `--text`. Task-8's
fail-clearly errors (relations on a span checkpoint, `enable_relations=false`,
`per_query` pool, empty relation list) surface as clean CLI errors
(`error: …`, exit 1) before any output via `Model::check_relation_support`.

### Download/cache contract (`src/hub.rs`)

- **ID → directory:** `<models-root>/<repo-name>`, e.g.
  `fastino/gliner2.5-small-v1` → `./models/gliner2.5-small-v1/`. One
  directory = one checkpoint's `config.json`, `tokenizer.json`,
  `encoder_config/config.json`, `model.safetensors`.
- **Revision pinning:** `--revision` wins; otherwise the documented pinned
  default — `fastino/gliner2.5-small-v1` → `7132dc4561c3f94563c6147e75ffa8ef34c4964a`,
  `fastino/gliner2-large-v1` → `5312584a6fd5543e457ba5f309ac5db226431d1a`
  (the revision the regression/oracle results were captured against, *not*
  the moving Hub `main`). Unknown IDs without `--revision` resolve `main` to
  a commit sha at download time and record it — later runs stay on that sha.
- **Cache entry:** `.gliner2-cache.json` in the model directory records
  `cache_version`, `model_id`, the resolved `revision` (commit sha), `origin`
  (`downloaded` / `adopted`), and per-file identity `{size, sha256}`.
- **Atomicity:** each file streams to a temp name and is renamed only after a
  full write + fsync; the cache entry is written **last**. An interrupted
  fetch leaves no cache entry and is rejected (below), never mistaken for a
  valid checkpoint. A failed run cleans up the files it downloaded.
- **Reuse verification:** before reuse, the entry's `model_id` and `revision`
  must match the request (explicit `--revision`, or the pinned default), then
  every required file is re-hashed (sha256, `sha2` `asm` ≈ 2.2 GB/s) and
  compared against the recorded identity. Truncated, missing, or modified
  files error with instructions; files from two IDs are never mixed and a
  different checkpoint is never silently overwritten (`refusing to mix
  checkpoints from different IDs`).
- **Adoption:** a complete pre-existing directory without a cache entry
  (e.g. manual `hf download`) is adopted on first `--model-id` use — its file
  identity is hashed and recorded so later runs verify it. A *partial*
  directory without an entry is rejected (`interrupted download … remove the
  directory and retry`).
- `HF_ENDPOINT` / `HF_TOKEN` are honored for the Hub base URL / auth.

Verified 2026-09-29: adoption + reuse on both pinned dirs (the adopted
small-v1 `model.safetensors` sha256 `4ee98278…55de2b` matches the Hub LFS
etag recorded in "Checkpoint inventory"); fresh downloads of **both** model
IDs into empty `--models-root` dirs (metadata `origin: "downloaded"` at the
pinned shas) with correct inference afterwards; reuse on those; truncated
weights → size-mismatch error; missing file → incomplete-cache error;
metadata-less partial dir → interrupted-download error; same-name different
org ID → never-mix error; `--revision` mismatch → revision error; unknown
ref → resolve error. Manual `hf download` is not a prerequisite anywhere.

### JSON output (`--json`)

Stable field names and struct ordering (agents diff without parsing text);
stdout is exactly one JSON document (logs go to stderr).

```json
{
  "schema_version": 1,
  "kind": "extraction",
  "model": {"id": "…|null", "revision": "…|null", "dir": "…", "architecture": "span|boundary"},
  "tasks": ["entities", "relations"],
  "threshold": 0.5,
  "entity_types": ["person"],
  "relation_types": ["founder"],
  "entities": [{"type": "…", "text": "…", "byte_start": 0, "byte_end": 5, "confidence": 0.99}],
  "relations": [{"type": "…", "confidence": 0.9,
                 "head": {"text": "…", "byte_start": 0, "byte_end": 10},
                 "tail": {"text": "…", "byte_start": 19, "byte_end": 24}}]
}
```

Requested-but-empty tasks yield `[]` (readable mode prints the explicit
"No … found above threshold" line). `id`/`revision` are `null` for a
`--model-dir` without cache metadata. `kind: "case_report"` is the case-mode
document: `cases: [{name, status: "pass"|"fail", mismatches: [...], entities,
relations}]`, `passed`, `failed`, `total` (actuals included per case for
drift diagnosis).

### Case-file format (versioned JSON, `src/cases.rs`)

JSON chosen over JSONL (single versioned document, strict schema). Top level:
`version` (must be `1`; unknown fields rejected), optional `model_id`
(checked against the run's identity when known), `cases: [...]`. Per case:

- `name` (optional, for reports), `text` (required, may be empty),
- `entities` / `relations` (string lists; at least one non-empty),
- `threshold` (optional, default 0.5),
- `expect_entities: [{type, text, byte_start, byte_end, confidence?}]`,
- `expect_relations: [{type, head: {text, byte_start, byte_end},
  tail: {…}, confidence?}]`,
- `confidence: {min?, max?}` optional bounds.

Expected offsets are half-open UTF-8 byte offsets into `text` and must slice
exactly to the expected `text` (validated at load — an inconsistent file is a
clean error, not a mysterious failure). Matching is **strict set equality**
on (type, text, byte offsets) plus confidence bounds on each matched pair:
missing expected items, unexpected extra items, and out-of-bounds
confidences each fail the case with an identifying line. The report is
concise per-case PASS/FAIL; exit is nonzero if any case fails.

Committed file `cases/cli-cases.json` (target `fastino/gliner2.5-small-v1`,
values from the Task 2 oracle): positive — `readme-entities` (oracle 01),
`adjacent-entities` (02), `combined-founder` (13), `relation-only-forward`
(14), `relation-head-after-tail` (15, direction preserved), `unicode-byte-offsets`
(08, non-ASCII byte offsets), `multi-type-combined` (18); negative —
`empty-text-negative` (10), `founder-flips-above-0.52` (17 at 0.52, the
documented flip); ambiguous — `near-threshold-founder` (17 at 0.5,
conf ≈ 0.5106 pinned to `[0.50, 0.53]`). All 10 pass end-to-end through the
CLI (`--model-dir` and `--model-id`); a deliberately wrong expected result
exits nonzero with `missing`/`unexpected`/`confidence out of bounds` lines
(verified), and a case file whose expected `text` contradicts its offsets is
rejected at load.

## Python oracle (epic Task 2)

`oracle/` holds the captured Python behavioral oracle for the boundary port:
18 fixed-corpus cases (`oracle/cases/NN_*.json`) run through the production
`extract()` path of the pinned GLiNER2 checkout on
`./models/gliner2.5-small-v1/`, plus `oracle/manifest.json` (checkpoint
identity, migrated config/active flags, conventions, case index) and the
reproducible capture script `oracle/capture_oracle.py` (its sanity checks must
pass before fixtures are written). Per case it records token IDs, first-subtoken
and query-marker positions, split-word offsets, normalized encoding text,
boundary/candidate shapes, selected candidate indices, raw candidate
proposal/compat/pair logits, thresholded candidates, null/count logits,
relation role routing, proposed argument pairs, raw relation logits, and final
entities/edges — enough for numerical/indexing Rust parity checks. Offsets are
Python code-point indices into the normalized text (input plus synthetic `.`);
see `oracle/README.md` and the manifest conventions before comparing. Epic
Task 5 additionally captured raw boundary encodings and marginals for the
short cases `01`/`05`/`10` (`raw_boundary`/`raw_core`) and one synthetic
padded batch (`oracle/synthetic_masks.json`) — see "Boundary encoder &
marginals" above.

Regenerate (Python 3.12 venv with `gliner2[local]` from the pinned checkout):

```bash
uv venv ~/.venvs/gliner2-oracle --python 3.12
VIRTUAL_ENV=~/.venvs/gliner2-oracle uv pip install -e "$HOME/Projects/GLiNER2[local]" protobuf
~/.venvs/gliner2-oracle/bin/python oracle/capture_oracle.py   # repo root
```

The script refuses to run against a different GLiNER2 commit or checkpoint
revision; captures are CPU-deterministic (byte-identical across reruns).

## Known limitations / next steps

- Classifications, JSON records, and JSON structures not yet implemented
  (entity and relation extraction only). The boundary architecture
  (`fastino/gliner2.5-small-v1`) **loads and validates** (config parse +
  active weight checks + dispatch; see "Architecture dispatch & config
  parsing" above), its **preprocessing and query routing are ported and
  oracle-checked** (epic Task 4; see "Boundary preprocessing & query routing"
  above), the **boundary encoder + start/end/inside marginals are ported and
  oracle-checked** (epic Task 5; see "Boundary encoder & marginals" above),
  the **shared candidate pool + pair scoring are ported and oracle-checked**
  (epic Task 6, exact indices/masks + `< 1e-4` logits on all 18 cases; see
  "Shared candidate pool & pair scoring" above), **entity extraction is
  ported and oracle-checked** (epic Task 7 — exact final entities on all 13
  entity cases; see "Boundary entity decoder" above), and **relation
  extraction is ported and oracle-checked** (epic Task 8 — exact proposed
  pairs and final directed edges on relation cases 12–18; see "Boundary
  relation extraction" above), and the **user CLI / agent verification
  harness is done** (epic Task 9 — `--relations`, JSON output, case files,
  `--model-id` download contract; see "User CLI & agent verification cases"
  above). The `per_query` candidate pool
  (`proposal.py`/`scoring.py`) is not ported: such a checkpoint loads and
  validates but scoring errors clearly. Boundary pooling/scoring/decode is
  batch-1 only, and the entity decoder supports only the `flat` overlap policy
  (`nested`/`longest` rejected at config load) with a single global threshold
  (Python's per-type schema thresholds are not exposed). Relation edge dedup's
  semantic key uses Rust `to_lowercase()` where Python uses `str.casefold()`
  (identical on ASCII; Python's aggressive full casefold — e.g. `ß` → `ss` —
  is not matched), and relation-only/combined requests share one threshold.
- `clf1`/`clf2` (binary span classifier) loaded but not wired into the scoring path
- Batch inference not implemented (single text at a time)
- No Metal / CUDA device tested yet (`--device cuda:0` flag exists)
- SpanRepLayer assumes markerV0 boundary-pair strategy; verify against gliner source if results look off
