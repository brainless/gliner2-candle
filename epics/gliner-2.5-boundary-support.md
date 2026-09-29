# Epic: GLiNER 2.5 boundary model support

## Goal and scope

Load GLiNER 2.5 checkpoints whose top-level config says `"architecture": "boundary"`
and extract entities and relations through the Rust CLI. Keep the `fastino/gliner2-*`
span path working. Start with `fastino/gliner2.5-small-v1`; validate base and
multi separately before claiming support for them. Python dispatches 2.5 through
`AutoExtractor`; `GLiNER2.from_pretrained` is its legacy span loader. Rust can
keep its current CLI name while dispatching by architecture.

This epic covers **entity and relation extraction** for boundary checkpoints.
Classification, JSON records, training, batching, and GPU parity are separate
work. Relation support is a dependent milestone: entity candidates and their
coordinate mapping must be correct before relation pair generation is added.
The existing span path remains entity-only unless separately requested.

The binary in `src/main.rs` already provides a basic CLI with `--text`,
`--entities`, `--model-id` download via `hf-hub`, and human-readable entity
output. Extend it as a supported **user-facing CLI** whose primary development
role is an **agent verification harness**. Users should be able to run one-off
entity or relation extraction without fixture files or debug knowledge. Agents
should be able to run
arbitrary texts, inspect exact spans and scores, run a reproducible corpus,
and get a nonzero exit status when expected extractions do not match. Both
uses must call the same production inference path.

Use the local `~/Projects/GLiNER2` checkout as the Python behavioral reference
(reviewed at `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`) and
`~/Projects/candle` for Candle API research. The Candle checkout is 0.11.0;
this crate depends on Candle 0.9.2, so verify APIs and numerics against the
**resolved 0.9.2 dependency** before choosing an implementation. This document
records source-derived behavior, **not a verified 2.5 checkpoint inventory**.
Pin the GLiNER2 commit and checkpoint revision used for implementation, inspect
the real config and all safetensors keys/shapes, then correct any mismatch here
and in `DEVELOP.md`.

## Existing Rust constraints

- `src/model.rs` loads span-only heads eagerly. Dispatch before those weight
  lookups. Prefer separate span/boundary model structs behind an enum or common
  prediction interface; preserve the span forward path.
- `src/processor.rs` builds an entity schema and first-subtoken routing, but
  its splitter is **not exact Python parity**. Python's
  `WhitespaceTokenSplitter` recognizes URLs, emails, and `@mentions`.
  Python offsets index Unicode code points; the current Rust offsets are
  UTF-8 bytes. Audit shared changes with span regression checks, or introduce
  boundary-specific preprocessing. Convert to byte offsets for Rust slicing.
- `src/inference.rs` greedily suppresses overlaps across all types and joins
  split words with spaces. Boundary inference resolves candidates **per query**
  and slices the exact surface from the original text. Keep a separate decoder
  so existing span behavior does not change accidentally.
- `src/config.rs` defaults legacy configs to span. Reject unknown architectures
  and unsupported *active* boundary modes explicitly; silently ignoring a
  checkpoint flag can produce plausible but wrong entities. A checkpoint may
  include weights for optional modules that are inactive in the selected path;
  validate task requirements and selected math rather than treating every
  weight as an inference requirement. Enabled records and classification are
  outside this epic, so do not run or expose those tasks.
- The current `--model-id` implementation uses the `hf-hub` cache directory
  returned for `config.json`; it does not create a project-local model directory
  or record a pinned revision. It assumes a single `model.safetensors` file.
  The model-specific cache contract below requires an explicit change.
- Model directories must be unambiguous: use `./models/gliner2-large-v1/`
  for the existing span checkpoint and `./models/gliner2.5-small-v1/` for
  the first boundary checkpoint. A directory contains one checkpoint's
  config, encoder config, tokenizer, and weights. Never download one model
  into another model's directory.

## Source-checked inference contract

Read the cited files at the pinned source commit during implementation.

1. **Input and query routing.** `gliner2/processor.py` owns the actual
   `schema + [SEP_TEXT] + text` input, sentence-final punctuation normalization,
   word splitting, subword conversion, and marker/word routing.
   `BoundaryExtractorModel._encode_core` in
   `gliner2/models/boundary/model.py` gathers first-subtoken text states and
   each `[E]` marker's contextual state as a query.
   `gliner2/processing/boundary_preprocessing.py` mainly builds query layouts
   and optional gold targets; it is **not** an alternate tokenizer. Entity
   queries use `[E]` markers; relation schemas contribute `[R]` head/tail
   role queries in transformed schema order. Keep output types aligned with
   their marker order. Match the request's `max_len` and the config's
   `token_pooling` or fail clearly if unsupported. Python's
   `SchemaTransformer._collate_batch` appends `.` to input without terminal
   `.`, `!`, or `?` **before** splitting and stores that normalized text for
   decoding. Match the normalized token sequence for model parity, retain the
   caller's exact input separately for CLI byte offsets. Reject a candidate
   whose mapped range extends into the synthetic suffix; test this case and
   document the resulting user-facing difference from Python's normalized
   output. Do not return a surface or byte range outside the caller's input.
2. **Coordinates.** `boundary/model.py` and `proposal.py` use half-open
   word-token intervals `[start,end)`: `L` text words yield `L+1` boundaries
   and `0 <= start < end <= L`. Python's target builder converts inclusive
   training labels, but inference candidates are already half-open.
   `gliner2/inference/candidate_decoder.py` maps to characters via
   `start_mappings[start]` and `end_mappings[end-1]`. Never reuse the span
   path's `(start,width)` decode or inclusive end index.
3. **Boundary encoding.** `boundary/encoding.py` combines left token/BOS and
   right token/EOS, projects each side, concatenates, projects, normalizes,
   and optionally applies pre-norm local self-attention and residual SwiGLU
   refinement. Respect configured counts/window and masks. Dropout is off at
   inference. The final valid boundary is EOS, including after truncation.
4. **Marginals.** `boundary/heads.py` creates start/end logits
   `[B,Q,L+1]` and inside logits `[B,Q,L]` with scaled dot products. Invalid
   positions use finite `MASK_LOGIT=-1e4`. The inside prefix is fp32 and
   centered by a per-query mean; interval scoring must restore
   `mean * (end-start)`. Omitting this changes scores.
5. **Proposals.** `boundary/proposal.py` selects high-marginal boundaries,
   scores conditional pairs in blocks or one full-width vectorized block,
   filters `end > start`, deduplicates, and retains up to `candidate_budget`
   per query. `boundary_top_k_alpha/max/bucket`, `export_mode`, and
   `bidirectional_proposals` affect selection. Proposer projections and gate
   are distinct learned weights from the reranker's endpoint projections.
   Selection uses full `compat + start + end`; reranking receives the
   **marginal-free** `compat_logits` prior.
6. **Pair score.** `boundary/scoring.py` adds reranker endpoint compatibility,
   start and end marginals **once**, proposal compatibility prior, optional
   content and endpoint-difference terms, inside interval evidence divided by
   `sqrt(length)`, and query-weighted `log1p(length)`,
   `length/text_length`, and `1/sqrt(length)` features. Active flags can
   change math and weight shapes: `enable_span_content`,
   `content_soft_max_pool`, `query_conditioned_inside_weight`,
   `endpoint_difference_features`, `reranker_endpoint_compat`,
   `multihead_pair_compat_heads`, and `enable_rotary_endpoints`.
   Support each enabled mode in the checkpoint or fail loading precisely.
   `boundary/rotary.py` rotates interleaved even/odd pairs in fp32 at
   boundary positions for both proposer and scorer. Candle's local 0.11
   `candle-nn/src/rotary_emb.rs::rope_i` requires a contiguous four-dimensional
   input and precomputed cosine/sine tensors; it is not a drop-in call for
   arbitrary gathered endpoints. Compare the resolved 0.9.2 API, shape,
   frequency, position indexing, broadcast, and dtype semantics before use.
7. **Entity decode.** `boundary/engine.py` divides pair logits by
   `pair_temperature` **before sigmoid**, thresholds valid candidates,
   optionally fills to `round(exp(count_log_rate))` when
   `adaptive_threshold` is set, and suppresses a query when sigmoid of its
   null logit exceeds `abstention_threshold`. The default `flat` policy in
   `gliner2/inference/overlap.py` deduplicates exact spans and performs
   **maximum-total-score weighted interval scheduling** per entity type.
   This differs from Rust's current global greedy suppression. Match other
   configured policies (`nested`, `longest`) or reject them explicitly.
   Extract the exact surface from the original string and return the existing
   `ExtractedEntity` shape with confidence and byte offsets.
8. **Relations.** `gliner2/inference/schema.py::Schema.relations` builds a
   relation schema with `head` and `tail` roles for each requested relation
   type. `boundary/model.py::_encode_core` makes a relation query state from
   those two role marker states (mean by default, concatenation when
   `directional_relation_states` is enabled). `boundary/relations.py`
   generates typed, capped head/tail candidate pairs from the boundary
   candidate logits; it does not run an all-pairs entity grid. Argument
   selection uses `sigmoid(candidates.pair_logits)` **without**
   `pair_temperature` and applies `relation_argument_proposal_threshold`
   before capped head/tail ranking and pair ranking. Preserve its stable tie
   order, default exclusion of identical head/tail spans, and pair cap. Its
   scorer gathers text states at each argument's start and `end-1`, uses relative
   position features and an MLP, with optional biaffine content.
   `boundary/engine.py::_decode_relations` applies
   `relation_temperature` before sigmoid, thresholds, converts both
   half-open argument spans to character offsets, and deduplicates overlapping
   or repeated-mention edges with `_deduplicate_relation_edges`.
   Respect `enable_relations`, `relation_heads_per_type`,
   `relation_tails_per_type`, `relation_pair_cap`,
   `relation_argument_proposal_threshold`,
   `directional_relation_states`, and `relation_biaffine_content`.
   Relation-only requests must still construct the two role queries and
   candidate spans needed by the scorer; they need not ask for an unrelated
   entity schema. Preserve relation direction in output. Inference must not
   first filter arguments through the entity decoder's threshold, abstention,
   or overlap policy.

`BoundaryHead.forward` in `boundary/model.py` selects either
`candidate_pool == "per_query"` (proposer + pair scorer) or `"shared"`
(`boundary/pool.py` builder and scorer). Never run the `per_query` path for a
`shared` checkpoint. The defaults in
`gliner2/configuration.py` do not replace the real serialized config,
especially around older `config_version` migration. The model constructs both
pools' modules even when only one path runs; the presence of weight keys alone
does not identify the active path. Check `boundary_head.candidate_pool` after
Python's version-aware config migration.

## Implementation sequence and acceptance gates

### 1. Pin checkpoint and inventory

The repository's legacy span checkpoint is already in
`./models/gliner2-large-v1/`; reuse it for regression checks without
downloading it again. Obtain `fastino/gliner2.5-small-v1` separately in
`./models/gliner2.5-small-v1/` for the inventory, then verify the CLI's
`--model-id` download/reuse path once boundary loading is implemented. Record
the Hub revision and local GLiNER2 commit for each checkpoint.
Inspect `config.json`, `encoder_config/config.json`, tokenizer files, and
**all** safetensors names and shapes. `GLiNER2::list_weight_keys` only lists
the first N keys; use a header reader or extend it for a complete inventory.
Record actual prefixes, shapes, and active inference flags in `DEVELOP.md`.
Account for any checkpoint index/shards if the downloaded artifact differs
from the current single-file loader. Verify the chosen source commit can
load the checkpoint, and distinguish optional modules present in the state
dict from modules executed by its config.
Confirm Candle's DeBERTa-v2 implementation can load this DeBERTa-v3 checkpoint,
including embeddings, relative position weights, and vocabulary size; an
architecture name alone does not establish numerical compatibility. Document
and implement any missing encoder prerequisite before claiming support.
Determine whether this checkpoint selects `per_query` or `shared`. Check that
relation extraction is actually enabled and relation scorer weights exist;
if not, entity support can proceed, but relation support for that checkpoint
cannot be claimed.

**Task 1 results (verified 2026-09-29, full inventory in `DEVELOP.md`
"Checkpoint inventory"):** `fastino/gliner2.5-small-v1` is pinned at Hub
revision `7132dc4561c3f94563c6147e75ffa8ef34c4964a` (single `model.safetensors`,
334 tensors, no shards) in `./models/gliner2.5-small-v1/`; the legacy span
checkpoint in `./models/gliner2-large-v1/` is revision
`5312584a6fd5543e457ba5f309ac5db226431d1a`. The pinned GLiNER2 commit
`55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` loads and runs the checkpoint
(entity + relation extraction verified). Corrections to expectations here:
the checkpoint selects **`candidate_pool: "shared"`** (not the `per_query`
default — Task 6 must port `pool.py`/`SharedPoolScorer` first; per_query
modules exist in the state dict but are inactive), relations **are** enabled
(`relation_scorer.*` present, `directional_relation_states` +
`relation_biaffine_content` on), `enable_records` is on with `record_decoder.*`
weights (out of scope — do not expose), the scoring/rotary flags
`enable_rotary_endpoints`, `reranker_endpoint_compat`,
`endpoint_difference_features`, `multihead_pair_compat_heads`,
`query_conditioned_inside_weight` affect only the inactive per_query path, and
the tokenizer is `tokenizer.json`-only (no `spm.model`; transformers-5-style
list-valued `extra_special_tokens` metadata). Candle 0.9.2's
`candle-transformers` DeBERTa-v2 loads this DeBERTa-v3 encoder with no missing
prerequisite (numerically matched the Python HF encoder to 4.5e-6 on a fixed
forward).

### 2. Capture a Python oracle before porting

With the pinned source and checkpoint, save config, token IDs, first-subtoken
positions, query marker positions, split-word offsets, normalized encoding text,
boundary/candidate shapes, selected candidate indices, candidate logits, and final entities for
a small fixed corpus. Include the README Apple sentence, adjacent entities,
punctuation, a multi-word entity, entities at both text edges, URL/email,
Unicode, and an overlap case. Use evaluation mode and fixed type order.
Add relation-only and combined entity/relation examples, including direction
reversal, duplicate mentions, and no relation. Capture relation role routing,
raw candidate logits, proposed argument pairs, relation logits, and final edges
as well. Include a text without terminal punctuation to test the appended
suffix and an empty or truncated text edge case if the checkpoint accepts it.
Keep fixtures compact; do not commit weights. Numerical/indexing comparisons
are required because plausible extracted text alone can hide port errors.

### 3. Config and architecture dispatch

Add an architecture enum to `src/config.rs`, defaulting missing values to
`span`. Parse active `boundary_head` fields plus `token_pooling`/`max_len`,
and validate dimensions as Python's `validate_boundary_head` does. Unknown
architectures or unsupported active modes must error. Keep span loading and
math behind the span variant and select before loading head weights. Expose
a shared prediction operation to the CLI; do not put boundary scoring
inside `SpanRepLayer` or `CountLSTM`.

### 4. Preprocessing and routing

Port the entity and relation schema behavior of `gliner2/processor.py` and
`_encode_core`, including relation role query routing and relation query state.
Compare Rust/Python token IDs, routing positions, normalized text, and offsets
against the oracle before adding boundary math. Reuse existing preprocessing
only where parity is established. Keep original input text for final slicing,
while matching Python's normalized encoding text. Reject a decoded span if
its mapped character end exceeds the caller's text. Convert surviving Python
code-point offsets to UTF-8 byte offsets and verify every returned surface by
slicing the caller's input.

### 5. Boundary encoder and marginals

Implement the active `encoding.py` and `heads.py` features. Compare
boundary 0 and L, masks, optional attention/refinement order, fp32 centered
prefix sums, and interval reconstruction against Python intermediates. Add
focused Rust checks for shape and one-token/edge intervals; this repo's lack
of a test directory is not a reason to skip meaningful numerical checks.

### 6. Proposal and pair scoring

Implement the checkpoint-selected candidate-pool path. For `per_query`, port
`proposal.py` and `scoring.py`, including adaptive top-k, bidirectional
selection, deterministic tie behavior, deduplication, marginal-free prior,
all active optional terms, and configured vectorized/streaming behavior. If
`shared`, port `pool.py` and its scorer instead. Bound memory by configured
top-k/budget rather than building an `L x L` span grid. Compare proposal
indices, valid masks, and pair logits with Python on short inputs. Explicitly
assert that marginals are added once.

### 7. Boundary entity decoder

Implement the entity decode contract above, including per-query overlap
resolution and exact original-text slicing.

### 8. Boundary relation extraction

Port `boundary/relations.py` and the relation-specific routing and decoding
described above. Confirm the checkpoint has relation weights and enables the
head before accepting a relation request. Define a Rust relation result with
type, directed head/tail text, half-open byte offsets, and confidence. Test
the typed/capped pair generator and optional checkpoint-enabled scorer terms
against Python intermediates, including raw-logit argument selection and
final edge deduplication. For span checkpoints or boundary checkpoints
without a relation head, requesting relations must fail clearly rather than
return an empty result that looks like a valid negative prediction.

### 9. User CLI and agent verification cases

Extend the **existing main binary**, rather than adding a separate test binary.
Keep `--model-dir`/`--model-id`, `--text`, `--entities`, and `--threshold`;
add `--relations` as comma-separated relation types and allow entity-only,
relation-only, or combined requests. Print detected architecture and active
tasks. In readable output, show every entity's type, confidence, exact text,
and half-open byte offsets; show every relation's type, confidence, directed
head/tail text and offsets. State explicitly when no extraction passes the
threshold. Keep `--help` clear and examples in `README.md` concise enough
for normal users. Validate missing text, empty type lists, threshold range,
model files, and unsupported tasks with actionable errors and nonzero exits.
The default one-off mode should need only text/type arguments and a model ID
or local path. `--model-id` must download the config, tokenizer, encoder
config, and weights automatically into a **model-specific directory** such as
`./models/gliner2.5-small-v1/`, reuse complete files on later runs, and give
a clear error if a file is unavailable. Define a stable mapping from model ID
to directory and verify the recorded checkpoint identity/revision before
reusing it; never mix files from two IDs or silently overwrite a different
checkpoint. Use an explicit revision option or a documented pinned default
for reproducible agent cases. Record resolved revision and file identity with
the cache entry, and complete downloads atomically so an interrupted fetch
cannot be mistaken for a valid checkpoint. `--model-dir` uses already
downloaded local files and takes precedence; it must not trigger a download.
Verify both paths for span and boundary checkpoints. Do not make manual `hf download` a prerequisite for
users or agent checks.

Add a machine-readable JSON output mode with stable field names and ordering
so agents can inspect or diff results without parsing display text. Add a
versioned case-file mode (JSON or JSONL; choose and document one) that runs
multiple texts **sequentially** through the same inference path. Each case
specifies text, requested entity/relation types, threshold, and expected
label/type plus exact text and offsets; optional confidence bounds can catch
large numerical drift without requiring bitwise equality. Print a concise
per-case pass/fail report, identify mismatches, and exit nonzero if any case
fails. Include positive, negative, and ambiguous examples; do not treat
`binary exited 0` or merely plausible output as a passing test.

### 10. End-to-end verification and docs

Compare Rust with the pinned Python oracle on token routing, candidate pairs,
candidate and relation logits within a stated numerical tolerance, and final
entities/edges with exact labels, surfaces, directions, and offsets. Check
each Rust byte offset by slicing the original UTF-8 string. Run the agent
case file through the CLI and verify its nonzero failure exit with one
deliberately wrong expected result. Run the existing span README example on
`./models/gliner2-large-v1/` before and after dispatch changes and compare
outputs. Run `cargo fmt --check`, `cargo check`, `cargo clippy --all-targets`,
and release inference for both architectures. If base or multi is claimed
supported, inventory and run each
checkpoint separately, including the multilingual splitter/pooling path.
Update `README.md` and `DEVELOP.md` with supported checkpoint IDs, CLI
examples for entity-only/relation-only/combined/case-file runs, real weight
keys, architecture diagrams, active flags, limitations, and oracle comparison.
Compile or plausible output alone does not complete the epic.
