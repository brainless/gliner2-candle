# Python oracle — GLiNER 2.5 boundary (epic Task 2)

Captured behavioral oracle for porting boundary-entity and relation inference
to Rust. One fixed corpus run through the **production** extraction path of the
pinned Python reference with the local 2.5 checkpoint; raw intermediates are
recorded for numerical/indexing comparison, not just final strings.

- Python reference: `~/Projects/GLiNER2` @ `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`
- Checkpoint: `models/gliner2.5-small-v1/` (Hub `fastino/gliner2.5-small-v1` @
  `7132dc4561c3f94563c6147e75ffa8ef34c4964a`; weights are **not** captured)
- API under test: `AutoExtractor.from_pretrained(...).extract(text, schema,
  threshold=0.5, include_confidence=True, include_spans=True)` in eval mode,
  entity/relation type order fixed as declared per case.

## Layout

| Path | Contents |
|---|---|
| `capture_oracle.py` | capture script (also runs sanity checks; fails loudly instead of saving bad fixtures) |
| `manifest.json` | checkpoint identity, migrated config + active flags, environment versions, conventions, case index |
| `cases/NN_*.json` | one fixture per corpus case (fields below) |
| `synthetic_masks.json` | synthetic padded `B=2` batch for boundary-encoder mask/window/EOS behavior (epic Task 5; inputs from a recorded fp32 lattice formula) |

Per case: `text`, `normalized_text` (input plus synthetic `.` when needed),
`token_ids`, `words` + `word_char_spans` (split-word offsets), 
`first_subtoken_positions`, `query_marker_positions`, `schema_marker_positions`/
`schema_tokens`, `query_layout`, `relation_role_routing`, `shapes` (boundary
marginals + candidate tensors), `selected_candidate_indices`,
`candidate_valid_mask`, `candidate_proposal_logits`, `candidate_compat_logits`,
`candidate_pair_logits` (raw, pre-temperature), `thresholded_candidates`,
`null_logits`, `count_log_rates`, `proposed_argument_pairs`, `relation_logits`
(raw, pre-temperature), `final_output` (public entities + relation edges), and
`sanity_checks`. See `manifest.json → conventions` for the index/offset
contract (code-point offsets into `normalized_text`, half-open `[start,end)`
word boundaries, shared query-agnostic candidate pool).

Epic Task 5 raw dumps (regenerated into the same files): cases `01`, `05`, and
`10` carry `raw_boundary` — boundary states `[L+1,d]` + mask, start/end
logits `[Q,L+1]`, inside logits `[Q,L]`, `inside_prefix` `[Q,L+1]`,
`inside_prefix_mean` `[Q]`, and `interval_scores` rows
`[batch,query,start,end,value]` (Python `interval_prefix_score`) — and case
`01` additionally carries `raw_core` (gathered `text_states`/`query_states`)
and the `after_layer_norm` / `after_attention_0` encoder stages.
`synthetic_masks.json` covers what the B=1 corpus cannot: per-sample EOS
placement, boundary/token/query masks, `MASK_LOGIT` fills, padding zeroing,
the centered fp32 inside prefix over masked tokens, and the attention window
(production `window=128` is inert at `N≤4`, so attention block 0 is also
re-run with `window=2`).

## Regenerate

```bash
# one-time env (Python 3.12; installs the pinned checkout in editable mode)
uv venv ~/.venvs/gliner2-oracle --python 3.12
VIRTUAL_ENV=~/.venvs/gliner2-oracle uv pip install -e "$HOME/Projects/GLiNER2[local]" protobuf

# capture (from the gliner2-candle repo root)
~/.venvs/gliner2-oracle/bin/python oracle/capture_oracle.py
```

The script refuses to run if `GLINER2_SRC` (default `~/Projects/GLiNER2`) is
not at the pinned commit or the local checkpoint revision differs. Captures are
deterministic (verified byte-identical across two runs on CPU).

## Corpus

Entities: `01_apple_readme` (README sentence), `02_adjacent_entities`,
`03_punctuation`, `04_multiword_entity`, `05_edges` (both text edges),
`06_no_terminal_punct` (appended-`.` suffix; controlled pair with 01),
`07_url_email`, `08_unicode` (diacritics + CJK), `09_overlap`,
`10_empty_text`, `11_truncated_text` (mid-entity truncation).
Relations: `12_relation_only`, `13_combined`, `14_direction_fwd` /
`15_direction_rev` (mention-order reversal), `16_duplicate_mentions`,
`17_no_relation`, `18_multi_type_combined` (two relation types).

## Observed behaviors worth knowing (captured faithfully)

- `01` and `06` produce **identical** results: `06` normalizes to `01`'s text.
- `05`/`06` pool contains suffix-crossing spans (e.g. `[2,4]` = `Bob.`) but none
  pass threshold 0.5 here; Python decodes against `normalized_text`, so a
  passing span would surface the synthetic `.`. Rust must reject mapped ranges
  extending into the suffix (epic §1) — that rule is inert for this corpus but
  is a real divergence guard.
- `07`: `WhitespaceTokenSplitter`'s URL pattern swallows trailing punctuation
  (`https://example.com/about;` is one word including `;`).
- `03`: decoded surfaces include adjacent punctuation (`Apple, Inc.`,
  `Cupertino, CA.` with the sentence period).
- `16`: `_deduplicate_relation_edges` upgrades a contained mention to the
  longest containing mention (`Steve Jobs` → `Steve Jobs returned`) and ranks
  candidate edges by mention distance before score; the final edge is not the
  highest-logit proposal (see `proposed_argument_pairs` vs `final_output`).
- `17`: "no relation" text yields a near-threshold **false positive** edge
  (Alice → Paris, logit 0.0424, confidence 0.51) — parity target, not an error.
- `10` (empty text) is accepted: normalized to `.`, one word, zero entities.
- Entity decode order is confidence-descending per type; relation `head`/`tail`
  confidence is the shared edge score.
