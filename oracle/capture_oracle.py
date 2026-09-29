#!/usr/bin/env python3
"""Capture a Python inference oracle for the GLiNER 2.5 boundary port (epic Task 2).

Runs a fixed corpus through the *production* extraction path of the pinned
Python reference (``~/Projects/GLiNER2`` @
``55656fbfa01d3d4a77485e1a1eeeaf682990ccdf``) with the local checkpoint
``./models/gliner2.5-small-v1`` and records raw intermediates that the Rust
port must match numerically/index-wise:

  config, token IDs, first-subtoken positions, query marker positions,
  split-word offsets, normalized encoding text, boundary/candidate shapes,
  selected candidate indices, candidate logits, final entities,
  relation role routing, proposed argument pairs, relation logits, final edges.

Intermediates are captured with forward hooks / thin wrappers around the
production modules (``_encode_core``, ``BoundaryHead.forward``,
``SharedPoolBuilder``/``_group_scored_candidates``,
``TypedRelationPairGenerator.generate``, ``SparseRelationScorer.forward``);
final entities/edges come from the public ``extract()`` result of the same run.
No weights are written. Boundary-encoder states and raw start/end/inside
marginals (epic Task 5) are dumped for the short cases 01/05/10 and for one
synthetic padded batch (``synthetic_masks.json``, mask/window/EOS behavior);
other cases keep shapes only.

Regenerate (from the gliner2-candle repo root):

    ~/.venvs/gliner2-oracle/bin/python oracle/capture_oracle.py

Environment / prerequisites (see oracle/README.md): Python 3.12 venv with
``gliner2[local]`` from the pinned checkout installed; the script itself pins
the checkout via ``GLINER2_SRC`` (default ``~/Projects/GLiNER2``) and refuses
to run if that checkout is not at the pinned commit.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# --------------------------------------------------------------------------
# Paths and pinned identities
# --------------------------------------------------------------------------

ORACLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORACLE_DIR.parent
MODEL_DIR = REPO_ROOT / "models" / "gliner2.5-small-v1"
CASES_DIR = ORACLE_DIR / "cases"

GLINER2_SRC = Path(
    os.environ.get("GLINER2_SRC", str(Path.home() / "Projects" / "GLiNER2"))
).resolve()
PINNED_GLINER2_COMMIT = "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf"
PINNED_CHECKPOINT_REVISION = "7132dc4561c3f94563c6147e75ffa8ef34c4964a"
CHECKPOINT_HUB_ID = "fastino/gliner2.5-small-v1"

THRESHOLD = 0.5

# Raw boundary-encoder / marginal values (epic Task 5 oracle) are dumped for
# these short cases only, keeping the other fixtures compact.
RAW_CASES = {"01_apple_readme", "05_edges", "10_empty_text"}

# Additionally dump gathered encoder states and per-stage boundary-encoder
# intermediates for the anchor case only (debug isolation of encoder/gather vs
# boundary-encoder math); the synthetic batch covers stage isolation generally.
RAW_DETAIL_CASES = {"01_apple_readme"}

# --------------------------------------------------------------------------
# Corpus. Entity/relation type order is FIXED: the declared list order defines
# the query order (entities first, then one head+tail role pair per relation
# type), which the fixtures record explicitly.
# --------------------------------------------------------------------------

CORPUS: List[Dict[str, Any]] = [
    {
        "id": "01_apple_readme",
        "task": "entities",
        "text": "Apple was founded by Steve Jobs in Cupertino.",
        "entity_types": ["person", "organization", "location"],
        "relation_types": [],
        "notes": "README quick-start sentence; baseline entity extraction.",
    },
    {
        "id": "02_adjacent_entities",
        "task": "entities",
        "text": "John Smith Jane Doe founded Acme Corp.",
        "entity_types": ["person", "organization"],
        "relation_types": [],
        "notes": "Two adjacent person mentions with no separator token.",
    },
    {
        "id": "03_punctuation",
        "task": "entities",
        "text": "Steve Jobs' Apple, Inc. is in Cupertino, CA.",
        "entity_types": ["person", "organization", "location"],
        "relation_types": [],
        "notes": "Apostrophe, commas and abbreviation periods around entities.",
    },
    {
        "id": "04_multiword_entity",
        "task": "entities",
        "text": "Barack Obama visited New York City last year.",
        "entity_types": ["person", "location"],
        "relation_types": [],
        "notes": "Multi-word entities (2 and 3 words).",
    },
    {
        "id": "05_edges",
        "task": "entities",
        "text": "Alice greeted Bob",
        "entity_types": ["person"],
        "relation_types": [],
        "notes": "Entities touch both text edges (char 0 and char end); "
        "also exercises the synthetic '.' suffix (no terminal punctuation).",
    },
    {
        "id": "06_no_terminal_punct",
        "task": "entities",
        "text": "Apple was founded by Steve Jobs in Cupertino",
        "entity_types": ["person", "organization", "location"],
        "relation_types": [],
        "notes": "Controlled pair with 01 (same sentence minus final '.'): "
        "tests the appended '.' suffix path.",
    },
    {
        "id": "07_url_email",
        "task": "entities",
        "text": "Email bob@example.com or see https://example.com/about; ask Alice.",
        "entity_types": ["person", "email", "url"],
        "relation_types": [],
        "notes": "WhitespaceTokenSplitter URL/email recognition.",
    },
    {
        "id": "08_unicode",
        "task": "entities",
        "text": "Ren\u00e9e M\u00fcller met \u7530\u4e2d\u592a\u90ce in S\u00e3o Paulo.",
        "entity_types": ["person", "location"],
        "relation_types": [],
        "notes": "Code-point vs UTF-8 byte offset stress (diacritics + CJK).",
    },
    {
        "id": "09_overlap",
        "task": "entities",
        "text": "John Smith visited New York City.",
        "entity_types": ["person", "location"],
        "relation_types": [],
        "notes": "Overlapping candidates ('John'/'John Smith', "
        "'New York'/'New York City'); overlap_policy=flat resolution.",
    },
    {
        "id": "10_empty_text",
        "task": "entities",
        "text": "",
        "entity_types": ["person", "organization", "location"],
        "relation_types": [],
        "notes": "Empty-text edge case; _normalize_text maps '' -> '.'.",
    },
    {
        "id": "11_truncated_text",
        "task": "entities",
        "text": "Apple was founded by Steve Jo",
        "entity_types": ["person", "organization"],
        "relation_types": [],
        "notes": "Text truncated mid-entity at the right edge.",
    },
    {
        "id": "12_relation_only",
        "task": "relations",
        "text": "Steve Jobs founded Apple in Cupertino.",
        "entity_types": [],
        "relation_types": ["founder"],
        "notes": "Relation-only request; expects directed founder edge.",
    },
    {
        "id": "13_combined",
        "task": "entities+relations",
        "text": "Steve Jobs founded Apple in Cupertino.",
        "entity_types": ["person", "organization", "location"],
        "relation_types": ["founder"],
        "notes": "Combined entity+relation on the same text as 12.",
    },
    {
        "id": "14_direction_fwd",
        "task": "relations",
        "text": "Steve Jobs founded Apple.",
        "entity_types": [],
        "relation_types": ["founder"],
        "notes": "Head mention precedes tail mention.",
    },
    {
        "id": "15_direction_rev",
        "task": "relations",
        "text": "Apple was founded by Steve Jobs.",
        "entity_types": [],
        "relation_types": ["founder"],
        "notes": "Direction reversal: tail mention precedes head mention; "
        "semantic edge direction must be preserved.",
    },
    {
        "id": "16_duplicate_mentions",
        "task": "relations",
        "text": "Steve Jobs founded Apple. Later, Steve Jobs returned to Apple.",
        "entity_types": [],
        "relation_types": ["founder"],
        "notes": "Duplicate mentions of both arguments -> head/tail "
        "cross-product proposals -> edge deduplication.",
    },
    {
        "id": "17_no_relation",
        "task": "relations",
        "text": "Alice visited Paris.",
        "entity_types": [],
        "relation_types": ["founder"],
        "notes": "No relation present; expected final edges = empty.",
    },
    {
        "id": "18_multi_type_combined",
        "task": "entities+relations",
        "text": "Apple is headquartered in Cupertino and was founded by Steve Jobs.",
        "entity_types": ["person", "organization", "location"],
        "relation_types": ["founder", "headquartered_in"],
        "notes": "Two relation types -> two head/tail role-query pairs; "
        "combined entity+relation request.",
    },
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def git_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def checkpoint_revision(model_dir: Path) -> str:
    meta = model_dir / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    if meta.exists():
        return meta.read_text().splitlines()[0].strip()
    return "unknown"


def lst(tensor) -> List:
    """torch tensor -> nested plain-python list (floats/ints)."""
    return tensor.detach().cpu().tolist()


def interval_score_rows(
    prefix, mean, query_mask=None, text_lengths=None,
) -> List[List]:
    """`interval_prefix_score` over every in-text [s,e) per sample/query.

    Rows are ``[batch, query, start, end, value]`` with ``0 <= start < end <=
    n_b`` — the candidate-decoder interval convention. Only intervals inside a
    sample's valid text are dumped: the mean restore ``mean * (end - start)``
    reconstructs the raw inside sum exactly only there (heads.py centers over
    valid tokens; the restore assumes the interval is fully valid).
    """
    import torch
    from gliner2.models.boundary.scoring import interval_prefix_score

    bsz, nq, lp = prefix.shape
    rows: List[List] = []
    for b in range(bsz):
        n = int(text_lengths[b]) if text_lengths is not None else lp - 1
        for qi in range(nq):
            if query_mask is not None and not bool(query_mask[b, qi]):
                continue
            for s in range(n):
                for e in range(s + 1, n + 1):
                    starts = torch.tensor([[[s]]], dtype=torch.long)
                    ends = torch.tensor([[[e]]], dtype=torch.long)
                    val = interval_prefix_score(
                        prefix[b:b + 1, qi:qi + 1], starts, ends,
                        mean[b:b + 1, qi:qi + 1],
                    )
                    rows.append([b, qi, s, e, float(val.reshape(-1)[0])])
    return rows


def check_interval_identity(case_id: str, rows: List[List], inside: List) -> None:
    """Interval reconstruction must equal the raw inside-logit sum on [s,e)."""
    for b, qi, s, e, val in rows:
        raw = inside[b][qi][s:e]
        want = float(sum(raw))
        if abs(val - want) > 1e-3:
            raise AssertionError(
                f"{case_id}: interval [{s},{e}) q={qi} b={b}: "
                f"reconstructed {val} != inside sum {want}"
            )


class Capture:
    """Accumulates intermediates for the case currently in flight."""

    def __init__(self, case_id: str, raw: bool = False, raw_detail: bool = False) -> None:
        self.data: Dict[str, Any] = {}
        self.case_id = case_id
        self.raw = raw
        self.raw_detail = raw_detail


# --------------------------------------------------------------------------
# Sanity checks (run before writing each fixture)
# --------------------------------------------------------------------------

def sanity_check(case: Dict[str, Any], fix: Dict[str, Any]) -> List[str]:
    checks: List[str] = []
    text = case["text"]
    norm = fix["normalized_text"]
    ids = fix["token_ids"]
    words = fix["words"]
    spans = fix["word_char_spans"]
    first = fix["first_subtoken_positions"]
    qpos = fix["query_marker_positions"]
    e_id = fix["marker_token_ids"]["[E]"]
    r_id = fix["marker_token_ids"]["[R]"]

    def ok(cond: bool, name: str, detail: str = "") -> None:
        if not cond:
            raise AssertionError(f"{case['id']}: check {name} failed {detail}")
        checks.append(name)

    ok(len(ids) == fix["sequence_length"], "token_ids_length",
       f"{len(ids)} != {fix['sequence_length']}")
    ok(len(words) == len(spans) == len(first), "words_offsets_align",
       f"{len(words)}/{len(spans)}/{len(first)}")
    ok(all(0 <= p < len(ids) for p in first), "first_subtoken_in_range")
    ok(all(0 <= p < len(ids) for p in qpos), "query_marker_in_range")

    expected_q = len(case["entity_types"]) + 2 * len(case["relation_types"])
    ok(len(qpos) == expected_q == len(fix["query_layout"]), "query_count",
       f"{len(qpos)} vs expected {expected_q}")

    # Marker ids at query positions: [E] for entity queries, [R] for relation
    # role queries (head, tail per relation type, in that order).
    expected_markers = [e_id] * len(case["entity_types"]) + [
        r_id
    ] * (2 * len(case["relation_types"]))
    ok([ids[p] for p in qpos] == expected_markers, "query_marker_token_ids")

    ok(all(0 <= s < e <= len(norm) for s, e in spans), "word_spans_in_range")
    ok(all(spans[i][0] <= spans[i + 1][0] for i in range(len(spans) - 1)),
       "word_spans_ordered")
    # Split words must slice the expected surface out of the normalized text
    # (token values are lowercased; compare case-folded).
    bad = [
        (w, s, norm[s:e])
        for w, (s, e) in zip(words, spans)
        if norm[s:e].lower() != w
    ]
    ok(not bad, "word_spans_slice_surface", f"{bad[:3]}")

    cidx = fix["selected_candidate_indices"]
    valid = fix["candidate_valid_mask"]
    ok(all(not v or (0 <= s < e <= len(words)) for (s, e), v in zip(cidx, valid)),
       "candidate_indices_in_word_range")
    ok(all(row == cidx for row in fix["_candidate_indices_per_query"]),
       "shared_pool_indices_query_agnostic")
    q = expected_q
    c = len(cidx)
    ok(len(fix["candidate_pair_logits"]) == q
       and all(len(row) == c for row in fix["candidate_pair_logits"]),
       "candidate_pair_logits_shape", f"want [{q},{c}]")
    ok(fix["shapes"]["boundary"]["start_logits"][1:] == [q, len(words) + 1]
       and fix["shapes"]["boundary"]["end_logits"][1:] == [q, len(words) + 1]
       and fix["shapes"]["boundary"]["inside_logits"][1:] == [q, len(words)]
       and fix["shapes"]["boundary"]["inside_prefix"][1:] == [q, len(words) + 1],
       "boundary_shapes",
       f"{fix['shapes']['boundary']} vs words={len(words)}")
    ok(len(fix["null_logits"]) == q and len(fix["count_log_rates"]) == q,
       "null_count_shapes")

    # Final entities: surfaces must slice out of the normalized text; spans
    # fully inside the caller's text must also slice the caller's text.
    for name, items in fix["final_output"].get("entities", {}).items():
        for item in items:
            s, e, surf = item["start"], item["end"], item["text"]
            ok(0 <= s < e <= len(norm), f"entity_span_in_range({name})",
               f"{s},{e}")
            ok(norm[s:e].strip() == surf, f"entity_surface({name})",
               f"{norm[s:e]!r} != {surf!r}")
            if e <= len(text):
                ok(text[s:e].strip() == surf, f"entity_surface_vs_input({name})",
                   f"{text[s:e]!r} != {surf!r}")

    # Final relation edges: same offset contract on both arguments.
    for rtype, edges in fix["final_output"].get("relation_extraction", {}).items():
        for edge in edges:
            for side in ("head", "tail"):
                arg = edge[side]
                s, e, surf = arg["start"], arg["end"], arg["text"]
                ok(0 <= s < e <= len(norm), f"edge_span_in_range({rtype}.{side})")
                ok(norm[s:e].strip() == surf, f"edge_surface({rtype}.{side})",
                   f"{norm[s:e]!r} != {surf!r}")

    ok(len(fix["relation_logits"]) == len(fix["proposed_argument_pairs"]),
       "relation_logits_align_pairs")
    for pair, logit in zip(fix["proposed_argument_pairs"], fix["relation_logits"]):
        hs, he = pair["head"]
        ts, te = pair["tail"]
        ok(0 <= hs < he <= len(words) and 0 <= ts < te <= len(words),
           "argument_pair_in_word_range", f"{pair}")
        ok(isinstance(logit, (int, float)), "relation_logit_numeric")

    # Thresholded candidates feed the entity decoder; their spans are in the
    # same word-index space as the pool.
    for qi, group in enumerate(fix["thresholded_candidates"]):
        for cand in group:
            ok(0 <= cand["start"] < cand["end"] <= len(words),
               f"thresholded_candidate_range(q={qi})", f"{cand}")

    expected_norm = "." if not text else (
        text if text.endswith((".", "!", "?")) else text + "."
    )
    ok(norm == expected_norm, "normalized_text_contract",
       f"{norm!r} != {expected_norm!r}")

    # Raw boundary/marginal dumps (epic Task 5 oracle cases).
    if "raw_boundary" in fix:
        rb = fix["raw_boundary"]
        l1 = len(words) + 1
        q = expected_q
        ok(len(rb["boundary_states"]) == l1 and all(
            len(row) > 0 for row in rb["boundary_states"]),
           "raw_boundary_states_len", f"{len(rb['boundary_states'])} != {l1}")
        ok(rb["boundary_mask"] == [1] * l1, "raw_boundary_mask_all_valid",
           f"{rb['boundary_mask']}")
        ok(len(rb["start_logits"]) == q and all(len(r) == l1 for r in rb["start_logits"]),
           "raw_start_shape", f"{len(rb['start_logits'])}x{len(rb['start_logits'][0])}")
        ok(len(rb["end_logits"]) == q and all(len(r) == l1 for r in rb["end_logits"]),
           "raw_end_shape")
        ok(len(rb["inside_logits"]) == q
           and all(len(r) == len(words) for r in rb["inside_logits"]),
           "raw_inside_shape")
        ok(len(rb["inside_prefix"]) == q and all(len(r) == l1 for r in rb["inside_prefix"]),
           "raw_prefix_shape")
        ok(len(rb["inside_prefix_mean"]) == q, "raw_prefix_mean_shape")
        ok(all(row[0] == 0.0 for row in rb["inside_prefix"]), "raw_prefix_zero_origin")
        if "after_layer_norm" in rb:
            ok(len(rb["after_layer_norm"]) == l1 and len(rb["after_attention_0"]) == l1,
               "raw_stage_shapes")
        check_interval_identity(case["id"], rb["interval_scores"], [rb["inside_logits"]])
        have = {(qi, s, e) for _, qi, s, e, _ in rb["interval_scores"]}
        for qi in range(q):
            ok((qi, 0, 1) in have, f"raw_interval_one_token(q={qi})")
            ok((qi, 0, len(words)) in have, f"raw_interval_full(q={qi})")
            if len(words) > 1:
                ok((qi, len(words) - 1, len(words)) in have,
                   f"raw_interval_last_token(q={qi})")
        if "raw_core" in fix:
            rc = fix["raw_core"]
            ok(len(rc["text_states"]) == len(words)
               and len(rc["query_states"]) == q,
               "raw_core_shapes",
               f"{len(rc['text_states'])}/{len(rc['query_states'])}")
    return checks


# --------------------------------------------------------------------------
# Capture wrappers
# --------------------------------------------------------------------------

def _store_hook(store: Dict[str, Any], key: str):
    def hook(_module, _inputs, output):
        store[key] = output.detach()
    return hook


def capture_synthetic(model) -> Dict[str, Any]:
    """Run the production boundary encoder + query head on a fixed padded batch.

    The corpus runs B=1 with all-valid masks, so mask/window behavior is
    invisible there. This synthetic B=2 / L=3 / Q=2 batch (one padded text
    token, one padded query) exercises, against the production modules and
    weights: EOS placed at each sample's own final boundary
    (``shift_right_with_eos``), boundary/token/query validity masks, finite
    ``MASK_LOGIT`` fills, padding zeroing of boundary states, the fp32
    mean-centered inside prefix over masked tokens, and the local-attention
    window (production ``window=128`` is inert at N<=4, so attention block 0
    is additionally re-run with ``window=2``).

    Inputs follow an exact fp32 lattice formula recorded in the fixture so the
    Rust side rebuilds them bit-identically without storing tensors; scattered
    samples are recorded to verify the formula agreement.
    """
    import torch

    bsz, length, nq, hidden = 2, 3, 2, 384
    dim = model.boundary_head.settings.boundary_dim
    text_vals = [
        [[((b * 131 + i * 17 + h * 7) % 32 - 16) / 8.0 for h in range(hidden)]
         for i in range(length)]
        for b in range(bsz)
    ]
    query_vals = [
        [[((b * 17 + q * 13 + h * 11) % 32 - 16) / 8.0 for h in range(hidden)]
         for q in range(nq)]
        for b in range(bsz)
    ]
    text_states = torch.tensor(text_vals, dtype=torch.float32)
    query_states = torch.tensor(query_vals, dtype=torch.float32)
    text_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    query_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    enc = model.boundary_head.boundary_encoder
    head = model.boundary_head.boundary_query_head
    stored: Dict[str, Any] = {}
    handles = [
        enc.layer_norm.register_forward_hook(_store_hook(stored, "after_layer_norm")),
        enc.attention_blocks[0].register_forward_hook(_store_hook(stored, "after_attention_0")),
        enc.refinement_blocks[0].register_forward_hook(_store_hook(stored, "after_refinement")),
    ]
    with torch.no_grad():
        encoding = enc(text_states, text_mask)
        marg = head(
            encoding.states, encoding.mask,
            text_states, text_mask,
            query_states, query_mask,
        )
    for handle in handles:
        handle.remove()

    blk = enc.attention_blocks[0]
    window = blk.window
    blk.window = 2
    with torch.no_grad():
        win2 = blk(stored["after_layer_norm"], encoding.mask)
    blk.window = window

    rows = interval_score_rows(
        marg.inside_prefix, marg.inside_prefix_mean,
        query_mask, text_lengths=[length, length - 1],
    )
    inside = lst(marg.inside_logits)
    check_interval_identity("synthetic_masks", rows, inside)

    checks: List[str] = []

    def ok(cond: bool, name: str) -> None:
        if not cond:
            raise AssertionError(f"synthetic_masks: check {name} failed")
        checks.append(name)

    b_states = lst(encoding.states)
    b_mask = [[int(v) for v in row] for row in lst(encoding.mask)]
    ok(b_mask == [[1, 1, 1, 1], [1, 1, 1, 0]], "boundary_mask_values")
    ok(b_states[1][3] == [0.0] * dim, "padding_boundary_zeroed")
    start = lst(marg.start_logits)
    ok(start[1][1][3] == -1.0e4, "padded_query_masked")
    ok(start[1][0][3] == -1.0e4, "padded_boundary_masked")
    ok(inside[1][0][2] == -1.0e4, "padded_token_masked")
    ok(all(lst(marg.inside_prefix)[b][q][0] == 0.0
           for b in range(bsz) for q in range(nq)), "prefix_zero_origin")

    def sample_rows(vals, index):
        return [
            [b, i, j, vals[b][i][j]]
            for b, i, j in index
        ]

    out = {
        "purpose": "Synthetic padded batch for boundary-encoder mask / window / "
                   "EOS-at-n_b checks (epic Task 5); production modules + weights",
        "shape": {"B": bsz, "L": length, "Q": nq, "H": hidden, "d": dim},
        "input_formula": {
            "text_states": "value(b,i,h) = ((b*131 + i*17 + h*7) % 32 - 16) / 8.0",
            "query_states": "value(b,q,h) = ((b*17 + q*13 + h*11) % 32 - 16) / 8.0",
            "note": "integer lattice -> fp32-exact; both sides must build "
                    "bit-identical inputs",
            "text_mask": [[1, 1, 1], [1, 1, 0]],
            "query_mask": [[1, 1], [1, 0]],
            "text_lengths": [length, length - 1],
        },
        "input_samples": {
            "text_states": sample_rows(text_vals, [(0, 0, 0), (0, 2, 7), (1, 1, 19), (1, 2, 383)]),
            "query_states": sample_rows(query_vals, [(0, 0, 0), (0, 1, 5), (1, 0, 383)]),
        },
        "outputs": {
            "after_layer_norm": lst(stored["after_layer_norm"]),
            "after_attention_0": lst(stored["after_attention_0"]),
            "after_attention_0_window_2": lst(win2),
            "after_refinement": lst(stored["after_refinement"]),
            "boundary_states": b_states,
            "boundary_mask": b_mask,
            "start_logits": start,
            "end_logits": lst(marg.end_logits),
            "inside_logits": inside,
            "inside_prefix": lst(marg.inside_prefix),
            "inside_prefix_mean": lst(marg.inside_prefix_mean[:, :, 0]),
            "interval_scores": rows,
        },
        "sanity_checks": checks,
    }
    return out

def install_captures(model, cap: Capture):
    """Wrap the production modules; return an undo() callable."""
    import gliner2.models.boundary.engine as engine_mod

    undo = []

    # -- preprocessing + query routing ------------------------------------
    orig_encode_core = model._encode_core

    def encode_core_wrapper(batch, _orig=orig_encode_core):
        core = _orig(batch)
        cap.data["sequence_length"] = int(batch.original_lengths[0])
        cap.data["token_ids"] = lst(batch.input_ids[0, : batch.original_lengths[0]])
        cap.data["words"] = list(batch.text_tokens[0])
        cap.data["word_char_spans"] = [
            [int(s), int(e)] for s, e in zip(batch.start_mappings[0], batch.end_mappings[0])
        ]
        cap.data["first_subtoken_positions"] = lst(
            batch.text_word_indices[0, : batch.text_word_counts[0]]
        )
        nq = int(batch.query_marker_mask[0].sum())
        cap.data["query_marker_positions"] = lst(batch.query_marker_indices[0, :nq])
        cap.data["schema_marker_positions"] = [
            list(map(int, positions))
            for positions in batch.schema_special_indices[0]
        ]
        cap.data["schema_tokens"] = [
            list(struct) for struct in batch.schema_tokens_list[0]
        ]
        cap.data["task_types"] = list(batch.task_types[0])
        cap.data["normalized_text"] = batch.original_texts[0]
        cap.data["word_offset"] = int(core["word_offsets"][0])
        cap.data["text_word_count"] = int(batch.text_word_counts[0])
        cap.data["query_layout"] = [
            {
                "query_id": j,
                "task_index": spec["group_index"],
                "task_type": spec["task_type"],
                "task_name": spec["task_name"],
                "field_index": spec["field_index"],
                "field_name": spec["field_name"],
            }
            for j, spec in enumerate(core["ext_specs"][0])
        ]
        routing = []
        for entry in core["rel_specs"][0]:
            spec = entry["spec"]
            routing.append({
                "relation_type": entry["relation_type"],
                "group_index": entry["group_index"],
                "head_query_id": list(spec.head_query_ids),
                "tail_query_id": list(spec.tail_query_ids),
                "query_state": (
                    "concat(head,tail)"
                    if model.boundary_settings.directional_relation_states
                    else "mean(head,tail)"
                ),
                "query_state_dim": int(entry["query_state"].shape[-1]),
            })
        cap.data["relation_role_routing"] = routing
        if cap.raw_detail:
            cap.data["raw_core"] = {
                "text_states": lst(core["text_states"][0]),
                "query_states": lst(core["query_states"][0]),
            }
        return core

    model._encode_core = encode_core_wrapper
    undo.append(lambda: model.__dict__.pop("_encode_core", None))

    # -- raw boundary encodings + marginals (epic Task 5 oracle) -----------
    def layer_norm_hook(_module, _inputs, output):
        cap.data.setdefault("raw_boundary", {})["after_layer_norm"] = lst(output[0])

    def attn0_hook(_module, _inputs, output):
        cap.data.setdefault("raw_boundary", {})["after_attention_0"] = lst(output[0])

    def enc_hook(_module, _inputs, output):
        rb = cap.data.setdefault("raw_boundary", {})
        rb["boundary_states"] = lst(output.states[0])
        rb["boundary_mask"] = [int(v) for v in lst(output.mask[0])]

    def query_head_hook(_module, _inputs, output):
        cap.data["_inside_prefix_shape"] = list(output.inside_prefix.shape)
        if not cap.raw:
            return
        rb = cap.data.setdefault("raw_boundary", {})
        rb["start_logits"] = lst(output.start_logits[0])
        rb["end_logits"] = lst(output.end_logits[0])
        rb["inside_logits"] = lst(output.inside_logits[0])
        rb["inside_prefix"] = lst(output.inside_prefix[0])
        rb["inside_prefix_mean"] = [
            float(v) for v in lst(output.inside_prefix_mean[0, :, 0])
        ]
        rows = interval_score_rows(
            output.inside_prefix, output.inside_prefix_mean,
            text_lengths=None,
        )
        check_interval_identity(cap.case_id, rows, lst(output.inside_logits))
        rb["interval_scores"] = rows

    if cap.raw:
        if cap.raw_detail:
            handle = model.boundary_head.boundary_encoder.layer_norm.register_forward_hook(
                layer_norm_hook
            )
            undo.append(handle.remove)
            handle = model.boundary_head.boundary_encoder.attention_blocks[0].register_forward_hook(
                attn0_hook
            )
            undo.append(handle.remove)
        handle = model.boundary_head.boundary_encoder.register_forward_hook(enc_hook)
        undo.append(handle.remove)
    handle = model.boundary_head.boundary_query_head.register_forward_hook(query_head_hook)
    undo.append(handle.remove)

    # -- shared pool builder (proposal/compat priors) ---------------------
    def pool_hook(_module, _inputs, output):
        cap.data["candidate_compat_logits"] = lst(output.compat_logits[0])
        cap.data["_pool_proposal_logits"] = lst(output.proposal_logits[0])

    handle = model.boundary_head.shared_pool_builder.register_forward_hook(pool_hook)
    undo.append(handle.remove)

    # -- boundary head output (marginal shapes, candidates, logits) -------
    def head_hook(_module, _inputs, output):
        candidates = output.candidates
        cap.data["shapes"] = {
            "boundary": {
                "start_logits": list(output.start_logits.shape),
                "end_logits": list(output.end_logits.shape),
                "inside_logits": list(output.inside_logits.shape),
                "inside_prefix": cap.data.get(
                    "_inside_prefix_shape", list(output.inside_logits.shape)
                ),
                "null_logits": list(output.null_logits.shape),
                "count_log_rates": list(output.count_log_rates.shape),
            },
            "candidates": {
                "indices": list(candidates.indices.shape),
                "pair_logits": list(candidates.pair_logits.shape),
                "proposal_logits": list(candidates.proposal_logits.shape),
                "valid_mask": list(candidates.valid_mask.shape),
            },
        }
        cap.data["selected_candidate_indices"] = [
            [int(s), int(e)] for s, e in lst(candidates.indices[0, 0])
        ]
        cap.data["_candidate_indices_per_query"] = [
            [[int(s), int(e)] for s, e in lst(candidates.indices[0, qi])]
            for qi in range(candidates.indices.shape[1])
        ]
        cap.data["candidate_valid_mask"] = [bool(v) for v in lst(candidates.valid_mask[0, 0])]
        cap.data["candidate_proposal_logits"] = [float(x) for x in lst(candidates.proposal_logits[0, 0])]
        cap.data["candidate_pair_logits"] = [
            [float(x) for x in row] for row in lst(candidates.pair_logits[0])
        ]
        cap.data["null_logits"] = [float(x) for x in lst(output.null_logits[0])]
        cap.data["count_log_rates"] = [float(x) for x in lst(output.count_log_rates[0])]

    handle = model.boundary_head.register_forward_hook(head_hook)
    undo.append(handle.remove)

    # -- thresholded candidates (entity decoder input) --------------------
    orig_group = engine_mod._group_scored_candidates

    def group_wrapper(*args, **kwargs):
        out = orig_group(*args, **kwargs)
        cap.data["thresholded_candidates"] = [
            [
                {"probability": float(p), "start": int(s), "end": int(e)}
                for p, s, e in group
            ]
            for group in out[0]
        ]
        return out

    engine_mod._group_scored_candidates = group_wrapper
    undo.append(lambda: setattr(engine_mod, "_group_scored_candidates", orig_group))

    # -- relation pair proposals -----------------------------------------
    if getattr(model, "relation_pair_generator", None) is not None:
        orig_generate = model.relation_pair_generator.generate

        def generate_wrapper(*args, **kwargs):
            pairs = orig_generate(*args, **kwargs)
            proposed = []
            for i in range(len(pairs)):
                head_key = pairs.head_keys[i] if pairs.head_keys else ("?", 0, 0)
                tail_key = pairs.tail_keys[i] if pairs.tail_keys else ("?", 0, 0)
                proposed.append({
                    "relation_type": pairs.relation_types[i],
                    "head": [int(pairs.head_start[i]), int(pairs.head_end[i])],
                    "tail": [int(pairs.tail_start[i]), int(pairs.tail_end[i])],
                    "head_prob": float(pairs.head_prob[i]),
                    "tail_prob": float(pairs.tail_prob[i]),
                    "head_query_id": int(head_key[0]),
                    "tail_query_id": int(tail_key[0]),
                })
            cap.data["proposed_argument_pairs"] = proposed
            return pairs

        model.relation_pair_generator.generate = generate_wrapper
        undo.append(
            lambda: model.relation_pair_generator.__dict__.pop("generate", None)
        )

    # -- relation scorer logits ------------------------------------------
    if getattr(model, "relation_scorer", None) is not None:
        def rel_hook(_module, _inputs, output):
            cap.data["relation_logits"] = [float(x) for x in lst(output)]

        handle = model.relation_scorer.register_forward_hook(rel_hook)
        undo.append(handle.remove)

    def undo_all():
        for fn in reversed(undo):
            fn()

    return undo_all


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    head = git_head(GLINER2_SRC)
    if head != PINNED_GLINER2_COMMIT:
        print(
            f"ERROR: GLiNER2 checkout at {GLINER2_SRC} is {head}, "
            f"expected pinned {PINNED_GLINER2_COMMIT}",
            file=sys.stderr,
        )
        return 1
    revision = checkpoint_revision(MODEL_DIR)
    if revision != PINNED_CHECKPOINT_REVISION:
        print(
            f"ERROR: checkpoint revision {revision} != pinned "
            f"{PINNED_CHECKPOINT_REVISION}",
            file=sys.stderr,
        )
        return 1

    sys.path.insert(0, str(GLINER2_SRC))
    import torch
    import transformers
    from gliner2 import AutoExtractor

    print(f"loading {MODEL_DIR} ...")
    model = AutoExtractor.from_pretrained(str(MODEL_DIR))
    model.eval()
    model.processor.change_mode(is_training=False)

    tokenizer = model.processor.tokenizer
    marker_ids = {
        name: int(tokenizer.convert_tokens_to_ids(name))
        for name in ("[E]", "[R]", "[P]", "[SEP_STRUCT]", "[SEP_TEXT]")
    }

    settings = dataclasses.asdict(model.boundary_settings)
    CASES_DIR.mkdir(exist_ok=True)
    case_index = []

    for case in CORPUS:
        cap = Capture(
            case["id"],
            raw=case["id"] in RAW_CASES,
            raw_detail=case["id"] in RAW_DETAIL_CASES,
        )
        undo = install_captures(model, cap)
        try:
            schema = model.create_schema()
            if case["entity_types"]:
                schema.entities(case["entity_types"])
            if case["relation_types"]:
                schema.relations(case["relation_types"])
            result = model.extract(
                case["text"],
                schema,
                threshold=THRESHOLD,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )
        finally:
            undo()

        fix = cap.data
        fix["marker_token_ids"] = marker_ids
        fix["final_output"] = result
        fix.setdefault("relation_role_routing", [])
        fix.setdefault("proposed_argument_pairs", [])
        fix.setdefault("relation_logits", [])
        fix.setdefault("thresholded_candidates", [])
        fix.setdefault("candidate_compat_logits", [])
        pool_proposal = fix.get("_pool_proposal_logits")
        if pool_proposal is not None:
            assert pool_proposal == fix["candidate_proposal_logits"], (
                f"{case['id']}: pool proposal logits != candidate proposal logits"
            )

        checks = sanity_check(case, fix)
        fix.pop("_candidate_indices_per_query", None)
        fix.pop("_pool_proposal_logits", None)
        fix.pop("_inside_prefix_shape", None)

        fixture = {
            "case_id": case["id"],
            "task": case["task"],
            "notes": case["notes"],
            "threshold": THRESHOLD,
            "eval_mode": True,
            "text": case["text"],
            "entity_types": case["entity_types"],
            "relation_types": case["relation_types"],
            "marker_token_ids": marker_ids,
            "normalized_text": fix["normalized_text"],
            "sequence_length": fix["sequence_length"],
            "token_ids": fix["token_ids"],
            "words": fix["words"],
            "word_char_spans": fix["word_char_spans"],
            "first_subtoken_positions": fix["first_subtoken_positions"],
            "query_marker_positions": fix["query_marker_positions"],
            "schema_marker_positions": fix["schema_marker_positions"],
            "schema_tokens": fix["schema_tokens"],
            "task_types": fix["task_types"],
            "word_offset": fix["word_offset"],
            "text_word_count": fix["text_word_count"],
            "query_layout": fix["query_layout"],
            "relation_role_routing": fix["relation_role_routing"],
            "shapes": fix["shapes"],
            "selected_candidate_indices": fix["selected_candidate_indices"],
            "candidate_valid_mask": fix["candidate_valid_mask"],
            "candidate_proposal_logits": fix["candidate_proposal_logits"],
            "candidate_compat_logits": fix["candidate_compat_logits"],
            "candidate_pair_logits": fix["candidate_pair_logits"],
            "thresholded_candidates": fix["thresholded_candidates"],
            "null_logits": fix["null_logits"],
            "count_log_rates": fix["count_log_rates"],
            "proposed_argument_pairs": fix["proposed_argument_pairs"],
            "relation_logits": fix["relation_logits"],
            "final_output": fix["final_output"],
            "sanity_checks": checks,
        }
        if "raw_boundary" in fix:
            fixture["raw_boundary"] = fix["raw_boundary"]
        if "raw_core" in fix:
            fixture["raw_core"] = fix["raw_core"]
        path = CASES_DIR / f"{case['id']}.json"
        path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=1, sort_keys=False) + "\n"
        )
        n_ent = sum(
            len(v) for v in fixture["final_output"].get("entities", {}).values()
        )
        n_edge = sum(
            len(v)
            for v in fixture["final_output"].get("relation_extraction", {}).values()
        )
        case_index.append({
            "case_id": case["id"],
            "file": f"cases/{case['id']}.json",
            "task": case["task"],
            "text": case["text"],
            "entity_types": case["entity_types"],
            "relation_types": case["relation_types"],
            "entities_extracted": n_ent,
            "edges_extracted": n_edge,
            "sanity_checks": len(checks),
        })
        print(f"  {case['id']}: {n_ent} entities, {n_edge} edges, "
              f"{len(checks)} checks OK")

    # Synthetic padded batch for mask/window/EOS behavior (epic Task 5).
    synthetic = capture_synthetic(model)
    (ORACLE_DIR / "synthetic_masks.json").write_text(
        json.dumps(synthetic, ensure_ascii=False, indent=1) + "\n"
    )
    print(f"  synthetic_masks: {len(synthetic['sanity_checks'])} checks OK")

    manifest = {
        "purpose": "Python behavioral oracle for epic gliner-2.5-boundary-support Tasks 2 and 5",
        "gliner2_reference": {
            "path": str(GLINER2_SRC),
            "commit": head,
        },
        "checkpoint": {
            "local_dir": "models/gliner2.5-small-v1",
            "hub_id": CHECKPOINT_HUB_ID,
            "hub_revision": revision,
            "weights": "model.safetensors (single file, 334 tensors, not captured)",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "capture_settings": {
            "eval_mode": True,
            "threshold": THRESHOLD,
            "type_order": "declared list order (fixed)",
            "include_confidence": True,
            "include_spans": True,
            "api": "GLiNER2.extract() -> batch_extract() (production path)",
        },
        "config": {
            "raw_config": json.loads((MODEL_DIR / "config.json").read_text()),
            "migrated_boundary_settings": settings,
            "enable_relations": bool(model.enable_relations),
            "enable_records": bool(model.enable_records),
            "candidate_pool": settings["candidate_pool"],
        },
        "conventions": {
            "offsets": "Python str indices = Unicode code points; half-open "
                       "[start,end); they index normalized_text "
                       "(= input text plus synthetic '.' when the input does "
                       "not end with '.', '!' or '?'; '' becomes '.')",
            "split_word_offsets": "word_char_spans[i] slices words[i] out of "
                                  "normalized_text (case-folded equality)",
            "word_index_space": "candidate/argument indices are half-open "
                                "[start,end) word boundaries on the text-word "
                                "axis (including any classification prefix; "
                                "word_offset shifts to document words)",
            "candidate_pool": "shared: selected_candidate_indices are "
                              "query-agnostic pool rows [start,end), common to "
                              "all queries; candidate_pair_logits is [Q][C] "
                              "raw (pre pair_temperature) logit of each pool "
                              "row per query",
            "relation_argument_selection": "sigmoid(candidate_pair_logits) "
                                           "WITHOUT pair_temperature, gated by "
                                           "relation_argument_proposal_threshold",
            "relation_logits": "raw SparseRelationScorer logits per proposed "
                               "pair (aligned with proposed_argument_pairs); "
                               "relation_temperature applied later, at decode",
            "marker_token_ids": marker_ids,
            "floats": "full Python float64 repr of the fp32 model values; "
                      "compare with tolerance ~1e-5",
            "raw_intermediates": (
                "Task 5 cases 01/05/10 carry raw_boundary (boundary states + "
                "mask, start/end/inside logits, inside_prefix, "
                "inside_prefix_mean, interval rows [batch,query,start,end,"
                "value]); case 01 additionally carries the after_layer_norm / "
                "after_attention_0 stages and raw_core (gathered text_states / "
                "query_states); synthetic_masks.json holds a padded B=2 batch "
                "(mask/window/EOS-at-n_b; inputs via the recorded fp32 lattice "
                "formula)"
            ),
            "not_captured": "weights, candidate-state tensors, relation "
                            "query-state vectors (shapes only), raw marginals "
                            "outside cases 01/05/10 (shapes only)",
        },
        "cases": case_index,
    }
    (ORACLE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
    )
    print(f"wrote {len(case_index)} cases + manifest.json + synthetic_masks.json "
          f"under {ORACLE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
