//! Boundary relation extraction (epic Task 8) — `gliner2/models/boundary/relations.py`
//! (`TypedRelationPairGenerator` + `SparseRelationScorer`) and the relation decode
//! contract of `boundary/engine.py::_decode_relations` /
//! `_deduplicate_relation_edges` at pinned GLiNER2 commit
//! `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`.
//!
//! ## Pair generation (`TypedRelationPairGenerator.generate_batched`, decode path)
//!
//! Typed and capped — never an all-pairs entity grid. For each relation type
//! the head mentions come from that type's head role query and the tail
//! mentions from its tail role query (shared candidate-pool rows; the same
//! pool row under a role query is one mention):
//!
//! 1. Argument probabilities are `sigmoid(pair_logits)` — **without**
//!    `pair_temperature` (that temperature is entity-decode only) — and are
//!    gated by `relation_argument_proposal_threshold` (`>=`, inclusive).
//! 2. Capped head/tail ranking (`relation_heads_per_type` /
//!    `relation_tails_per_type`): rank key is probability descending; ties
//!    break by `(span_start, span_end, flat_index)` ascending (Python's
//!    stable secondary sort before the stable descending argsort).
//! 3. Pair ranking over the capped cross product: score = head_prob × tail_prob
//!    descending; ties break by `(head_slot, tail_slot)` ascending (the
//!    flattened `hi * tails_per_type + ti` order). Capped at
//!    `relation_pair_cap`.
//! 4. Identical head/tail spans are excluded by default (`allow_self=false`,
//!    Python `RelationTypeSpec` default — every schema-built spec).
//!
//! ## Scoring (`SparseRelationScorer`)
//!
//! Text states are gathered at each argument's start and `end-1`; features are
//! the four endpoint states, the relation query state, `sign(tail_start -
//! head_start)` and `|tail_start - head_start| / L`, fed through
//! `mlp = (Linear, GELU, Dropout, Linear)` (indices 0/3). With
//! `relation_biaffine_content` the score additionally gets a biaffine content
//! term: mean-pooled argument contents projected and gated by the relation
//! query, `(head · gate · tail) / sqrt(H)`, plus a content linear
//! `Linear(2H + rq, 1)`.
//!
//! ## Decode (`engine.py::_decode_relations`)
//!
//! `relation_temperature` divides the raw logits **before** the sigmoid,
//! thresholding is `>= threshold` (inclusive; strict `<` drops), both half-open
//! argument spans map to caller-text byte offsets through
//! [`TextMap::candidate_bytes`] (suffix-reaching candidates are rejected in
//! full — the same rules as entities), and edges per relation type collapse
//! through [`deduplicate_relation_edges`] (contained-mention upgrade, exact
//! dedup to the best score, semantic collapse by mention distance before
//! score, strict-subset dominance). Relation direction is preserved: `head`
//! always comes from the head role query, `tail` from the tail role query.
//!
//! Relation arguments are **never** filtered through the entity decoder's
//! threshold, abstention, or overlap policy.
use anyhow::{bail, Result};
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::boundary_decode::sigmoid;
use crate::boundary_pre::TextMap;
use crate::config::BoundaryHeadConfig;

/// One directed relation argument: trimmed surface plus half-open byte offsets
/// into the caller's exact input (the untrimmed mapped range, mirroring
/// Python's `(stripped_text, h0, h1)` edge tuples and the entity convention).
#[derive(Debug, Clone, PartialEq)]
pub struct RelationArgument {
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
}

/// One directed relation edge: `head` has `relation_type` with `tail`.
/// `confidence` is `sigmoid(raw_logit / relation_temperature)` (the same value
/// Python attaches to both arguments).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedRelation {
    pub relation_type: String,
    pub head: RelationArgument,
    pub tail: RelationArgument,
    pub confidence: f32,
}

/// `RelationProposalSettings` (relations.py) + the decode-side
/// `relation_temperature`, all from `boundary_head`.
#[derive(Debug, Clone, Copy)]
pub struct RelationSettings {
    pub heads_per_type: usize,
    pub tails_per_type: usize,
    pub pair_cap: usize,
    /// Raw-sigmoid gate for arguments (`>=`, inclusive).
    pub argument_threshold: f32,
    /// Divides the relation logits before the sigmoid at decode.
    pub relation_temperature: f32,
}

impl RelationSettings {
    pub fn from_config(cfg: &BoundaryHeadConfig) -> Self {
        Self {
            heads_per_type: cfg.relation_heads_per_type,
            tails_per_type: cfg.relation_tails_per_type,
            pair_cap: cfg.relation_pair_cap,
            argument_threshold: cfg.relation_argument_proposal_threshold as f32,
            relation_temperature: cfg.relation_temperature as f32,
        }
    }
}

/// One relation type with its allowed head/tail role queries (decode-side
/// `RelationTypeSpec`; `allow_self` is `false` for every schema-built spec).
#[derive(Debug, Clone)]
pub struct RelationTypeSpec {
    pub relation_type: String,
    pub head_query_ids: Vec<usize>,
    pub tail_query_ids: Vec<usize>,
    pub allow_self: bool,
}

/// One typed head×tail proposal (`RelationPairBatch` row, compact decode form):
/// half-open word intervals plus the raw-sigmoid argument probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationPair {
    pub relation_index: usize,
    pub head_start: u32,
    pub head_end: u32,
    pub tail_start: u32,
    pub tail_end: u32,
    pub head_prob: f32,
    pub tail_prob: f32,
    /// Role query the head mention was drawn from (presentation metadata).
    pub head_query_id: usize,
    /// Role query the tail mention was drawn from.
    pub tail_query_id: usize,
}

/// One candidate argument in the capped head/tail ranking.
#[derive(Debug, Clone, Copy)]
struct ArgumentCandidate {
    query_id: usize,
    start: u32,
    end: u32,
    prob: f32,
}

/// Capped ranking of one side's arguments (Python `select()` inside
/// `TypedRelationPairGenerator.generate_batched`).
///
/// `probs` are **raw** `sigmoid(pair_logits)` values (`[Q][C]`, no
/// `pair_temperature`); eligible rows are valid pool rows under a member
/// query with `prob >= argument_threshold`. Ranking is probability descending
/// with `(span_start, span_end, flat_index)` ascending tie order; at most
/// `requested` arguments survive.
fn select_arguments(
    pool_indices: &[(u32, u32)],
    pool_valid: &[bool],
    probs: &[Vec<f32>],
    member_queries: &[usize],
    requested: usize,
    argument_threshold: f32,
) -> Vec<ArgumentCandidate> {
    let c = pool_indices.len();
    let mut entries: Vec<(usize, usize)> = Vec::new(); // (query, pool slot)
    for &q in member_queries {
        let Some(row) = probs.get(q) else {
            continue;
        };
        for (slot, &prob) in row.iter().enumerate().take(c) {
            if !pool_valid[slot] {
                continue;
            }
            // Python gates on `flat_prob >= argument_threshold` (inclusive).
            if prob < argument_threshold {
                continue;
            }
            entries.push((q, slot));
        }
    }
    // Python's secondary order: stable sort by span end, then stable sort by
    // span start = `(start, end, flat_index)` ascending (flat = q * C + slot).
    entries.sort_by_key(|&(q, slot)| (pool_indices[slot].0, pool_indices[slot].1, q * c + slot));
    // Stable descending score sort: ties keep the secondary order.
    entries.sort_by(|a, b| {
        probs[b.0][b.1]
            .partial_cmp(&probs[a.0][a.1])
            .unwrap_or(Ordering::Equal)
    });
    entries.truncate(requested);
    entries
        .iter()
        .map(|&(q, slot)| ArgumentCandidate {
            query_id: q,
            start: pool_indices[slot].0,
            end: pool_indices[slot].1,
            prob: probs[q][slot],
        })
        .collect()
}

/// Generate typed, capped head×tail proposals from the boundary candidate
/// logits (Python `TypedRelationPairGenerator.generate` decode path).
///
/// * `pool_indices` / `pool_valid` — query-agnostic candidate-pool rows.
/// * `pair_logits` — `[Q][C]` raw (pre `pair_temperature`) per-query logits.
///
/// Returns one list of [`RelationPair`] per relation type flattened in
/// **relation-index order** (Python's `[B, R, pair_cap]` batch order), and
/// within a relation type by pair score descending with
/// `(head_slot, tail_slot)` ascending ties, capped at `settings.pair_cap`.
pub fn generate_relation_pairs(
    pool_indices: &[(u32, u32)],
    pool_valid: &[bool],
    pair_logits: &[Vec<f32>],
    relation_specs: &[RelationTypeSpec],
    settings: &RelationSettings,
) -> Result<Vec<RelationPair>> {
    let c = pool_indices.len();
    if pool_valid.len() != c {
        bail!(
            "candidate pool length mismatch: indices {c}, valid {}",
            pool_valid.len()
        );
    }
    for (q, row) in pair_logits.iter().enumerate() {
        if row.len() != c {
            bail!(
                "pair_logits row {q} has {} entries, expected {c} pool rows",
                row.len()
            );
        }
    }

    // Raw `torch.sigmoid(candidates.pair_logits)` — deliberately WITHOUT
    // `pair_temperature` (epic item 8; asserted in the tests below).
    let probs: Vec<Vec<f32>> = pair_logits
        .iter()
        .map(|row| row.iter().map(|&logit| sigmoid(logit)).collect())
        .collect();

    let mut out = Vec::new();
    for (relation_index, spec) in relation_specs.iter().enumerate() {
        let heads = select_arguments(
            pool_indices,
            pool_valid,
            &probs,
            &spec.head_query_ids,
            settings.heads_per_type,
            settings.argument_threshold,
        );
        let tails = select_arguments(
            pool_indices,
            pool_valid,
            &probs,
            &spec.tail_query_ids,
            settings.tails_per_type,
            settings.argument_threshold,
        );
        let mut pairs: Vec<(f32, RelationPair)> = Vec::new();
        for head in &heads {
            for tail in &tails {
                // Default exclusion of identical head/tail spans
                // (`allow_self | ~same_span` in Python).
                if !spec.allow_self && (head.start, head.end) == (tail.start, tail.end) {
                    continue;
                }
                pairs.push((
                    head.prob * tail.prob,
                    RelationPair {
                        relation_index,
                        head_start: head.start,
                        head_end: head.end,
                        tail_start: tail.start,
                        tail_end: tail.end,
                        head_prob: head.prob,
                        tail_prob: tail.prob,
                        head_query_id: head.query_id,
                        tail_query_id: tail.query_id,
                    },
                ));
            }
        }
        // Stable descending score sort keeps the (head_slot, tail_slot)
        // ascending grid order on ties (Python's stable argsort over the
        // flattened `hi * tails_per_type + ti` scores).
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        pairs.truncate(settings.pair_cap);
        out.extend(pairs.into_iter().map(|(_, pair)| pair));
    }
    Ok(out)
}

/// `relations.py::SparseRelationScorer` — scores proposed relation pairs from
/// the gathered text states at each argument's start/end-1, relative position
/// features, an MLP, and (when `relation_biaffine_content`) biaffine content.
pub struct SparseRelationScorer {
    /// `mlp` Sequential(Linear(in, H), GELU, Dropout, Linear(H, 1)) — PyTorch
    /// indices 0 and 3.
    mlp_0: Linear,
    mlp_3: Linear,
    head_content_projection: Option<Linear>,
    tail_content_projection: Option<Linear>,
    relation_content_gate: Option<Linear>,
    content_linear: Option<Linear>,
    hidden_size: usize,
    relation_query_dim: usize,
}

impl SparseRelationScorer {
    /// Load `relation_scorer.*` (flag-gated shapes already pinned by
    /// `expected_boundary_weights`).
    pub fn load(vb: VarBuilder, cfg: &BoundaryHeadConfig, hidden: usize) -> Result<Self> {
        let rq = if cfg.directional_relation_states {
            2 * hidden
        } else {
            hidden
        };
        let in_dim = 4 * hidden + rq + 2;
        let biaffine = cfg.relation_biaffine_content;
        Ok(Self {
            mlp_0: linear(in_dim, hidden, vb.pp("mlp").pp("0"))?,
            mlp_3: linear(hidden, 1, vb.pp("mlp").pp("3"))?,
            head_content_projection: if biaffine {
                Some(linear(hidden, hidden, vb.pp("head_content_projection"))?)
            } else {
                None
            },
            tail_content_projection: if biaffine {
                Some(linear(hidden, hidden, vb.pp("tail_content_projection"))?)
            } else {
                None
            },
            relation_content_gate: if biaffine {
                Some(linear(rq, hidden, vb.pp("relation_content_gate"))?)
            } else {
                None
            },
            content_linear: if biaffine {
                Some(linear(2 * hidden + rq, 1, vb.pp("content_linear"))?)
            } else {
                None
            },
            hidden_size: hidden,
            relation_query_dim: rq,
        })
    }

    /// Score each pair in `pairs` — raw logits (pre `relation_temperature`).
    ///
    /// * `text_states` — `[1, L, H]` first-subtoken states (the scorer's
    ///   `boundary_states` parameter is fed `core["text_states"]`).
    /// * `relation_query_states` — `[1, R, rq]` per-relation query states
    ///   (`concat(head,tail)` when `directional_relation_states`, else mean).
    pub fn forward(
        &self,
        text_states: &Tensor,
        relation_query_states: &Tensor,
        pairs: &[RelationPair],
    ) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let (l, h) = {
            let t = text_states.dims();
            if t.len() != 3 || t[0] != 1 {
                bail!("text_states must be [1, L, H], got {t:?}");
            }
            (t[1], t[2])
        };
        if h != self.hidden_size {
            bail!(
                "text_states hidden {h} != scorer hidden {}",
                self.hidden_size
            );
        }
        let rq = self.relation_query_dim;
        let rt = relation_query_states.dims();
        if rt.len() != 3 || rt[0] != 1 || rt[2] != rq {
            bail!("relation_query_states must be [1, R, {rq}], got {rt:?}");
        }
        let rel_count = rt[1];

        let device = text_states.device();
        let p = pairs.len();
        let gather = |get: fn(&RelationPair) -> u32| -> Result<Tensor> {
            // Python `pos.clamp(0, max(length - 1, 0))` before the gather.
            let idx: Vec<u32> = pairs
                .iter()
                .map(|pair| get(pair).min(l.saturating_sub(1) as u32))
                .collect();
            Ok(text_states
                .index_select(&Tensor::from_vec(idx, (p,), device)?, 1)?
                .squeeze(0)?) // [P, H]
        };
        let h_start = gather(|pair| pair.head_start)?;
        let h_end = gather(|pair| pair.head_end.saturating_sub(1))?;
        let t_start = gather(|pair| pair.tail_start)?;
        let t_end = gather(|pair| pair.tail_end.saturating_sub(1))?;

        let rel_idx: Vec<u32> = pairs
            .iter()
            .map(|pair| pair.relation_index.min(rel_count.saturating_sub(1)) as u32)
            .collect();
        let rel = relation_query_states
            .index_select(&Tensor::from_vec(rel_idx, (p,), device)?, 1)?
            .squeeze(0)?; // [P, rq]

        // Positional features from integer indices (Python builds them in the
        // activation dtype before the concatenation).
        let mut order = Vec::with_capacity(p);
        let mut dist = Vec::with_capacity(p);
        let denom = (l.max(1)) as f32;
        for pair in pairs {
            let delta = pair.tail_start as i64 - pair.head_start as i64;
            order.push(delta.signum() as f32);
            dist.push(delta.unsigned_abs() as f32 / denom);
        }
        let order_t = Tensor::from_vec(order, (p, 1), device)?;
        let dist_t = Tensor::from_vec(dist, (p, 1), device)?;

        let feats = Tensor::cat(
            &[&h_start, &h_end, &t_start, &t_end, &rel, &order_t, &dist_t],
            1,
        )?; // [P, 4H + rq + 2]
        let mut scores = self
            .mlp_3
            .forward(&self.mlp_0.forward(&feats)?.gelu_erf()?)?
            .squeeze(1)?
            .to_vec1::<f32>()?;

        if let (Some(hc_proj), Some(tc_proj), Some(gate_proj), Some(content_linear)) = (
            &self.head_content_projection,
            &self.tail_content_projection,
            &self.relation_content_gate,
            &self.content_linear,
        ) {
            // fp32 prefix over the word axis, mirroring
            // `boundary_states.float().cumsum(1)`; `pool(start, end)` is
            // `(prefix[end] - prefix[start]) / max(end - start, 1)` with the
            // Python clamps on the gather indices (start/end, not end-1).
            let rows = text_states.squeeze(0)?.to_vec2::<f32>()?;
            let mut prefix: Vec<Vec<f32>> = Vec::with_capacity(l + 1);
            prefix.push(vec![0f32; h]);
            for row in &rows {
                let mut acc = prefix.last().unwrap().clone();
                for (a, &v) in acc.iter_mut().zip(row) {
                    *a += v;
                }
                prefix.push(acc);
            }
            let pool =
                |start: fn(&RelationPair) -> u32, end: fn(&RelationPair) -> u32| -> Vec<f32> {
                    let mut out = Vec::with_capacity(p * h);
                    for pair in pairs {
                        let s = (start(pair) as usize).min(l);
                        let e = (end(pair) as usize).min(l);
                        let width = (end(pair) as i64 - start(pair) as i64).max(1) as f32;
                        for (acc, base) in prefix[e].iter().zip(&prefix[s]) {
                            out.push((acc - base) / width);
                        }
                    }
                    out
                };
            let head_pool = Tensor::from_vec(
                pool(|pair| pair.head_start, |pair| pair.head_end),
                (p, h),
                device,
            )?;
            let tail_pool = Tensor::from_vec(
                pool(|pair| pair.tail_start, |pair| pair.tail_end),
                (p, h),
                device,
            )?;
            let head_content_t = hc_proj.forward(&head_pool)?;
            let tail_content_t = tc_proj.forward(&tail_pool)?;
            let head_content = head_content_t.to_vec2::<f32>()?;
            let tail_content = tail_content_t.to_vec2::<f32>()?;
            let gate_raw = gate_proj.forward(&rel)?.to_vec2::<f32>()?;
            let content_cat = Tensor::cat(&[&head_content_t, &tail_content_t, &rel], 1)?;
            let linear_term = content_linear
                .forward(&content_cat)?
                .squeeze(1)?
                .to_vec1::<f32>()?;
            let hidden_sqrt = (h as f32).sqrt();
            for (i, score) in scores.iter_mut().enumerate() {
                let mut biaffine = 0f32;
                for k in 0..h {
                    let gate = sigmoid(gate_raw[i][k]);
                    biaffine += head_content[i][k] * gate * tail_content[i][k];
                }
                *score += biaffine / hidden_sqrt + linear_term[i];
            }
        }
        Ok(scores)
    }
}

/// One decoded edge (Python `_decode_relations`'s edge dict): trimmed surface
/// plus untrimmed byte offsets for each directed argument.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationEdge {
    pub score: f32,
    pub head: (String, usize, usize),
    pub tail: (String, usize, usize),
}

/// Python `str.casefold()`-style key: case-folded with collapsed whitespace.
/// Rust has no built-in casefold; `to_lowercase()` matches on the ASCII/latin
/// corpus this port targets (Python's aggressive full casefold — e.g. `ß` →
/// `ss` — is a documented limitation in DEVELOP.md).
fn semantic_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn token_set(value: &str) -> Vec<String> {
    let mut tokens: Vec<String> = semantic_text(value)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

fn tokens_eq(a: &[String], b: &[String]) -> bool {
    a == b
}

fn tokens_subset(a: &[String], b: &[String]) -> bool {
    a.len() < b.len() && a.iter().all(|t| b.contains(t))
}

/// Port of `engine.py::_deduplicate_relation_edges`: collapse overlap and
/// repeated-mention relation cross-products into semantic edges.
///
/// 1. **Contained-mention upgrade** (`canonical_mentions`): each side's
///    mentions are upgraded to their largest containing mention (char-span
///    containment; ties impossible — same length+start means same span).
/// 2. Exact dedup on the upgraded `(head, tail)` coordinates keeps the
///    highest score.
/// 3. Semantic collapse on case-folded surfaces keeps the best
///    `(mention_distance, -score, head_start, tail_start)` — mention distance
///    **before** score (Task 2 finding).
/// 4. Strict-subset dominance: an edge whose one argument's token set is a
///    strict subset of another edge's (same opposite endpoint) is dropped.
///
/// For fewer than two input edges the list is returned unchanged (Python's
/// early return); otherwise the kept edges are sorted by
/// `(head_start, tail_start, -score)`.
pub fn deduplicate_relation_edges(edges: &[RelationEdge]) -> Vec<RelationEdge> {
    if edges.len() < 2 {
        return edges.to_vec();
    }

    fn canonical_mentions(mentions: &[(String, usize, usize)]) -> Vec<(String, usize, usize)> {
        /// `(text, start, end)` tuple (Python's mention triple).
        type Mention = (String, usize, usize);
        // Distinct coordinates in first-seen order (Python dict semantics;
        // duplicate coordinates carry identical mention tuples).
        let mut distinct: Vec<(usize, usize)> = Vec::new();
        for mention in mentions {
            let key = (mention.1, mention.2);
            if !distinct.contains(&key) {
                distinct.push(key);
            }
        }
        let lookup = |key: (usize, usize)| -> Mention {
            mentions
                .iter()
                .find(|m| (m.1, m.2) == key)
                .cloned()
                .expect("canonical key present")
        };
        let mut canonical: HashMap<(usize, usize), Mention> = HashMap::new();
        for &coords in &distinct {
            let (start, end) = coords;
            // Python: `max(containing, key=lambda c: (c[2] - c[1], -c[1]))` —
            // largest char span wins, then the smaller start (first max wins
            // unreachable ties).
            let mut best: Option<((usize, usize), Mention)> = None;
            for &other in &distinct {
                let mention = lookup(other);
                if mention.1 <= start && mention.2 >= end {
                    let key = (mention.2 - mention.1, usize::MAX - mention.1);
                    if best.as_ref().is_none_or(|(k, _)| key > *k) {
                        best = Some((key, mention));
                    }
                }
            }
            canonical.insert(coords, best.expect("each mention contains itself").1);
        }
        mentions
            .iter()
            .map(|m| canonical.get(&(m.1, m.2)).cloned().expect("canonicalized"))
            .collect()
    }

    let head_canonical =
        canonical_mentions(&edges.iter().map(|e| e.head.clone()).collect::<Vec<_>>());
    let tail_canonical =
        canonical_mentions(&edges.iter().map(|e| e.tail.clone()).collect::<Vec<_>>());

    // Exact dedup on upgraded coordinates, keeping the best score (first wins
    // score ties — Python `edge["score"] > previous["score"]`).
    let mut exact: Vec<RelationEdge> = Vec::new();
    let mut exact_index: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
    for (i, edge) in edges.iter().enumerate() {
        let normalized = RelationEdge {
            score: edge.score,
            head: head_canonical[i].clone(),
            tail: tail_canonical[i].clone(),
        };
        let key = (
            normalized.head.1,
            normalized.head.2,
            normalized.tail.1,
            normalized.tail.2,
        );
        match exact_index.get(&key) {
            None => {
                exact_index.insert(key, exact.len());
                exact.push(normalized);
            }
            Some(&j) => {
                if normalized.score > exact[j].score {
                    exact[j] = normalized;
                }
            }
        }
    }

    // Semantic collapse: `rank = (distance, -score, head_start, tail_start)`
    // with `distance = max(hs - te, ts - he, 0)` — mention distance BEFORE
    // score. Strictly-better rank replaces (first wins ties).
    let rank = |edge: &RelationEdge| -> (i64, f32, usize, usize) {
        let (_, hs, he) = edge.head;
        let (_, ts, te) = edge.tail;
        let distance = (hs as i64 - te as i64).max(ts as i64 - he as i64).max(0);
        (distance, -edge.score, hs, ts)
    };
    let mut semantic: Vec<RelationEdge> = Vec::new();
    let mut semantic_index: HashMap<(String, String), usize> = HashMap::new();
    for edge in exact {
        let key = (semantic_text(&edge.head.0), semantic_text(&edge.tail.0));
        match semantic_index.get(&key) {
            None => {
                semantic_index.insert(key, semantic.len());
                semantic.push(edge);
            }
            Some(&j) => {
                if rank(&edge) < rank(&semantic[j]) {
                    semantic[j] = edge;
                }
            }
        }
    }

    // Strict-subset dominance (compared against ALL semantic values).
    let token_sets: Vec<(Vec<String>, Vec<String>)> = semantic
        .iter()
        .map(|e| (token_set(&e.head.0), token_set(&e.tail.0)))
        .collect();
    let mut kept: Vec<RelationEdge> = Vec::new();
    for (i, edge) in semantic.iter().enumerate() {
        let dominated = token_sets
            .iter()
            .enumerate()
            .any(|(j, (other_h, other_t))| {
                if j == i {
                    return false;
                }
                let (h, t) = &token_sets[i];
                (tokens_subset(h, other_h) && tokens_eq(t, other_t))
                    || (tokens_subset(t, other_t) && tokens_eq(h, other_h))
            });
        if !dominated {
            kept.push(edge.clone());
        }
    }

    kept.sort_by(|a, b| {
        a.head
            .1
            .cmp(&b.head.1)
            .then(a.tail.1.cmp(&b.tail.1))
            .then(b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
    });
    kept
}

/// Port of `engine.py::_decode_relations` (post-scoring stage): calibrate the
/// raw logits with `relation_temperature`, threshold, map both half-open
/// argument spans to caller-text bytes, and deduplicate per relation type.
///
/// * `pairs` / `relation_logits` are aligned outputs of
///   [`generate_relation_pairs`] and [`SparseRelationScorer::forward`].
/// * Output order matches Python's `final_output.relation_extraction`:
///   relation types in first-surviving-pair order (relation-index order for
///   the relation-major proposal list), and within a type the
///   [`deduplicate_relation_edges`] order.
pub fn decode_relations_from_pairs(
    text_map: &TextMap,
    relation_specs: &[RelationTypeSpec],
    pairs: &[RelationPair],
    relation_logits: &[f32],
    relation_temperature: f32,
    threshold: f32,
) -> Result<Vec<ExtractedRelation>> {
    if pairs.len() != relation_logits.len() {
        bail!(
            "relation logits {} do not align with {} proposed pairs",
            relation_logits.len(),
            pairs.len()
        );
    }
    let mut group_order: Vec<String> = Vec::new();
    let mut groups: Vec<(String, Vec<RelationEdge>)> = Vec::new();
    for (pair, &logit) in pairs.iter().zip(relation_logits) {
        let spec = relation_specs.get(pair.relation_index).ok_or_else(|| {
            anyhow::anyhow!(
                "relation pair references missing relation spec {}",
                pair.relation_index
            )
        })?;
        // `torch.sigmoid(logits / relation_temperature)` — division BEFORE the
        // sigmoid (contrast with the raw-sigmoid argument selection above).
        let score = sigmoid(logit / relation_temperature);
        if score < threshold {
            continue;
        }
        let Some(head_range) =
            text_map.candidate_bytes(pair.head_start as usize, pair.head_end as usize)
        else {
            continue;
        };
        let Some(tail_range) =
            text_map.candidate_bytes(pair.tail_start as usize, pair.tail_end as usize)
        else {
            continue;
        };
        let head_surface = text_map.surface(head_range.clone()).trim();
        let tail_surface = text_map.surface(tail_range.clone()).trim();
        if head_surface.is_empty() || tail_surface.is_empty() {
            continue;
        }
        let edge = RelationEdge {
            score,
            head: (head_surface.to_string(), head_range.start, head_range.end),
            tail: (tail_surface.to_string(), tail_range.start, tail_range.end),
        };
        match group_order.iter().position(|n| n == &spec.relation_type) {
            Some(pos) => groups[pos].1.push(edge),
            None => {
                group_order.push(spec.relation_type.clone());
                groups.push((spec.relation_type.clone(), vec![edge]));
            }
        }
    }

    let mut out = Vec::new();
    for (relation_type, edges) in groups {
        for edge in deduplicate_relation_edges(&edges) {
            out.push(ExtractedRelation {
                relation_type: relation_type.clone(),
                head: RelationArgument {
                    text: edge.head.0,
                    char_start: edge.head.1,
                    char_end: edge.head.2,
                },
                tail: RelationArgument {
                    text: edge.tail.0,
                    char_start: edge.tail.1,
                    char_end: edge.tail.2,
                },
                confidence: edge.score,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Focused epic-Task-8 checks (no model required): typed/capped pair
    //! generation (caps, tie order, identical-span exclusion, proposal
    //! threshold, raw-logit argument selection without `pair_temperature`),
    //! the biaffine scorer term, and final edge dedup (contained-mention
    //! upgrade). End-to-end oracle comparison lives in `crate::boundary`
    //! (`task8_*`).
    use super::*;
    use candle_core::{DType, Device};

    fn settings(threshold: f32) -> RelationSettings {
        RelationSettings {
            heads_per_type: 32,
            tails_per_type: 32,
            pair_cap: 128,
            argument_threshold: threshold,
            relation_temperature: 1.0,
        }
    }

    fn spec(head: usize, tail: usize) -> RelationTypeSpec {
        RelationTypeSpec {
            relation_type: "rel".to_string(),
            head_query_ids: vec![head],
            tail_query_ids: vec![tail],
            allow_self: false,
        }
    }

    #[test]
    fn pair_generator_caps_and_tie_order() {
        // 4 pool rows, head role = query 0, tail role = query 1. Two head
        // arguments tie exactly (logit 1.0) and two tail arguments tie
        // exactly (logit 2.0).
        let pool = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let valid = [true; 4];
        let pair_logits = vec![vec![1.0, 1.0, 0.5, -1.0], vec![-9.0, -9.0, 2.0, 2.0]];
        let mut s = settings(0.25);
        s.heads_per_type = 2;
        s.tails_per_type = 2;
        s.pair_cap = 2;
        let pairs =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &s).unwrap();

        // Head ties break by (start, end, flat): (0,1) before (1,2). Tail
        // ties break the same way: (2,3) before (3,4). All four pair products
        // are equal (0.7311 × 0.8808) — pair ties break by (head_slot,
        // tail_slot): (0,0), (0,1) survive the cap of 2.
        assert_eq!(pairs.len(), 2, "pair_cap = 2");
        assert_eq!((pairs[0].head_start, pairs[0].head_end), (0, 1));
        assert_eq!((pairs[0].tail_start, pairs[0].tail_end), (2, 3));
        assert_eq!((pairs[1].head_start, pairs[1].head_end), (0, 1));
        assert_eq!((pairs[1].tail_start, pairs[1].tail_end), (3, 4));

        // Uncapped run emits all four in (head_slot, tail_slot) grid order.
        s.pair_cap = 128;
        let all = generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &s).unwrap();
        assert_eq!(all.len(), 4);
        let order: Vec<((u32, u32), (u32, u32))> = all
            .iter()
            .map(|p| ((p.head_start, p.head_end), (p.tail_start, p.tail_end)))
            .collect();
        assert_eq!(
            order,
            vec![
                ((0, 1), (2, 3)),
                ((0, 1), (3, 4)),
                ((1, 2), (2, 3)),
                ((1, 2), (3, 4)),
            ],
            "pair tie order is the (head_slot, tail_slot) grid order"
        );
    }

    #[test]
    fn pair_generator_caps_head_and_tail_arguments() {
        let pool = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let valid = [true; 4];
        // Head query 0 ranks all four rows; tail query 1 only row 3.
        let pair_logits = vec![vec![4.0, 3.0, 2.0, 1.0], vec![-9.0, -9.0, -9.0, 3.0]];
        let mut s = settings(0.0);
        s.heads_per_type = 1;
        s.tails_per_type = 1;
        let pairs =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &s).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].head_start, pairs[0].head_end), (0, 1), "top head");
        assert_eq!((pairs[0].tail_start, pairs[0].tail_end), (3, 4), "top tail");
    }

    #[test]
    fn pair_generator_excludes_identical_head_tail_spans() {
        let pool = [(0, 1), (1, 2)];
        let valid = [true; 2];
        // Both roles rank span (0,1) first, then (1,2).
        let pair_logits = vec![vec![5.0, 0.0], vec![5.0, 0.0]];
        let s = settings(0.0);
        let pairs =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &s).unwrap();
        assert!(
            pairs
                .iter()
                .all(|p| (p.head_start, p.head_end) != (p.tail_start, p.tail_end)),
            "identical head/tail spans are excluded by default"
        );
        // Cross pairs still exist; only the two identical pairs are missing.
        assert_eq!(pairs.len(), 2);

        // allow_self=true lifts the exclusion.
        let mut self_spec = spec(0, 1);
        self_spec.allow_self = true;
        let with_self =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[self_spec], &s).unwrap();
        assert_eq!(with_self.len(), 4);
        assert_eq!(
            with_self
                .iter()
                .filter(|p| (p.head_start, p.head_end) == (p.tail_start, p.tail_end))
                .count(),
            2
        );
    }

    #[test]
    fn pair_generator_applies_the_proposal_threshold_inclusively() {
        let pool = [(0, 1), (1, 2)];
        let valid = [true; 2];
        // sigmoid(0) = 0.5 exactly on the eligible rows (kept at threshold
        // 0.5, inclusive); sigmoid(-1) = 0.269 rows are gated out.
        let pair_logits = vec![vec![0.0, -1.0], vec![-1.0, 0.0]];
        let s = settings(0.5);
        let pairs =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &s).unwrap();
        assert_eq!(
            pairs.len(),
            1,
            "only the >= threshold arguments are eligible"
        );
        assert_eq!((pairs[0].head_start, pairs[0].head_end), (0, 1));
        assert_eq!((pairs[0].tail_start, pairs[0].tail_end), (1, 2));
        // Strictly above 0.5 nothing survives — the gate is exactly `>=`.
        let strict = settings(0.500001);
        let none =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &strict).unwrap();
        assert!(none.is_empty(), "prob 0.5 is dropped above threshold 0.5");
    }

    #[test]
    fn pair_generator_uses_raw_pair_logits_without_pair_temperature() {
        let pool = [(0, 1), (1, 2)];
        let valid = [true; 2];
        // sigmoid(0.5) = 0.6225 >= 0.6 keeps the arguments; with a
        // hypothetical pair_temperature = 2 the gate would see
        // sigmoid(0.25) = 0.5621 and drop them. Entity `pair_temperature`
        // must NOT reach argument selection (epic item 8).
        let pair_logits = vec![vec![0.5, -0.5], vec![-0.5, 0.5]];
        let s = settings(0.6);
        let pairs =
            generate_relation_pairs(&pool, &valid, &pair_logits, &[spec(0, 1)], &s).unwrap();
        assert_eq!(
            pairs.len(),
            1,
            "raw sigmoid keeps the pair pair_temperature would drop"
        );
        assert_eq!((pairs[0].head_start, pairs[0].head_end), (0, 1));
        assert_eq!((pairs[0].tail_start, pairs[0].tail_end), (1, 2));
        assert!(
            (pairs[0].head_prob - sigmoid(0.5)).abs() < 1e-6,
            "head_prob is sigmoid(raw logit): {} vs {}",
            pairs[0].head_prob,
            sigmoid(0.5)
        );
        assert!(
            (pairs[0].tail_prob - sigmoid(0.5)).abs() < 1e-6,
            "tail_prob is sigmoid(raw logit)"
        );
        assert!(
            sigmoid(0.5 / 2.0) < 0.6,
            "temperature-scaled prob would drop"
        );
    }

    #[test]
    fn edge_dedup_upgrades_contained_mentions() {
        // Case-16-shaped cross product: duplicate mentions of both arguments
        // plus contained partial mentions ("Steve", "Steve Jobs returned").
        let edges = vec![
            RelationEdge {
                score: 0.6296,
                head: ("Steve Jobs".into(), 0, 10),
                tail: ("Apple".into(), 56, 61),
            },
            RelationEdge {
                score: 0.6436,
                head: ("Steve Jobs".into(), 33, 43),
                tail: ("Apple".into(), 56, 61),
            },
            RelationEdge {
                score: 0.7331,
                head: ("Steve Jobs".into(), 0, 10),
                tail: ("Apple".into(), 19, 24),
            },
            RelationEdge {
                score: 0.6623,
                head: ("Steve Jobs".into(), 33, 43),
                tail: ("Apple".into(), 19, 24),
            },
            RelationEdge {
                score: 0.5027,
                head: ("Steve Jobs returned".into(), 33, 52),
                tail: ("Apple".into(), 56, 61),
            },
            RelationEdge {
                score: 0.6238,
                head: ("Steve".into(), 0, 5),
                tail: ("Apple".into(), 56, 61),
            },
            RelationEdge {
                score: 0.7272,
                head: ("Steve".into(), 0, 5),
                tail: ("Apple".into(), 19, 24),
            },
        ];
        let kept = deduplicate_relation_edges(&edges);
        // Contained head mentions upgrade to their largest container, the
        // semantic collapse prefers the closer tail mention, and the
        // strict-subset dominance drops the short-head variant: one edge.
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].head, ("Steve Jobs returned".to_string(), 33, 52));
        assert_eq!(kept[0].tail, ("Apple".to_string(), 56, 61));
        assert_eq!(kept[0].score, 0.6436, "kept the upgraded edge's own score");
    }

    #[test]
    fn edge_dedup_keeps_the_best_score_per_exact_span() {
        let edges = vec![
            RelationEdge {
                score: 0.7,
                head: ("Alice".into(), 0, 5),
                tail: ("Bob".into(), 10, 13),
            },
            RelationEdge {
                score: 0.9,
                head: ("Alice".into(), 0, 5),
                tail: ("Bob".into(), 10, 13),
            },
        ];
        let kept = deduplicate_relation_edges(&edges);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].score, 0.9, "exact duplicate keeps the higher score");
    }

    #[test]
    fn edge_dedup_semantic_rank_prefers_mention_distance_before_score() {
        // Same surfaces ("Apple"); the closer tail wins even with a lower
        // score (Task 2's surprise).
        let edges = vec![
            RelationEdge {
                score: 0.9,
                head: ("Steve Jobs".into(), 0, 10),
                tail: ("Apple".into(), 100, 105),
            },
            RelationEdge {
                score: 0.6,
                head: ("Steve Jobs".into(), 0, 10),
                tail: ("Apple".into(), 20, 25),
            },
        ];
        let kept = deduplicate_relation_edges(&edges);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].score, 0.6, "closer mention wins before score");
        assert_eq!(kept[0].tail.1, 20);
    }

    // ── scorer: MLP term + optional biaffine content ─────────────────────────

    fn lin(rows: Vec<f32>, out: usize, inn: usize) -> Linear {
        Linear::new(
            Tensor::from_vec(rows, (out, inn), &Device::Cpu).unwrap(),
            Some(Tensor::zeros(out, DType::F32, &Device::Cpu).unwrap()),
        )
    }

    fn zero_scorer(biaffine: bool) -> SparseRelationScorer {
        // hidden = 2, relation_query_dim = 2 (mean mode), in_dim = 4*2+2+2 = 12.
        let h = 2usize;
        let rq = 2usize;
        let in_dim = 4 * h + rq + 2;
        SparseRelationScorer {
            mlp_0: lin(vec![0.0; h * in_dim], h, in_dim),
            mlp_3: lin(vec![0.0; h], 1, h),
            head_content_projection: biaffine.then(|| lin(vec![1., 0., 0., 1.], h, h)),
            tail_content_projection: biaffine.then(|| lin(vec![1., 0., 0., 1.], h, h)),
            relation_content_gate: biaffine.then(|| lin(vec![0.0; h * rq], h, rq)),
            content_linear: biaffine.then(|| lin(vec![0.0; 2 * h + rq], 1, 2 * h + rq)),
            hidden_size: h,
            relation_query_dim: rq,
        }
    }

    #[test]
    fn relation_scorer_biaffine_term_is_present_and_additive() {
        // text_states [1, 3, 2]: head [0,2) pools to (2, 3), tail [2,3) to
        // (5, 6); identity content projections and a zero gate logit give
        // gate = 0.5 per dim.
        let text_states =
            Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 3, 2), &Device::Cpu).unwrap();
        let rel_states = Tensor::from_vec(vec![0f32, 0.], (1, 1, 2), &Device::Cpu).unwrap();
        let pairs = vec![RelationPair {
            relation_index: 0,
            head_start: 0,
            head_end: 2,
            tail_start: 2,
            tail_end: 3,
            head_prob: 1.0,
            tail_prob: 1.0,
            head_query_id: 0,
            tail_query_id: 1,
        }];

        let plain = zero_scorer(false)
            .forward(&text_states, &rel_states, &pairs)
            .unwrap();
        assert_eq!(plain, vec![0.0], "zero-weights MLP term is 0");

        // With biaffine content the score gains
        // (head_content * gate * tail_content).sum / sqrt(2) = 14 / sqrt(2)
        // (the zero-weights content linear contributes 0).
        let with_biaffine = zero_scorer(true)
            .forward(&text_states, &rel_states, &pairs)
            .unwrap();
        let want = 14.0f32 / (2f32).sqrt();
        assert!(
            (with_biaffine[0] - want).abs() < 1e-4,
            "biaffine term: {} != {want}",
            with_biaffine[0]
        );
        assert!(
            (with_biaffine[0] - plain[0]).abs() > 1.0,
            "biaffine term must actually change the score"
        );

        // The content linear is additive alongside the biaffine product.
        let mut with_linear = zero_scorer(true);
        // bias 1.25 on the content linear.
        with_linear.content_linear = Some(Linear::new(
            Tensor::zeros((1, 6), DType::F32, &Device::Cpu).unwrap(),
            Some(Tensor::from_vec(vec![1.25f32], (1,), &Device::Cpu).unwrap()),
        ));
        let got = with_linear
            .forward(&text_states, &rel_states, &pairs)
            .unwrap();
        assert!((got[0] - (want + 1.25)).abs() < 1e-4, "got {}", got[0]);
    }

    #[test]
    fn relation_scorer_endpoint_gathers_use_start_and_end_minus_one() {
        // A one-word tail [2,3) gathers word 2 for both endpoints; a longer
        // head [0,2) gathers words 0 and 1. Hidden = 2 with an MLP that reads
        // feature dim 0 (h_start[0]) and dim 2 (h_end[0]).
        let h = 2usize;
        let rq = 2usize;
        let in_dim = 4 * h + rq + 2;
        let mut mlp_0 = vec![0f32; h * in_dim];
        // out0 = feats[0] (h_start[0]), out1 = feats[2] (h_end[0]).
        mlp_0[0] = 1.0;
        mlp_0[in_dim + 2] = 1.0;
        let mut scorer = zero_scorer(false);
        scorer.mlp_0 = lin(mlp_0, h, in_dim);
        // mlp_3 = sum of its two inputs: score = h_start[0] + h_end[0].
        scorer.mlp_3 = lin(vec![1.0, 1.0], 1, h);

        let text_states =
            Tensor::from_vec(vec![1f32, 2., 30., 40., 5., 6.], (1, 3, 2), &Device::Cpu).unwrap();
        let rel_states = Tensor::from_vec(vec![0f32, 0.], (1, 1, 2), &Device::Cpu).unwrap();
        let pairs = vec![RelationPair {
            relation_index: 0,
            head_start: 0,
            head_end: 2,
            tail_start: 2,
            tail_end: 3,
            head_prob: 1.0,
            tail_prob: 1.0,
            head_query_id: 0,
            tail_query_id: 1,
        }];
        let got = scorer.forward(&text_states, &rel_states, &pairs).unwrap();
        // h_start = word 0 = (1, 2) -> feats[0] = 1; h_end = word 1 =
        // (30, 40) -> feats[2] = 30. The MLP applies GELU between the layers
        // (gelu(30) == 30 in f32), so the score is gelu(1) + 30.
        let gelu1 = Tensor::from_vec(vec![1f32], (1,), &Device::Cpu)
            .unwrap()
            .gelu_erf()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let want = gelu1 + 30.0;
        assert!(
            (got[0] - want).abs() < 1e-4,
            "endpoint gather wrong: {} != {want}",
            got[0]
        );
    }

    #[test]
    fn decode_threshold_is_inclusive_and_temperature_precedes_sigmoid() {
        // Hand-built text map: caller text == normalized text, no suffix.
        let text = "Steve Jobs founded Apple.";
        let words = [(0usize, 5usize), (6, 10), (11, 18), (19, 24), (24, 25)];
        let text_map = TextMap::new(text.to_string(), text.to_string(), None, &words);
        let specs = vec![spec(0, 1)];
        let pairs = vec![RelationPair {
            relation_index: 0,
            head_start: 0,
            head_end: 2,
            tail_start: 3,
            tail_end: 4,
            head_prob: 0.9,
            tail_prob: 0.9,
            head_query_id: 0,
            tail_query_id: 1,
        }];
        // sigmoid(0.0) == 0.5 exactly: kept at threshold 0.5 (inclusive).
        let got = decode_relations_from_pairs(&text_map, &specs, &pairs, &[0.0], 1.0, 0.5).unwrap();
        assert_eq!(got.len(), 1);
        assert!((got[0].confidence - 0.5).abs() < 1e-6);
        assert_eq!(got[0].head.text, "Steve Jobs");
        assert_eq!(got[0].tail.text, "Apple");
        assert_eq!((got[0].head.char_start, got[0].head.char_end), (0, 10));
        assert_eq!((got[0].tail.char_start, got[0].tail.char_end), (19, 24));

        // relation_temperature divides BEFORE the sigmoid: logit 2.0 with
        // temperature 2.0 calibrates to sigmoid(1.0), not sigmoid(2.0)/2.
        let calibrated =
            decode_relations_from_pairs(&text_map, &specs, &pairs, &[2.0], 2.0, 0.5).unwrap();
        let want = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((calibrated[0].confidence - want).abs() < 1e-6);
        let wrong_after = (1.0 / (1.0 + (-2.0f32).exp())) / 2.0;
        assert!((calibrated[0].confidence - wrong_after).abs() > 1e-3);
    }

    #[test]
    fn decode_rejects_suffix_reaching_arguments() {
        // Caller text "Alice greeted Bob" (no terminal punctuation) gets a
        // synthetic '.' suffix; a tail argument reaching into the suffix must
        // be rejected in full (same TextMap rule as entities).
        let text = "Alice greeted Bob";
        let normalized = "Alice greeted Bob.";
        let words = [(0usize, 5usize), (6, 13), (14, 17), (17, 18)];
        let text_map = TextMap::new(text.to_string(), normalized.to_string(), Some(17), &words);
        let specs = vec![spec(0, 1)];
        let pairs = vec![
            RelationPair {
                relation_index: 0,
                head_start: 0,
                head_end: 1,
                tail_start: 2,
                tail_end: 3,
                head_prob: 1.0,
                tail_prob: 1.0,
                head_query_id: 0,
                tail_query_id: 1,
            },
            RelationPair {
                relation_index: 0,
                head_start: 0,
                head_end: 1,
                // [2, 4) = "Bob." — crosses into the synthetic suffix.
                tail_start: 2,
                tail_end: 4,
                head_prob: 1.0,
                tail_prob: 1.0,
                head_query_id: 0,
                tail_query_id: 1,
            },
        ];
        let got =
            decode_relations_from_pairs(&text_map, &specs, &pairs, &[3.0, 3.0], 1.0, 0.5).unwrap();
        assert_eq!(
            got.len(),
            1,
            "suffix-reaching pair rejected in full: {got:?}"
        );
        assert_eq!(got[0].tail.text, "Bob");
        assert_eq!((got[0].tail.char_start, got[0].tail.char_end), (14, 17));
    }
}
