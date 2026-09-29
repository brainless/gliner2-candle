//! Shared document candidate pool + pooled pair scorer (epic Task 6).
//!
//! Ports `gliner2/models/boundary/pool.py` at the pinned commit for the
//! checkpoint-selected `candidate_pool == "shared"` path
//! (`fastino/gliner2.5-small-v1`):
//!
//! - [`DocumentCandidatePool`] — `DocumentCandidatePool.forward`: query-max
//!   endpoint union (`amax` over queries), top-`pool_boundary_top_k` boundary
//!   selection per side, one query-agnostic Cartesian pairing pass with the
//!   `end > start` filter, per-query `min_pool_per_query` quota reservation
//!   (synthetic priority band `-MASK_LOGIT * 0.5 + rank`), then
//!   `_deduplicate_pool` (duplicate keys keep their highest-priority
//!   occurrence) capped at `pool_size` rows.
//! - [`SharedPoolScorer`] — `SharedPoolScorer.forward`: candidate features
//!   computed once (endpoint projections, 3 length features
//!   `log1p(len)`, `len / text_len`, `1/sqrt(len)`, the **marginal-free**
//!   `compat_logits` prior through `prior_projection`, optional span-content
//!   mean pooling) and scored against all queries in one pass (dot product,
//!   FiLM-conditioned MLP), then start/end marginals added **exactly once**
//!   plus inside interval evidence divided by `sqrt(length)`.
//!
//! Inactive for this checkpoint and not executed: `OverlapBiasedCandidateAttention`
//! (`candidate_attention_layers`), `EvidenceConditionedQueryAttention`
//! (`query_attention_layers`) — both are rejected at config load while
//! unported; soft-max content pooling (`content_soft_max_pool`) is likewise
//! rejected. The per-query path (`proposal.py`/`scoring.py`,
//! `boundary_proposer`/`pair_scorer` weights) is never run for a `shared`
//! checkpoint and is not ported.
//!
//! Memory is bounded by the configured top-k/budget, never an `L x L` span
//! grid: the pairing universe is `pool_boundary_top_k^2` rows (32x32 here),
//! the quota band `Q * min_pool_per_query`, and the retained pool
//! `pool_size` (192) rows; the scorer materializes `C x Q x pair_dim`
//! FiLM states (`192 x Q x 128`).
//!
//! Ordering is plain stable sorts over `f32` slices matching PyTorch's
//! `torch.argsort(..., descending=True, stable=True)` semantics exactly
//! (equal keys retain input order), so tie behaviour is deterministic.
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use crate::boundary_enc::MASK_LOGIT;
use crate::config::BoundaryHeadConfig;

/// `pool.py PooledCandidates` — the deduplicated document pool (query-agnostic
/// rows shared by every query).
pub struct PooledSpans {
    /// `[1, C, 2]` u32 half-open `[start, end)` word-boundary indices;
    /// invalid rows hold `(0, 0)` (Python zeroes them).
    pub indices: Tensor,
    /// `[1, C]` u8 — row validity after dedup + budget cap.
    pub mask: Tensor,
    /// `[1, C]` f32 — full proposal prior `compat + union_start + union_end`
    /// (Python `selected_score`); invalid rows hold [`MASK_LOGIT`].
    pub proposal_logits: Tensor,
    /// `[1, C]` f32 — **marginal-free** endpoint compatibility
    /// (`(start_proj * end_proj).sum(-1) / sqrt(boundary_dim)`); invalid rows
    /// hold `0`. The scorer consumes this as its prior so start/end marginals
    /// are counted once in the pair score (proposal.py Finding 7).
    pub compat_logits: Tensor,
}

/// `pool.py SharedPoolScorer.forward` result, kept in Python's raw
/// candidate-major `[1, C, Q]` order so the additive decomposition is
/// inspectable (epic "marginals are added once").
#[allow(dead_code)] // fields are consumed by the epic checks and Task 7 decode
pub struct ScoreParts {
    /// `[1, C, Q]` final pair logits (masked with [`MASK_LOGIT`]).
    pub pair_logits: Tensor,
    /// `[1, C, Q]` score before start/end/inside terms (dot + FiLM).
    pub pre_marginal: Tensor,
    /// `[1, C, Q]` gathered start marginals — added once, coefficient 1.
    pub start_term: Tensor,
    /// `[1, C, Q]` gathered end marginals — added once, coefficient 1.
    pub end_term: Tensor,
    /// `[1, C, Q]` inside interval evidence `/(sqrt(length))` (0 when
    /// `use_inside_evidence` is off).
    pub inside_term: Tensor,
}

/// Epic Task 6 output: per-query candidate intervals + raw pair logits
/// (pre `pair_temperature`, matching the oracle `candidate_*` fields).
#[allow(dead_code)] // fields are consumed by epic Tasks 7–8 decode paths
pub struct CandidatePoolOutput {
    /// `[1, C, 2]` u32 — query-agnostic pool rows (shared by all queries).
    pub indices: Tensor,
    /// `[1, C]` u8.
    pub valid_mask: Tensor,
    /// `[1, C]` f32 — proposal prior (invalid rows [`MASK_LOGIT`]).
    pub proposal_logits: Tensor,
    /// `[1, C]` f32 — marginal-free compat prior (invalid rows `0`).
    pub compat_logits: Tensor,
    /// `[1, Q, C]` f32 — raw pair logit per query/candidate.
    pub pair_logits: Tensor,
}

/// Stable descending order — `torch.argsort(x, descending=True, stable=True)`:
/// equal values retain input order (index tie-break ascending).
fn stable_desc_f32(values: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Stable ascending order — `torch.argsort(keys, stable=True)`.
fn stable_asc_u64(values: &[u64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].cmp(&values[b]));
    order
}

/// `proposal.py::select_top_boundaries` — top-`k` boundaries by logit with
/// invalid positions floor-filled at [`MASK_LOGIT`], stable (index
/// tie-break). Invalid slots carry index `0` and `valid = false`, exactly
/// like Python's `idx = torch.where(valid, idx, 0)`.
fn select_top_boundaries(scores: &[f32], valid: &[bool], k: usize) -> (Vec<usize>, Vec<bool>) {
    let n = scores.len();
    let k = k.min(n);
    let masked: Vec<f32> = (0..n)
        .map(|i| if valid[i] { scores[i] } else { MASK_LOGIT })
        .collect();
    let order = stable_desc_f32(&masked);
    let mut idx = Vec::with_capacity(k);
    let mut out_valid = Vec::with_capacity(k);
    for &i in order.iter().take(k) {
        out_valid.push(valid[i]);
        idx.push(if valid[i] { i } else { 0 });
    }
    (idx, out_valid)
}

/// `pool.py::_deduplicate_pool` — collapse duplicate `key = start*N + end`
/// rows keeping the highest-priority occurrence (the first occurrence in
/// score-descending order), then retain the top `capacity` rows by score.
/// Output length is exactly `capacity` (padded with `(0, false)` like
/// Python's `F.pad`). Returned keys are the post-order raw keys — callers
/// zero them on `!valid` like `DocumentCandidatePool.forward`.
fn deduplicate_pool(
    keys: &[u64],
    scores: &[f32],
    valid: &[bool],
    capacity: usize,
    n_boundaries: usize,
) -> (Vec<u64>, Vec<bool>) {
    let m = keys.len();
    let invalid_key = (n_boundaries * n_boundaries) as u64;
    let keys: Vec<u64> = (0..m)
        .map(|i| if valid[i] { keys[i] } else { invalid_key })
        .collect();
    let scores: Vec<f32> = (0..m)
        .map(|i| if valid[i] { scores[i] } else { MASK_LOGIT })
        .collect();

    // Group equal keys while keeping the highest-scoring occurrence first:
    // score-descending stable, then key-ascending stable.
    let by_score = stable_desc_f32(&scores);
    let mut keys: Vec<u64> = by_score.iter().map(|&i| keys[i]).collect();
    let mut scores: Vec<f32> = by_score.iter().map(|&i| scores[i]).collect();
    let mut valid: Vec<bool> = by_score.iter().map(|&i| valid[i]).collect();
    let by_key = stable_asc_u64(&keys);
    keys = by_key.iter().map(|&i| keys[i]).collect();
    scores = by_key.iter().map(|&i| scores[i]).collect();
    valid = by_key.iter().map(|&i| valid[i]).collect();

    let mut keep = vec![false; m];
    for i in 0..m {
        let first = i == 0 || keys[i] != keys[i - 1];
        keep[i] = valid[i] && first;
    }

    let sel_scores: Vec<f32> = (0..m)
        .map(|i| if keep[i] { scores[i] } else { MASK_LOGIT })
        .collect();
    let order = stable_desc_f32(&sel_scores);
    let take = capacity.min(m);
    let mut out_keys = Vec::with_capacity(capacity);
    let mut out_valid = Vec::with_capacity(capacity);
    for &i in order.iter().take(take) {
        out_keys.push(keys[i]);
        out_valid.push(keep[i]);
    }
    while out_keys.len() < capacity {
        out_keys.push(0);
        out_valid.push(false);
    }
    (out_keys, out_valid)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (x, y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

fn as_usize(v: &[u32]) -> Vec<usize> {
    v.iter().map(|&i| i as usize).collect()
}

fn as_bool(v: &[u8]) -> Vec<bool> {
    v.iter().map(|&i| i != 0).collect()
}

/// `pool.py DocumentCandidatePool` — one deduplicated span pool per document.
pub struct DocumentCandidatePool {
    start_projection: Linear,
    end_projection: Linear,
    boundary_dim: usize,
    pool_boundary_top_k: usize,
    pool_size: usize,
    min_pool_per_query: usize,
}

impl DocumentCandidatePool {
    /// Load `boundary_head.shared_pool_builder.*` (two `Linear(boundary_dim,
    /// boundary_dim)` endpoint projections) and the pool knobs.
    pub fn load(vb: VarBuilder, cfg: &BoundaryHeadConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        Ok(Self {
            start_projection: linear(d, d, vb.pp("start_projection"))?,
            end_projection: linear(d, d, vb.pp("end_projection"))?,
            boundary_dim: d,
            pool_boundary_top_k: cfg.pool_boundary_top_k,
            pool_size: cfg.pool_size,
            min_pool_per_query: cfg.min_pool_per_query,
        })
    }

    /// `DocumentCandidatePool.forward` (inference path: no gold injection —
    /// `gold_pairs is None` in `BoundaryHead.forward`'s eval call).
    ///
    /// Inputs: `boundary_states [1,N,d]`, `boundary_mask [1,N]` u8,
    /// `query_mask [1,Q]` u8, `start_logits`/`end_logits [1,Q,N]`.
    pub fn forward(
        &self,
        boundary_states: &Tensor,
        boundary_mask: &Tensor,
        query_mask: &Tensor,
        start_logits: &Tensor,
        end_logits: &Tensor,
    ) -> Result<PooledSpans> {
        if boundary_states.dim(0)? != 1 {
            bail!("boundary candidate pooling supports batch size 1 only (no batch inference)");
        }
        let dev = boundary_states.device();
        let b_states = boundary_states.squeeze(0)?; // [N, d]
        let n = b_states.dim(0)?;
        let d = self.boundary_dim;
        let sl = start_logits.squeeze(0)?.to_vec2::<f32>()?; // [Q, N]
        let el = end_logits.squeeze(0)?.to_vec2::<f32>()?;
        let q = sl.len();
        let bm = as_bool(&boundary_mask.squeeze(0)?.to_vec1::<u8>()?);
        let qm = as_bool(&query_mask.squeeze(0)?.to_vec1::<u8>()?);

        // Query-conditioned endpoint union (`amax` over queries with the
        // `MASK_LOGIT` floor at invalid query/boundary positions).
        let any_query = qm.iter().any(|&v| v);
        let mut union_start = vec![MASK_LOGIT; n];
        let mut union_end = vec![MASK_LOGIT; n];
        let mut union_valid = vec![false; n];
        for i in 0..n {
            let mut ms = MASK_LOGIT;
            let mut me = MASK_LOGIT;
            for qq in 0..q {
                let (vs, ve) = if bm[i] && qm[qq] {
                    (sl[qq][i], el[qq][i])
                } else {
                    (MASK_LOGIT, MASK_LOGIT)
                };
                ms = ms.max(vs);
                me = me.max(ve);
            }
            union_start[i] = ms;
            union_end[i] = me;
            union_valid[i] = bm[i] && any_query;
        }

        // Top boundary selection per side, then one query-agnostic Cartesian
        // pairing pass with the `end > start` filter.
        let (starts, starts_valid) =
            select_top_boundaries(&union_start, &union_valid, self.pool_boundary_top_k);
        let (ends, ends_valid) =
            select_top_boundaries(&union_end, &union_valid, self.pool_boundary_top_k);
        let (ks, ke) = (starts.len(), ends.len());
        let scale = 1.0 / (d as f32).sqrt();

        let start_all = self.start_projection.forward(&b_states)?.to_vec2::<f32>()?;
        let end_all = self.end_projection.forward(&b_states)?.to_vec2::<f32>()?;

        let p_count = ks * ke;
        let mut pair_s = vec![0usize; p_count];
        let mut pair_e = vec![0usize; p_count];
        let mut pair_valid = vec![false; p_count];
        let mut compat = vec![0f32; p_count];
        let mut union_score = vec![0f32; p_count];
        for i in 0..ks {
            for j in 0..ke {
                let p = i * ke + j;
                pair_s[p] = starts[i];
                pair_e[p] = ends[j];
                pair_valid[p] = starts_valid[i] && ends_valid[j] && ends[j] > starts[i];
                compat[p] = dot(&start_all[starts[i]], &end_all[ends[j]]) * scale;
                union_score[p] = compat[p] + union_start[starts[i]] + union_end[ends[j]];
            }
        }

        // Quota band: each active query's strongest `min_pool_per_query`
        // pairs are reserved at synthetic priority `-MASK_LOGIT * 0.5 + rank`
        // (above ordinary global scores, below training gold injection).
        let mut all_keys: Vec<u64> = Vec::with_capacity(q * self.min_pool_per_query + p_count);
        let mut all_scores: Vec<f32> = Vec::with_capacity(q * self.min_pool_per_query + p_count);
        let mut all_valid: Vec<bool> = Vec::with_capacity(q * self.min_pool_per_query + p_count);
        let quota = self.min_pool_per_query.min(p_count);
        if quota > 0 {
            for qq in 0..q {
                let masked: Vec<f32> = (0..p_count)
                    .map(|p| {
                        if pair_valid[p] && qm[qq] {
                            sl[qq][pair_s[p]] + el[qq][pair_e[p]] + compat[p]
                        } else {
                            MASK_LOGIT
                        }
                    })
                    .collect();
                let ranked = stable_desc_f32(&masked);
                for (r, &p) in ranked.iter().take(quota).enumerate() {
                    all_keys.push((pair_s[p] * n + pair_e[p]) as u64);
                    all_valid.push(pair_valid[p] && qm[qq]);
                    all_scores.push(-MASK_LOGIT * 0.5 + (quota - r) as f32);
                }
            }
        }
        for p in 0..p_count {
            all_keys.push((pair_s[p] * n + pair_e[p]) as u64);
            all_scores.push(union_score[p]);
            all_valid.push(pair_valid[p]);
        }

        let (sel_keys, sel_valid) =
            deduplicate_pool(&all_keys, &all_scores, &all_valid, self.pool_size, n);
        let c = sel_keys.len();
        let mut starts_s = vec![0usize; c];
        let mut ends_s = vec![0usize; c];
        for i in 0..c {
            let key = if sel_valid[i] { sel_keys[i] } else { 0 };
            starts_s[i] = (key / n as u64) as usize;
            ends_s[i] = (key % n as u64) as usize;
        }

        // Recompute the (deterministic) proposal scores for retained rows:
        // `selected_score = compat + union marginals`, `compat` marginal-free.
        let mut prop = vec![0f32; c];
        let mut comp = vec![0f32; c];
        for i in 0..c {
            if sel_valid[i] {
                let cd = dot(&start_all[starts_s[i]], &end_all[ends_s[i]]) * scale;
                comp[i] = cd;
                prop[i] = cd + union_start[starts_s[i]] + union_end[ends_s[i]];
            } else {
                comp[i] = 0.0;
                prop[i] = MASK_LOGIT;
            }
        }

        let mut idx_flat = Vec::with_capacity(2 * c);
        for i in 0..c {
            idx_flat.push(starts_s[i] as u32);
            idx_flat.push(ends_s[i] as u32);
        }
        Ok(PooledSpans {
            indices: Tensor::from_vec(idx_flat, (1, c, 2), dev)?,
            mask: Tensor::from_vec(sel_valid.iter().map(|&v| v as u8).collect(), (1, c), dev)?,
            proposal_logits: Tensor::from_vec(prop, (1, c), dev)?,
            compat_logits: Tensor::from_vec(comp, (1, c), dev)?,
        })
    }
}

/// `content.py SpanContentPooler` — mean span pooling from token prefixes
/// (`build_prefix` + `pool`). The smooth-maximum branch (`content_soft_max_pool`)
/// is rejected at config load and never executes here.
struct SpanContentPooler {
    value_projection: Linear,
    layer_norm: LayerNorm,
    content_dim: usize,
}

impl SpanContentPooler {
    /// Pool `[start, end)` row means over `text_states [L, H]` into
    /// `[C, content_dim]` (fp32 prefix, `sum / max(end - start, 1)`).
    fn pool(
        &self,
        text_states: &Tensor,
        text_mask: &Tensor,
        starts: &Tensor,
        ends: &Tensor,
    ) -> Result<Tensor> {
        let dev = text_states.device();
        let values = self.value_projection.forward(text_states)?; // [L, c]
        let m = text_mask.to_dtype(DType::F32)?.unsqueeze(1)?; // [L, 1]
        let values = values.broadcast_mul(&m)?;
        let zeros = Tensor::zeros((1, self.content_dim), DType::F32, dev)?;
        let prefix = Tensor::cat(&[&zeros, &values.cumsum(0)?], 0)?; // [L+1, c]
        let span_sum = (prefix.index_select(ends, 0)? - prefix.index_select(starts, 0)?)?;
        let s_vec = as_usize(&starts.to_vec1::<u32>()?);
        let e_vec = as_usize(&ends.to_vec1::<u32>()?);
        let len: Vec<f32> = s_vec
            .iter()
            .zip(&e_vec)
            .map(|(&s, &e)| (e - s).max(1) as f32)
            .collect();
        let len_t = Tensor::from_vec(len, (span_sum.dim(0)?, 1), dev)?;
        let pooled = span_sum.broadcast_div(&len_t)?;
        Ok(self.layer_norm.forward(&pooled)?)
    }
}

/// `pool.py SharedPoolScorer` — compute candidate features once and score all
/// queries in one FiLM-conditioned pass.
pub struct SharedPoolScorer {
    start_projection: Linear,
    end_projection: Linear,
    length_projection: Linear,
    prior_projection: Linear,
    content_pooler: Option<SpanContentPooler>,
    content_projection: Option<Linear>,
    candidate_norm: LayerNorm,
    query_projection: Linear,
    film: Linear,
    film_output_0: Linear,
    film_output_3: Linear,
    pair_dim: usize,
}

impl SharedPoolScorer {
    /// Load `boundary_head.shared_pool_scorer.*` per the flag-gated
    /// `expected_boundary_weights` spec. `candidate_layers`/`query_layers`
    /// are rejected at config load while unported (`candidate_attention_layers`
    /// / `query_attention_layers` > 0).
    pub fn load(vb: VarBuilder, cfg: &BoundaryHeadConfig, hidden_size: usize) -> Result<Self> {
        let b = cfg.boundary_dim;
        let p = cfg.pair_dim;
        let c = cfg.content_dim;
        let content_pooler = if cfg.enable_span_content {
            Some(SpanContentPooler {
                value_projection: linear(
                    hidden_size,
                    c,
                    vb.pp("content_pooler").pp("value_projection"),
                )?,
                layer_norm: layer_norm(c, 1e-5, vb.pp("content_pooler").pp("layer_norm"))?,
                content_dim: c,
            })
        } else {
            None
        };
        let content_projection = if content_pooler.is_some() {
            Some(linear(c, p, vb.pp("content_projection"))?)
        } else {
            None
        };
        Ok(Self {
            start_projection: linear(b, p, vb.pp("start_projection"))?,
            end_projection: linear(b, p, vb.pp("end_projection"))?,
            length_projection: linear(3, p, vb.pp("length_projection"))?,
            prior_projection: linear(1, p, vb.pp("prior_projection"))?,
            content_pooler,
            content_projection,
            candidate_norm: layer_norm(p, 1e-5, vb.pp("candidate_norm"))?,
            query_projection: linear(hidden_size, p, vb.pp("query_projection"))?,
            film: linear(p, 2 * p, vb.pp("film"))?,
            film_output_0: linear(p, 64, vb.pp("film_output").pp("0"))?,
            film_output_3: linear(64, 1, vb.pp("film_output").pp("3"))?,
            pair_dim: p,
        })
    }

    /// `SharedPoolScorer.forward` — see [`ScoreParts`] for the additive
    /// decomposition: `pair_logits = pre_marginal + start_term + end_term +
    /// inside_term` with the start/end marginals added **once** each.
    ///
    /// Inputs follow `BoundaryHead.forward`'s shared call: pooled rows,
    /// `[1,N,d]` boundary states, `[1,Q,H]` query states + mask, the raw
    /// `[1,Q,N]` marginal logits, `inside_prefix`/`inside_prefix_mean`
    /// (Task 5's fp32 centered prefix), and the gathered text states/mask.
    // Mirrors pool.py's 11-argument forward 1:1.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        pooled: &PooledSpans,
        boundary_states: &Tensor,
        query_states: &Tensor,
        query_mask: &Tensor,
        start_logits: &Tensor,
        end_logits: &Tensor,
        inside_prefix: Option<&Tensor>,
        inside_prefix_mean: Option<&Tensor>,
        text_states: &Tensor,
        text_mask: &Tensor,
        text_lengths: usize,
    ) -> Result<ScoreParts> {
        if boundary_states.dim(0)? != 1 {
            bail!("shared pool scoring supports batch size 1 only (no batch inference)");
        }
        let dev = boundary_states.device();
        let b_states = boundary_states.squeeze(0)?; // [N, d_b]
        let q_states = query_states.squeeze(0)?; // [Q, H]
        let qm = as_bool(&query_mask.squeeze(0)?.to_vec1::<u8>()?);
        let s_log = start_logits.squeeze(0)?; // [Q, N]
        let e_log = end_logits.squeeze(0)?;
        let n = s_log.dim(1)?;

        let idx = pooled.indices.squeeze(0)?; // [C, 2]
        let starts_u = idx.narrow(1, 0, 1)?.squeeze(1)?; // [C] u32
        let ends_u = idx.narrow(1, 1, 1)?.squeeze(1)?;
        let starts = as_usize(&starts_u.to_vec1::<u32>()?);
        let ends = as_usize(&ends_u.to_vec1::<u32>()?);
        let valid = as_bool(&pooled.mask.squeeze(0)?.to_vec1::<u8>()?);
        let c = starts.len();
        let q = q_states.dim(0)?;

        // Candidate features computed once (pool.py `SharedPoolScorer.forward`).
        let start_rep = self
            .start_projection
            .forward(&b_states)?
            .index_select(&starts_u, 0)?; // [C, p]
        let end_rep = self
            .end_projection
            .forward(&b_states)?
            .index_select(&ends_u, 0)?;
        let tl = text_lengths.max(1) as f32;
        let mut len_feats = Vec::with_capacity(3 * c);
        for (&s, &e) in starts.iter().zip(&ends) {
            let l = (e - s).max(1) as f32;
            len_feats.push((1.0 + l).ln()); // log1p(length)
            len_feats.push(l / tl); // length / text_length
            len_feats.push(l.sqrt().recip()); // 1 / sqrt(length)
        }
        let len_t = Tensor::from_vec(len_feats, (c, 3), dev)?;
        // Marginal-free compat prior (Finding 7): NOT `proposal_logits`.
        let prior = pooled.compat_logits.squeeze(0)?.unsqueeze(1)?; // [C, 1]
        let mut candidate = start_rep.broadcast_add(&end_rep)?;
        candidate = candidate.broadcast_add(&self.length_projection.forward(&len_t)?)?;
        candidate = candidate.broadcast_add(&self.prior_projection.forward(&prior)?)?;
        if let Some(cp) = &self.content_pooler {
            let content = cp.pool(
                &text_states.squeeze(0)?,
                &text_mask.squeeze(0)?,
                &starts_u,
                &ends_u,
            )?;
            let proj = self
                .content_projection
                .as_ref()
                .expect("content_projection present with content_pooler")
                .forward(&content)?;
            candidate = candidate.broadcast_add(&proj)?;
        }
        candidate = self.candidate_norm.forward(&candidate)?;
        let row_mask = pooled.mask.squeeze(0)?.to_dtype(DType::F32)?.unsqueeze(1)?; // [C, 1]
        candidate = candidate.broadcast_mul(&row_mask)?;

        let query = self.query_projection.forward(&q_states)?; // [Q, p]

        // Base score: candidate/query dot scaled by 1/sqrt(pair_dim) plus the
        // FiLM-conditioned MLP (`film` -> gamma/beta -> `film_output`).
        let scale = 1.0 / (self.pair_dim as f64).sqrt();
        let score = (candidate.matmul(&query.t()?)? * scale)?; // [C, Q]
        let film = self.film.forward(&query)?; // [Q, 2p]
        let gamma = film.narrow(1, 0, self.pair_dim)?;
        let beta = film.narrow(1, self.pair_dim, self.pair_dim)?;
        let conditioned = candidate
            .unsqueeze(1)?
            .broadcast_mul(&(&gamma.unsqueeze(0)? + 1.0)?)?
            .broadcast_add(&beta.unsqueeze(0)?)?; // [C, Q, p]
        let film_out = self
            .film_output_3
            .forward(&self.film_output_0.forward(&conditioned)?.gelu_erf()?)?
            .squeeze(2)?; // [C, Q]
        let pre_marginal = score.broadcast_add(&film_out)?;

        // Start/end marginals gathered at the selected endpoints — added
        // exactly once (see `marginals_are_added_exactly_once`).
        let s_idx = clamped_index(&starts, n, dev);
        let e_idx = clamped_index(&ends, n, dev);
        let start_term = s_log.index_select(&s_idx, 1)?.transpose(0, 1)?; // [C, Q]
        let end_term = e_log.index_select(&e_idx, 1)?.transpose(0, 1)?;

        // Inside interval evidence `/(sqrt(length))` with the fp32 mean
        // restore (`heads.py`/`scoring.py interval_prefix_score` convention).
        let mut lens = vec![0f32; c];
        for (i, (&s, &e)) in starts.iter().zip(&ends).enumerate() {
            lens[i] = e.min(n - 1).saturating_sub(s.min(n - 1)) as f32;
        }
        let inside_term = match inside_prefix {
            None => Tensor::zeros((c, q), DType::F32, dev)?,
            Some(prefix) => {
                let p = prefix.squeeze(0)?; // [Q, N]
                let interval =
                    (p.index_select(&e_idx, 1)? - p.index_select(&s_idx, 1)?)?.transpose(0, 1)?; // [C, Q]
                let mut interval = interval;
                if let Some(mean) = inside_prefix_mean {
                    // mean * (e - s) restores the centered prefix.
                    let mean = mean.squeeze(0)?.squeeze(1)?; // [Q]
                                                             // (e - s) is per candidate; broadcast over queries.
                    let ls = Tensor::from_vec(lens.clone(), (c, 1), dev)?;
                    interval = interval.broadcast_add(&mean.unsqueeze(0)?.broadcast_mul(&ls)?)?;
                }
                let div: Vec<f32> = lens.iter().map(|&l| l.max(1.0).sqrt().recip()).collect();
                interval.broadcast_mul(&Tensor::from_vec(div, (c, 1), dev)?)?
            }
        };

        let pair = pre_marginal
            .broadcast_add(&start_term)?
            .broadcast_add(&end_term)?
            .broadcast_add(&inside_term)?;
        // Python masks invalid rows and queries with the `MASK_LOGIT` sentinel.
        let mut keep = Vec::with_capacity(c * q);
        for &v in &valid {
            for &qv in &qm {
                keep.push((v && qv) as u8);
            }
        }
        let keep = Tensor::from_vec(keep, (c, q), dev)?;
        let floor = Tensor::full(MASK_LOGIT, (c, q), dev)?;
        let pair_logits = keep.where_cond(&pair, &floor)?;

        Ok(ScoreParts {
            pair_logits: pair_logits.unsqueeze(0)?,
            pre_marginal: pre_marginal.unsqueeze(0)?,
            start_term: start_term.unsqueeze(0)?,
            end_term: end_term.unsqueeze(0)?,
            inside_term: inside_term.unsqueeze(0)?,
        })
    }
}

/// Clamp gather indices to `[0, n-1]` (Python `gather_states` convention).
fn clamped_index(idx: &[usize], n: usize, dev: &Device) -> Tensor {
    let v: Vec<u32> = idx.iter().map(|&i| i.min(n - 1) as u32).collect();
    Tensor::from_vec(v, (idx.len(),), dev).expect("index tensor")
}

#[cfg(test)]
mod tests {
    //! Focused epic Task 6 checks that do not need the checkpoint: dedup,
    //! `end > start`, budget cap, tie-order determinism, and the
    //! marginals-added-once decomposition on a synthetic scorer.
    use super::*;
    use candle_core::Device;

    #[test]
    fn dedup_collapses_duplicate_keys_keeping_highest_priority() {
        // Key 7 appears twice (scores 5 and 9): the 9 copy survives.
        let keys = [7u64, 3, 7, 5];
        let scores = [5f32, 4.0, 9.0, 4.0];
        let valid = [true, true, true, true];
        let (k, v) = deduplicate_pool(&keys, &scores, &valid, 8, 4);
        let rows: Vec<(u64, bool)> = k.iter().copied().zip(v.iter().copied()).collect();
        // Order: descending kept score (9, 5, 4) with the key-ascending
        // tie-break between the two score-4 rows (key 3 before key 5);
        // padding rows are (0, false).
        assert_eq!(rows[0], (7, true));
        assert_eq!(rows[1], (3, true));
        assert_eq!(rows[2], (5, true));
        assert!(rows[3..].iter().all(|&(_, ok)| !ok), "padded rows invalid");
        assert_eq!(rows.len(), 8, "capacity padding");
    }

    #[test]
    fn dedup_keeps_highest_score_copy_and_caps_at_budget() {
        let keys = [1u64, 1, 1, 2, 3, 4];
        let scores = [1f32, 2.0, 3.0, 10.0, 9.0, 8.0];
        let valid = [true, true, true, true, true, true];
        let (k, v) = deduplicate_pool(&keys, &scores, &valid, 2, 4);
        assert_eq!(k.len(), 2);
        assert_eq!(k[0], 2, "score 10 wins the capacity");
        assert_eq!(k[1], 3, "score 9 second");
        assert!(v.iter().all(|&ok| ok));
    }

    #[test]
    fn dedup_invalid_rows_never_win() {
        let keys = [2u64, 2, 9];
        let scores = [1f32, 7.0, 3.0]; // first copy invalid despite its key
        let valid = [false, true, true];
        let (k, v) = deduplicate_pool(&keys, &scores, &valid, 4, 4);
        // Invalid entries collapse to the sentinel key and lose to valid rows.
        assert_eq!(k[0], 2);
        assert!(v[0]);
        assert_eq!(k[1], 9);
        assert!(v[1]);
    }

    #[test]
    fn tie_order_is_deterministic_and_index_ascending() {
        // Equal logits: selection keeps ascending boundary order.
        let scores = [1f32, 5.0, 5.0, 5.0, 2.0];
        let valid = [true, true, true, true, true];
        let (idx, ok) = select_top_boundaries(&scores, &valid, 3);
        assert_eq!(idx, vec![1, 2, 3], "stable sort: ties keep index order");
        assert!(ok.iter().all(|&v| v));
        // Equal pool scores: output order is (start, end) key ascending.
        let keys = [9u64, 5, 7];
        let scores = [1f32, 1.0, 1.0];
        let valid = [true, true, true];
        let (k, v) = deduplicate_pool(&keys, &scores, &valid, 3, 4);
        assert_eq!(k, vec![5, 7, 9], "tie-break is key ascending");
        assert!(v.iter().all(|&ok| ok));
        // Re-running is bit-identical.
        let (k2, _) = deduplicate_pool(&keys, &scores, &valid, 3, 4);
        assert_eq!(k, k2);
    }

    #[test]
    fn invalid_and_oob_end_pairs_never_validate() {
        // Synthetic pool with identity projections over 4 boundaries.
        let d = 2usize;
        let pool = DocumentCandidatePool {
            start_projection: Linear::new(
                Tensor::from_vec(vec![1f32, 0., 0., 1.], (d, d), &Device::Cpu).unwrap(),
                Some(Tensor::zeros(d, DType::F32, &Device::Cpu).unwrap()),
            ),
            end_projection: Linear::new(
                Tensor::from_vec(vec![1f32, 0., 0., 1.], (d, d), &Device::Cpu).unwrap(),
                Some(Tensor::zeros(d, DType::F32, &Device::Cpu).unwrap()),
            ),
            boundary_dim: d,
            pool_boundary_top_k: 4,
            pool_size: 16,
            min_pool_per_query: 2,
        };
        // 3 boundaries over a 2-token text (n=3): only end > start is valid.
        let n = 3;
        let q = 1;
        let states =
            Tensor::from_vec(vec![1f32, 0., 0.5, 0.5, 0., 1.], (1, n, d), &Device::Cpu).unwrap();
        let bmask = Tensor::from_vec(vec![1u8, 1, 1], (1, n), &Device::Cpu).unwrap();
        let qmask = Tensor::from_vec(vec![1u8], (1, q), &Device::Cpu).unwrap();
        let sl = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, q, n), &Device::Cpu).unwrap();
        let el = Tensor::from_vec(vec![3.0f32, 2.0, 1.0], (1, q, n), &Device::Cpu).unwrap();
        let pooled = pool.forward(&states, &bmask, &qmask, &sl, &el).unwrap();
        let idx = pooled.indices.to_vec3::<u32>().unwrap();
        let ok = pooled.mask.to_vec2::<u8>().unwrap();
        for (row, &keep) in idx[0].iter().zip(&ok[0]) {
            let (s, e) = (row[0] as usize, row[1] as usize);
            if keep != 0 {
                assert!(e > s, "valid pool row must satisfy end > start: [{s},{e})");
                assert!(s < n && e <= n, "row [{s},{e}) out of boundary range");
            } else {
                assert_eq!((s, e), (0, 0), "invalid rows are zeroed");
            }
        }
    }

    #[test]
    fn pool_respects_pool_size_budget() {
        let d = 2usize;
        let pool = DocumentCandidatePool {
            start_projection: Linear::new(
                Tensor::from_vec(vec![1f32, 0., 0., 1.], (d, d), &Device::Cpu).unwrap(),
                Some(Tensor::zeros(d, DType::F32, &Device::Cpu).unwrap()),
            ),
            end_projection: Linear::new(
                Tensor::from_vec(vec![1f32, 0., 0., 1.], (d, d), &Device::Cpu).unwrap(),
                Some(Tensor::zeros(d, DType::F32, &Device::Cpu).unwrap()),
            ),
            boundary_dim: d,
            pool_boundary_top_k: 8,
            pool_size: 3,
            min_pool_per_query: 2,
        };
        // 8 boundaries -> up to 28 end>start pairs; budget must hold 3 rows.
        let n = 8;
        let states = Tensor::from_vec(
            (0..n * d).map(|i| i as f32).collect::<Vec<f32>>(),
            (1, n, d),
            &Device::Cpu,
        )
        .unwrap();
        let bmask = Tensor::ones((1, n), DType::U8, &Device::Cpu).unwrap();
        let qmask = Tensor::ones((1, 1), DType::U8, &Device::Cpu).unwrap();
        let sl = Tensor::from_vec(
            (0..n).map(|i| (i as f32) * 0.7).collect::<Vec<f32>>(),
            (1, 1, n),
            &Device::Cpu,
        )
        .unwrap();
        let el = Tensor::from_vec(
            (0..n).map(|i| (i as f32) * 0.3 + 1.0).collect::<Vec<f32>>(),
            (1, 1, n),
            &Device::Cpu,
        )
        .unwrap();
        let pooled = pool.forward(&states, &bmask, &qmask, &sl, &el).unwrap();
        let ok = pooled.mask.to_vec2::<u8>().unwrap();
        assert_eq!(ok[0].len(), 3, "pool row count is exactly pool_size");
        assert!(ok[0].iter().filter(|&&v| v != 0).count() <= 3);
    }

    /// Synthetic scorer with hand-built weights: the pair score must equal
    /// `pre + start + end + inside` with the marginals appearing exactly once
    /// (epic: "Explicitly assert that marginals are added once").
    #[test]
    fn marginals_are_added_exactly_once() {
        let dev = Device::Cpu;
        let d = 2usize;
        let p = 2usize;
        let q = 2usize;
        let c = 2usize;
        let l = 2usize;
        let n = l + 1;
        let lin = |rows: Vec<f32>, out: usize, inn: usize| -> Linear {
            Linear::new(
                Tensor::from_vec(rows, (out, inn), &dev).unwrap(),
                Some(Tensor::zeros(out, DType::F32, &dev).unwrap()),
            )
        };
        let scorer = SharedPoolScorer {
            start_projection: lin(vec![1., 0., 0., 1.], p, d),
            end_projection: lin(vec![1., 0., 0., 1.], p, d),
            length_projection: lin(vec![0.; 3 * p], p, 3),
            // Prior passes into dim 0 only (a uniform shift would cancel in
            // the LayerNorm below — see the prior-channel check at the end).
            prior_projection: lin(vec![1., 0.], p, 1),
            content_pooler: None,
            content_projection: None,
            candidate_norm: LayerNorm::new(
                Tensor::ones(p, DType::F32, &dev).unwrap(),
                Tensor::zeros(p, DType::F32, &dev).unwrap(),
                1e-5,
            ),
            query_projection: lin(vec![1., 0., 0., 1.], p, d),
            film: lin(vec![0.; 2 * p * p], 2 * p, p),
            film_output_0: lin(vec![0.; 64 * p], 64, p),
            film_output_3: lin(vec![0.; 64], 1, 64),
            pair_dim: p,
        };
        let pooled = PooledSpans {
            indices: Tensor::from_vec(vec![0u32, 1, 1, 2], (1, c, 2), &dev).unwrap(),
            mask: Tensor::from_vec(vec![1u8, 1], (1, c), &dev).unwrap(),
            // Proposal priors differ from the marginal-free compat priors
            // (proposal = compat + marginals in the real pool).
            proposal_logits: Tensor::from_vec(vec![3f32, -2.], (1, c), &dev).unwrap(),
            compat_logits: Tensor::from_vec(vec![0.25f32, -0.5], (1, c), &dev).unwrap(),
        };
        let boundary_states =
            Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, n, d), &dev).unwrap();
        let query_states = Tensor::from_vec(vec![1f32, 0., 0., 1.], (1, q, d), &dev).unwrap();
        let query_mask = Tensor::from_vec(vec![1u8, 1], (1, q), &dev).unwrap();
        let start_logits =
            Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], (1, q, n), &dev).unwrap();
        let end_logits =
            Tensor::from_vec(vec![1.1f32, 1.2, 1.3, 1.4, 1.5, 1.6], (1, q, n), &dev).unwrap();
        let inside_prefix =
            Tensor::from_vec(vec![0f32, 0.5, 1.5, 0.2, 0.7, 1.2], (1, q, n), &dev).unwrap();
        let inside_mean = Tensor::from_vec(vec![0.25f32, -0.25], (1, q, 1), &dev).unwrap();
        let text_states = Tensor::zeros((1, l, d), DType::F32, &dev).unwrap();
        let text_mask = Tensor::ones((1, l), DType::U8, &dev).unwrap();

        let parts = scorer
            .forward(
                &pooled,
                &boundary_states,
                &query_states,
                &query_mask,
                &start_logits,
                &end_logits,
                Some(&inside_prefix),
                Some(&inside_mean),
                &text_states,
                &text_mask,
                l,
            )
            .unwrap();
        let pair = parts
            .pair_logits
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let pre = parts
            .pre_marginal
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let st = parts
            .start_term
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let et = parts.end_term.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let it = parts
            .inside_term
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let sl = start_logits.to_vec3::<f32>().unwrap();
        let el = end_logits.to_vec3::<f32>().unwrap();
        let starts = [0usize, 1];
        let ends = [1usize, 2];
        for ci in 0..c {
            for qi in 0..q {
                let s = starts[ci];
                let e = ends[ci];
                // Coefficient exactly 1: the gathered marginals appear once.
                assert_eq!(st[ci][qi], sl[0][qi][s], "start marginal once, coeff 1");
                assert_eq!(et[ci][qi], el[0][qi][e], "end marginal once, coeff 1");
                let len = (e - s) as f32;
                let want_inside = (inside_prefix.to_vec3::<f32>().unwrap()[0][qi][e]
                    - inside_prefix.to_vec3::<f32>().unwrap()[0][qi][s]
                    + inside_mean.to_vec3::<f32>().unwrap()[0][qi][0] * len)
                    / len.sqrt();
                assert!(
                    (it[ci][qi] - want_inside).abs() < 1e-6,
                    "inside term {} != {want_inside}",
                    it[ci][qi]
                );
                let once = pre[ci][qi] + st[ci][qi] + et[ci][qi] + it[ci][qi];
                assert!(
                    (pair[ci][qi] - once).abs() < 1e-6,
                    "pair = pre + marginals once: {} != {once}",
                    pair[ci][qi]
                );
                let twice = pre[ci][qi] + 2.0 * st[ci][qi] + 2.0 * et[ci][qi] + it[ci][qi];
                assert!(
                    (pair[ci][qi] - twice).abs() > 1e-3,
                    "marginals look double-added at (c={ci}, q={qi})"
                );
            }
        }

        // Prior-channel guard: the scorer must consume the marginal-free
        // `compat_logits` prior. Feeding `proposal_logits` (which carry the
        // marginals in the real pool) instead changes the pre-marginal score —
        // i.e. the accepted path cannot have the marginals entering twice,
        // once via the prior and once via the explicit additive terms above.
        let pooled_bug = PooledSpans {
            indices: pooled.indices.clone(),
            mask: pooled.mask.clone(),
            proposal_logits: pooled.proposal_logits.clone(),
            compat_logits: pooled.proposal_logits.clone(),
        };
        let bug = scorer
            .forward(
                &pooled_bug,
                &boundary_states,
                &query_states,
                &query_mask,
                &start_logits,
                &end_logits,
                Some(&inside_prefix),
                Some(&inside_mean),
                &text_states,
                &text_mask,
                l,
            )
            .unwrap();
        let bug_pre = bug
            .pre_marginal
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            bug_pre != pre,
            "prior channel is inert — cannot verify it carries compat only"
        );
    }
}
