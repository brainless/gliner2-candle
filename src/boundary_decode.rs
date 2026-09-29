//! Boundary entity decoder (epic Task 7) — `boundary/engine.py` entity
//! decode + `gliner2/inference/overlap.py` overlap resolution.
//!
//! Contract (pinned GLiNER2 commit; epic "Source-checked inference contract"
//! item 7):
//!
//! 1. `probability = sigmoid(pair_logit / pair_temperature)` — the division
//!    happens **before** the sigmoid.
//! 2. Per query, candidates are thresholded (`prob >= threshold`, inclusive)
//!    on valid pool rows; with `adaptive_threshold` the surviving set is
//!    additionally filled up to `round(exp(count_log_rate))` top-ranked rows
//!    (threshold hits are never removed, invalid rows never added).
//! 3. A query is suppressed entirely when `sigmoid(null_logit)` **exceeds**
//!    `abstention_threshold` (strictly greater; `enable_abstention`).
//! 4. Overlap resolution runs **per query** (never across types) with the
//!    configured `overlap_policy`; this checkpoint selects `flat`, which
//!    deduplicates exact spans to their highest-ranked representative and
//!    then runs maximum-total-score weighted interval scheduling with
//!    Python's deterministic tie behavior. `nested`/`longest` are rejected at
//!    config load (`check_supported`), so only `flat` reaches this module.
//! 5. Surviving half-open word intervals map to byte offsets via
//!    [`TextMap::candidate_bytes`] (`start_mappings[start]` /
//!    `end_mappings[end-1]`, epic item 2 — never the span path's
//!    `(start,width)` decode); candidates whose mapped range reaches into the
//!    synthetic `'.'` suffix are rejected in full. The surface is sliced from
//!    the caller's exact input and trimmed; nothing outside the caller's
//!    input is ever returned.
//!
//! Output order matches Python's `final_output`: entity types in declared
//! (query) order, and within a type the [`resolve_flat`] rank order
//! (descending confidence, then ascending start/end).
use anyhow::{bail, Result};
use std::cmp::Ordering;

use crate::boundary_enc::MASK_LOGIT;
use crate::boundary_pre::{QueryEntry, TextMap};
use crate::config::BoundaryHeadConfig;
use crate::inference::ExtractedEntity;

/// One thresholded candidate: a half-open word interval `[start, end)` with
/// its calibrated probability (post `pair_temperature`, post sigmoid).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredSpan {
    pub probability: f32,
    pub start: u32,
    pub end: u32,
}

/// f32 sigmoid matching `torch.sigmoid` on the same f32 input (the engine's
/// `torch.sigmoid(candidates.pair_logits / pair_temperature)`).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Port of `_group_scored_candidates` for **one query** (`boundary/model.py`).
///
/// * `pair_logits` — raw (pre `pair_temperature`) pool-row logits `[C]`.
/// * `eligible` — `valid_mask & query_mask` per pool row.
/// * `count_log_rate` — `Some(rate)` enables the `adaptive_threshold` fill
///   (`predicted = round(exp(rate))`, clamped to `[0, C]`); `None` thresholds
///   only.
///
/// Returns kept rows in pool-row order (Python's `keep.nonzero()` order).
pub fn threshold_query_candidates(
    pair_logits: &[f32],
    eligible: &[bool],
    indices: &[(u32, u32)],
    pair_temperature: f32,
    threshold: f32,
    count_log_rate: Option<f32>,
) -> Result<Vec<ScoredSpan>> {
    let c = pair_logits.len();
    if eligible.len() != c || indices.len() != c {
        bail!(
            "candidate slice length mismatch: pair_logits {c}, eligible {}, indices {}",
            eligible.len(),
            indices.len()
        );
    }
    // `torch.sigmoid(pair_logits / pair_temperature)` — division BEFORE sigmoid.
    let probs: Vec<f32> = pair_logits
        .iter()
        .map(|&logit| sigmoid(logit / pair_temperature))
        .collect();
    // Python compares f32 probabilities against the threshold cast to the
    // probability dtype (`threshold.to(dtype=probs.dtype)`).
    let mut keep: Vec<bool> = (0..c)
        .map(|i| eligible[i] && probs[i] >= threshold)
        .collect();

    if let Some(rate) = count_log_rate {
        // `predicted_count = torch.exp(rate).round().long().clamp(min=0, max=c)`
        // — torch.round is ties-to-even.
        let rounded = rate.exp().round_ties_even();
        let predicted = if rounded.is_finite() {
            (rounded as i64).clamp(0, c as i64) as usize
        } else {
            c
        };
        // `ranked_scores = probs.masked_fill(~eligible, MASK_LOGIT)`; stable
        // descending order (index tie-break); `rank` is the inverse
        // permutation. The fill is a union with the threshold hits: it can
        // add top-ranked rows below threshold but never removes hits and
        // never adds ineligible rows (`keep | (eligible & rank < predicted)`).
        let ranked_scores: Vec<f32> = (0..c)
            .map(|i| if eligible[i] { probs[i] } else { MASK_LOGIT })
            .collect();
        let mut order: Vec<usize> = (0..c).collect();
        order.sort_by(|&a, &b| {
            ranked_scores[b]
                .partial_cmp(&ranked_scores[a])
                .unwrap_or(Ordering::Equal)
        });
        let mut rank = vec![0usize; c];
        for (position, &row) in order.iter().enumerate() {
            rank[row] = position;
        }
        for i in 0..c {
            if eligible[i] && rank[i] < predicted {
                keep[i] = true;
            }
        }
    }

    Ok((0..c)
        .filter(|&i| keep[i])
        .map(|i| ScoredSpan {
            probability: probs[i],
            start: indices[i].0,
            end: indices[i].1,
        })
        .collect())
}

/// `overlap.py::resolve_overlaps` for the canonical `flat` policy
/// (`disallow`): exact-boundary duplicates collapse to their highest-ranked
/// representative, then **maximum-total-score weighted interval scheduling**
/// selects a non-overlapping set. Ties follow Python exactly: equal totals
/// prefer the larger set, then the lexicographically better
/// (confidence, start, end) ranking. The result is ranked by descending
/// confidence, then ascending start/end.
pub fn resolve_flat(scored: &[ScoredSpan]) -> Vec<ScoredSpan> {
    if scored.is_empty() {
        return Vec::new();
    }

    /// Python `rank_key(row) = (-score, start, end, index)`; `index` is the
    /// position in the input list (`enumerate(items)`).
    #[derive(Clone, Copy)]
    struct RankKey {
        neg_score: f64,
        start: u32,
        end: u32,
        index: usize,
    }

    fn rank_key(span: &ScoredSpan, index: usize) -> RankKey {
        RankKey {
            neg_score: -(span.probability as f64),
            start: span.start,
            end: span.end,
            index,
        }
    }

    fn cmp_rank(a: &RankKey, b: &RankKey) -> Ordering {
        a.neg_score
            .partial_cmp(&b.neg_score)
            .unwrap_or(Ordering::Equal)
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
            .then(a.index.cmp(&b.index))
    }

    /// Python `selection_key`: the rank keys of a selection, sorted; compared
    /// as sequences (lexicographic, shorter prefix is smaller).
    fn cmp_selection(a: &[RankKey], b: &[RankKey]) -> Ordering {
        for (x, y) in a.iter().zip(b.iter()) {
            match cmp_rank(x, y) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        a.len().cmp(&b.len())
    }

    let rank = |i: usize| rank_key(&scored[i], i);

    let mut ranked: Vec<usize> = (0..scored.len()).collect();
    ranked.sort_by(|&i, &j| cmp_rank(&rank(i), &rank(j)));

    // Exact-span dedup: first (highest-ranked) representative per boundary.
    let mut distinct: Vec<usize> = Vec::new();
    let mut seen: Vec<(u32, u32)> = Vec::new();
    for &i in &ranked {
        let boundaries = (scored[i].start, scored[i].end);
        if !seen.contains(&boundaries) {
            seen.push(boundaries);
            distinct.push(i);
        }
    }

    // Weighted interval scheduling over `by_end` (ascending end, then start,
    // then descending score, then input index — Python's sort key).
    let mut by_end: Vec<usize> = distinct;
    by_end.sort_by(|&i, &j| {
        let (a, b) = (&scored[i], &scored[j]);
        a.end
            .cmp(&b.end)
            .then(a.start.cmp(&b.start))
            .then(
                b.probability
                    .partial_cmp(&a.probability)
                    .unwrap_or(Ordering::Equal),
            )
            .then(i.cmp(&j))
    });

    // predecessors[pos] = last q < pos with end_q <= start_pos (else -1).
    let predecessors: Vec<isize> = by_end
        .iter()
        .enumerate()
        .map(|(pos, &i)| {
            let start = scored[i].start;
            // bisect_right(ends[0..pos], start) - 1
            let (mut lo, mut hi) = (0usize, pos);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if scored[by_end[mid]].end <= start {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as isize - 1
        })
        .collect();

    // best[k] = best (total, selection) over by_end[0..k]; totals accumulate
    // in f64 exactly like Python's float sums of the f32 scores.
    let mut best: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
    for (pos, &i) in by_end.iter().enumerate() {
        // pred = -1 → the empty prefix `best[0]`; pred = k → `best[k+1]`.
        let prev_slot = (predecessors[pos] + 1).max(0) as usize;
        let (prev_score, prev_selection) = &best[prev_slot];
        let mut with_selection = prev_selection.clone();
        with_selection.push(pos);
        let with_item = (prev_score + scored[i].probability as f64, with_selection);
        let without_item = best[pos].clone();
        let chosen = match with_item
            .0
            .partial_cmp(&without_item.0)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Greater => with_item,
            Ordering::Less => without_item,
            Ordering::Equal => {
                if with_item.1.len() > without_item.1.len() {
                    with_item
                } else if with_item.1.len() < without_item.1.len() {
                    without_item
                } else {
                    let key_of = |sel: &[usize]| -> Vec<RankKey> {
                        let mut keys: Vec<RankKey> = sel.iter().map(|&p| rank(by_end[p])).collect();
                        keys.sort_by(cmp_rank);
                        keys
                    };
                    let (with_keys, without_keys) = (key_of(&with_item.1), key_of(&without_item.1));
                    if cmp_selection(&with_keys, &without_keys) == Ordering::Less {
                        with_item
                    } else {
                        without_item
                    }
                }
            }
        };
        best.push(chosen);
    }

    let mut selected: Vec<usize> = best
        .last()
        .expect("best has the empty prefix entry")
        .1
        .iter()
        .map(|&pos| by_end[pos])
        .collect();
    selected.sort_by(|&i, &j| cmp_rank(&rank(i), &rank(j)));
    selected.into_iter().map(|i| scored[i]).collect()
}

/// Entity-decode configuration (epic Task 7), all from `boundary_head`.
pub struct BoundaryDecoder {
    pair_temperature: f32,
    adaptive_threshold: bool,
    enable_abstention: bool,
    abstention_threshold: f64,
}

impl BoundaryDecoder {
    pub fn new(cfg: &BoundaryHeadConfig) -> Self {
        Self {
            pair_temperature: cfg.pair_temperature as f32,
            adaptive_threshold: cfg.adaptive_threshold,
            enable_abstention: cfg.enable_abstention,
            abstention_threshold: cfg.abstention_threshold,
        }
    }

    /// Full entity decode for one sample: per-query threshold (+ optional
    /// adaptive fill), per-query abstention, per-query `flat` overlap
    /// resolution, and exact caller-text slicing.
    ///
    /// * `pool_indices` / `pool_valid` — the query-agnostic candidate pool
    ///   rows (`[start, end)` half-open word intervals).
    /// * `pair_logits` — `[Q][C]` raw (pre `pair_temperature`) pair logits.
    /// * `null_logits` / `count_log_rates` — per-query `[Q]` raw values from
    ///   `null_projection` / `count_head`; `None` when the head is disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_entities(
        &self,
        text_map: &TextMap,
        query_layout: &[QueryEntry],
        pool_indices: &[(u32, u32)],
        pool_valid: &[bool],
        pair_logits: &[Vec<f32>],
        null_logits: Option<&[f32]>,
        count_log_rates: Option<&[f32]>,
        threshold: f32,
    ) -> Result<Vec<ExtractedEntity>> {
        if self.adaptive_threshold && count_log_rates.is_none() {
            bail!(
                "adaptive threshold decoding requires count_log_rates \
                 (boundary_head.enable_count_head is disabled on this checkpoint)"
            );
        }
        let c = pool_indices.len();
        if pool_valid.len() != c {
            bail!(
                "candidate pool length mismatch: indices {c}, valid {}",
                pool_valid.len()
            );
        }

        let mut out = Vec::new();
        for spec in query_layout.iter().filter(|q| q.task_type == "entities") {
            let q = spec.query_id;
            let logits = pair_logits.get(q).ok_or_else(|| {
                anyhow::anyhow!(
                    "pair_logits missing row for query {q} ({} rows)",
                    pair_logits.len()
                )
            })?;
            if logits.len() != c {
                bail!(
                    "pair_logits row {q} has {} entries, expected {c} pool rows",
                    logits.len()
                );
            }

            // Abstention (`engine.py::_decode_entities`): suppress the whole
            // query when sigmoid(null) strictly exceeds the threshold.
            if self.enable_abstention {
                if let Some(nulls) = null_logits {
                    if let Some(&null_logit) = nulls.get(q) {
                        let null_prob = sigmoid(null_logit);
                        if (null_prob as f64) > self.abstention_threshold {
                            continue;
                        }
                    }
                }
            }

            // `eligible = valid_mask & query_mask` (all queries valid at B=1).
            let eligible: Vec<bool> = pool_valid.to_vec();
            let count_log_rate = if self.adaptive_threshold {
                Some(
                    count_log_rates
                        .and_then(|r| r.get(q).copied())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "adaptive threshold decoding requires count_log_rates[{q}]"
                            )
                        })?,
                )
            } else {
                None
            };
            let scored = threshold_query_candidates(
                logits,
                &eligible,
                pool_indices,
                self.pair_temperature,
                threshold,
                count_log_rate,
            )?;
            for span in resolve_flat(&scored) {
                let start = span.start as usize;
                let end = span.end as usize;
                // Half-open word interval → caller-text bytes; `None` rejects
                // empty/invalid intervals and ranges reaching into the
                // synthetic `'.'` suffix (Python would decode those against
                // its normalized text and surface the synthetic punctuation —
                // the documented user-facing difference).
                let Some(range) = text_map.candidate_bytes(start, end) else {
                    continue;
                };
                let surface = text_map.surface(range.clone()).trim();
                if surface.is_empty() {
                    continue;
                }
                out.push(ExtractedEntity {
                    text: surface.to_string(),
                    entity_type: spec.field_name.clone(),
                    char_start: range.start,
                    char_end: range.end,
                    confidence: span.probability,
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! Focused epic-Task-7 checks (no model required): the decode contract's
    //! arithmetic and selection semantics. End-to-end oracle comparison lives
    //! in `crate::boundary` (`task7_*`).
    use super::*;

    fn map(
        text: &str,
        normalized: &str,
        suffix_cp: Option<usize>,
        words: &[(usize, usize)],
    ) -> TextMap {
        TextMap::new(text.to_string(), normalized.to_string(), suffix_cp, words)
    }

    fn entities_query(query_id: usize, field_name: &str) -> QueryEntry {
        QueryEntry {
            query_id,
            task_index: 0,
            task_type: "entities".to_string(),
            task_name: "entities".to_string(),
            field_index: query_id,
            field_name: field_name.to_string(),
        }
    }

    #[test]
    fn pair_temperature_is_applied_before_sigmoid() {
        // logit 2.0 with temperature 2.0 must be sigmoid(1.0), NOT
        // sigmoid(2.0)/2 or sigmoid(logit) then scaled.
        let got = threshold_query_candidates(&[2.0], &[true], &[(0, 1)], 2.0, 0.0, None).unwrap();
        assert_eq!(got.len(), 1);
        let want = 1.0 / (1.0 + (-1.0f32).exp());
        assert!(
            (got[0].probability - want).abs() < 1e-6,
            "prob {} != sigmoid(logit/temp) {want}",
            got[0].probability
        );
        let wrong_after = 1.0 / (1.0 + (-2.0f32).exp()) / 2.0;
        let wrong_no_temp = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((got[0].probability - wrong_after).abs() > 1e-3);
        assert!((got[0].probability - wrong_no_temp).abs() > 1e-3);
    }

    #[test]
    fn threshold_is_inclusive_at_the_boundary() {
        // prob == threshold exactly must be kept (`probs >= threshold`).
        let prob = 0.5f32;
        let logit = (prob / (1.0 - prob)).ln(); // sigmoid(logit) == 0.5
        let got = threshold_query_candidates(
            &[logit, logit - 0.01],
            &[true, true],
            &[(0, 1), (1, 2)],
            1.0,
            0.5,
            None,
        )
        .unwrap();
        assert_eq!(
            got.len(),
            1,
            "exact-threshold row kept, below-threshold dropped"
        );
        assert_eq!((got[0].start, got[0].end), (0, 1));
        assert!((got[0].probability - 0.5).abs() < 1e-6);
    }

    #[test]
    fn abstention_suppresses_the_query_strictly() {
        let tmap = map("a b.", "a b.", None, &[(0, 1), (2, 3), (3, 4)]);
        let layout = vec![entities_query(0, "person")];
        let indices = vec![(0, 1)];
        let valid = vec![true];
        let pair = vec![vec![10.0]]; // passes any threshold
        let decoder = BoundaryDecoder {
            pair_temperature: 1.0,
            adaptive_threshold: false,
            enable_abstention: true,
            abstention_threshold: 0.5,
        };
        // null logit 0.0 -> sigmoid 0.5; `>` is strict, so no abstention.
        let kept = decoder
            .decode_entities(
                &tmap,
                &layout,
                &indices,
                &valid,
                &pair,
                Some(&[0.0]),
                None,
                0.5,
            )
            .unwrap();
        assert_eq!(kept.len(), 1, "sigmoid(null) == threshold must NOT abstain");
        // null logit slightly above 0 -> sigmoid > 0.5 -> suppressed.
        let abstained = decoder
            .decode_entities(
                &tmap,
                &layout,
                &indices,
                &valid,
                &pair,
                Some(&[0.001]),
                None,
                0.5,
            )
            .unwrap();
        assert!(
            abstained.is_empty(),
            "sigmoid(null) > threshold must abstain"
        );
        // With abstention disabled the query survives regardless.
        let no_abstention = BoundaryDecoder {
            enable_abstention: false,
            ..decoder
        };
        let kept = no_abstention
            .decode_entities(
                &tmap,
                &layout,
                &indices,
                &valid,
                &pair,
                Some(&[5.0]),
                None,
                0.5,
            )
            .unwrap();
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn adaptive_fill_tops_up_but_never_removes_threshold_hits() {
        // Three rows: two above threshold, one below. Predicted count 3 fills
        // the gap; predicted count 1 must still keep both threshold hits.
        let logits = vec![3.0f32, 2.0, -2.0]; // probs ~0.953, ~0.881, ~0.119
        let eligible = vec![true, true, true];
        let indices = vec![(0, 1), (1, 2), (2, 3)];
        // rate = ln(3) -> predicted 3: all rows kept.
        let filled =
            threshold_query_candidates(&logits, &eligible, &indices, 1.0, 0.5, Some(3f32.ln()))
                .unwrap();
        assert_eq!(
            filled.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            vec![(0, 1), (1, 2), (2, 3)],
            "adaptive fill adds the top-ranked below-threshold row"
        );
        // rate = ln(0.5) -> round 0.5 ties-to-even -> 0: only threshold hits.
        let zero_count =
            threshold_query_candidates(&logits, &eligible, &indices, 1.0, 0.5, Some(0.5f32.ln()))
                .unwrap();
        assert_eq!(
            zero_count.len(),
            2,
            "predicted count 0 keeps threshold hits only"
        );
        // rate = ln(10) -> clamped to C=3; ineligible rows are never added.
        let clamped = threshold_query_candidates(
            &logits,
            &[true, true, false],
            &indices,
            1.0,
            0.5,
            Some(10f32.ln()),
        )
        .unwrap();
        assert_eq!(
            clamped.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            vec![(0, 1), (1, 2)],
            "clamp to C never adds ineligible rows"
        );
        // Without adaptive fill the below-threshold row stays out.
        let plain =
            threshold_query_candidates(&logits, &eligible, &indices, 1.0, 0.5, None).unwrap();
        assert_eq!(plain.len(), 2);
    }

    #[test]
    fn adaptive_fill_requires_count_log_rates() {
        let decoder = BoundaryDecoder {
            pair_temperature: 1.0,
            adaptive_threshold: true,
            enable_abstention: false,
            abstention_threshold: 0.5,
        };
        let tmap = map("a.", "a.", None, &[(0, 1), (1, 2)]);
        let err = decoder
            .decode_entities(
                &tmap,
                &[entities_query(0, "person")],
                &[(0, 1)],
                &[true],
                &[vec![2.0]],
                None,
                None,
                0.5,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("count_log_rates"),
            "missing count head must fail clearly: {err}"
        );
    }

    #[test]
    fn weighted_interval_scheduling_is_optimal_and_deterministic() {
        // Case 02 pattern: two adjacent smaller scores beat one larger span.
        let spans = vec![
            ScoredSpan {
                probability: 0.838,
                start: 0,
                end: 4,
            },
            ScoredSpan {
                probability: 0.823,
                start: 2,
                end: 4,
            },
            ScoredSpan {
                probability: 0.711,
                start: 0,
                end: 2,
            },
        ];
        let got = resolve_flat(&spans);
        assert_eq!(
            got.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            vec![(2, 4), (0, 2)],
            "WIS must pick the 0.823+0.711 pair over the single 0.838 span"
        );
        // Nested: the best total is the outer span alone (0.9 > 0.4+0.45).
        let nested = vec![
            ScoredSpan {
                probability: 0.40,
                start: 1,
                end: 3,
            },
            ScoredSpan {
                probability: 0.90,
                start: 0,
                end: 4,
            },
            ScoredSpan {
                probability: 0.45,
                start: 2,
                end: 4,
            },
        ];
        let got = resolve_flat(&nested);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].start, got[0].end), (0, 4));
        // Crossing equal scores: the lexicographically better ranking wins
        // (ascending start), deterministically across reruns.
        let crossing = vec![
            ScoredSpan {
                probability: 0.6,
                start: 0,
                end: 3,
            },
            ScoredSpan {
                probability: 0.6,
                start: 2,
                end: 5,
            },
        ];
        for _ in 0..4 {
            let got = resolve_flat(&crossing);
            assert_eq!(got.len(), 1);
            assert_eq!(
                (got[0].start, got[0].end),
                (0, 3),
                "tie prefers lower start"
            );
        }
        // Chain of disjoint spans keeps everything, sorted by score.
        let disjoint = vec![
            ScoredSpan {
                probability: 0.5,
                start: 4,
                end: 6,
            },
            ScoredSpan {
                probability: 0.7,
                start: 0,
                end: 2,
            },
            ScoredSpan {
                probability: 0.6,
                start: 2,
                end: 4,
            },
        ];
        let got = resolve_flat(&disjoint);
        assert_eq!(
            got.iter().map(|s| s.start).collect::<Vec<_>>(),
            vec![0, 2, 4],
            "output ranked by descending confidence"
        );
    }

    #[test]
    fn exact_span_duplicates_collapse_to_highest_ranked() {
        let spans = vec![
            ScoredSpan {
                probability: 0.5,
                start: 1,
                end: 3,
            },
            ScoredSpan {
                probability: 0.8,
                start: 1,
                end: 3,
            },
            ScoredSpan {
                probability: 0.8,
                start: 1,
                end: 3,
            },
            ScoredSpan {
                probability: 0.9,
                start: 5,
                end: 6,
            },
        ];
        let got = resolve_flat(&spans);
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].probability, got[0].start, got[0].end), (0.9, 5, 6));
        assert_eq!(
            (got[1].probability, got[1].start, got[1].end),
            (0.8, 1, 3),
            "exact-span dedup keeps the highest-ranked copy only"
        );
    }

    #[test]
    fn suffix_rejection_case_06() {
        // Oracle case 06: caller text without terminal punctuation. Words 0..8
        // map the caller's text; word 8 is the synthetic '.'.
        //                 apple    was  founded   by steve jobs   in cupertino    .
        let words = [
            (0, 5),
            (6, 9),
            (10, 17),
            (18, 20),
            (21, 26),
            (27, 31),
            (32, 34),
            (35, 44),
            (44, 45),
        ];
        let text = "Apple was founded by Steve Jobs in Cupertino";
        let normalized = "Apple was founded by Steve Jobs in Cupertino.";
        let tmap = map(text, normalized, Some(44), &words);
        let layout = vec![entities_query(0, "location")];
        let decoder = BoundaryDecoder {
            pair_temperature: 1.0,
            adaptive_threshold: false,
            enable_abstention: true,
            abstention_threshold: 0.5,
        };
        // Disjoint rows: [7,9) swallows the synthetic '.' (Python would surface
        // "Cupertino." from its normalized text) and must be dropped in full,
        // while the clean [4,6) survives.
        let got = decoder
            .decode_entities(
                &tmap,
                &layout,
                &[(7, 9), (4, 6)],
                &[true, true],
                &[vec![10.0, 9.0]],
                Some(&[-5.0]),
                None,
                0.5,
            )
            .unwrap();
        assert_eq!(
            got.len(),
            1,
            "candidate crossing into the synthetic suffix must be dropped"
        );
        assert_eq!(got[0].text, "Steve Jobs");
        assert_eq!((got[0].char_start, got[0].char_end), (21, 31));
        // Nothing outside the caller's input is ever returned.
        for entity in &got {
            assert!(entity.char_end <= text.len());
        }
        // Only the suffix-crossing row: empty output.
        let only_suffix = decoder
            .decode_entities(
                &tmap,
                &layout,
                &[(7, 9)],
                &[true],
                &[vec![10.0]],
                Some(&[-5.0]),
                None,
                0.5,
            )
            .unwrap();
        assert!(only_suffix.is_empty());
        // The clean half-open mapping itself: [7,8) → bytes 35..44.
        let clean = decoder
            .decode_entities(
                &tmap,
                &layout,
                &[(7, 8)],
                &[true],
                &[vec![10.0]],
                Some(&[-5.0]),
                None,
                0.5,
            )
            .unwrap();
        assert_eq!(clean.len(), 1);
        assert_eq!(clean[0].text, "Cupertino");
        assert_eq!((clean[0].char_start, clean[0].char_end), (35, 44));
        // Shadowing divergence: [7,9) and [7,8) overlap, `flat` WIS keeps the
        // higher-scoring [7,9), which the suffix rule then rejects — Rust
        // emits nothing here. Python resolves overlaps first and surfaces
        // "Cupertino." from the normalized text (the documented user-facing
        // difference from the epic §1 suffix rule).
        let shadowed = decoder
            .decode_entities(
                &tmap,
                &layout,
                &[(7, 9), (7, 8)],
                &[true, true],
                &[vec![10.0, 9.0]],
                Some(&[-5.0]),
                None,
                0.5,
            )
            .unwrap();
        assert!(
            shadowed.is_empty(),
            "suffix-crossing span shadows its clean neighbor (Python emits 'Cupertino.')"
        );
    }

    #[test]
    fn empty_text_case_10_rejects_every_candidate() {
        // Oracle case 10: caller text "" normalizes to "."; the single word is
        // the synthetic suffix, so every candidate is rejected.
        let tmap = map("", ".", Some(0), &[(0, 1)]);
        let layout = vec![
            entities_query(0, "person"),
            entities_query(1, "organization"),
            entities_query(2, "location"),
        ];
        let indices = vec![(0, 1)];
        let valid = vec![true];
        let pair = vec![vec![10.0], vec![10.0], vec![10.0]];
        let decoder = BoundaryDecoder {
            pair_temperature: 1.0,
            adaptive_threshold: false,
            enable_abstention: true,
            abstention_threshold: 0.5,
        };
        let got = decoder
            .decode_entities(
                &tmap,
                &layout,
                &indices,
                &valid,
                &pair,
                Some(&[-5.0, -5.0, -5.0]),
                None,
                0.5,
            )
            .unwrap();
        assert!(
            got.is_empty(),
            "empty caller text can never surface entities"
        );
    }
}
