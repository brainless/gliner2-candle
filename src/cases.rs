//! Versioned agent case file (epic Task 9): **JSON** (chosen over JSONL and
//! documented in README.md / DEVELOP.md).
//!
//! ```json
//! {
//!   "version": 1,
//!   "model_id": "fastino/gliner2.5-small-v1",
//!   "cases": [
//!     {
//!       "name": "readme-entities",
//!       "text": "Apple was founded by Steve Jobs in Cupertino.",
//!       "entities": ["person", "organization", "location"],
//!       "relations": ["founder"],
//!       "threshold": 0.5,
//!       "expect_entities": [
//!         {"type": "person", "text": "Steve Jobs", "byte_start": 21, "byte_end": 31,
//!          "confidence": {"min": 0.9, "max": 1.0}}
//!       ],
//!       "expect_relations": [
//!         {"type": "founder",
//!          "head": {"text": "Steve Jobs", "byte_start": 21, "byte_end": 31},
//!          "tail": {"text": "Apple", "byte_start": 0, "byte_end": 5}}
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! - `version` (required) must be `1`; unknown fields are rejected.
//! - `model_id` (optional) is checked against the `--model-id` identity when
//!   known; run the file against the checkpoint it was written for.
//! - Per case: `name` (optional, for reporting), `text` (required, may be
//!   empty), at least one of `entities`/`relations` (comma-free string
//!   lists), optional `threshold` (default 0.5), and expected results.
//! - Expected offsets are half-open UTF-8 **byte** offsets into `text` and
//!   must slice exactly to the expected `text` (validated at load).
//! - `confidence` bounds are optional `{min, max}` (either side optional):
//!   they catch large numerical drift without bitwise equality.
//! - Matching is strict set equality on (type, text, byte offsets) plus the
//!   confidence bounds of each matched item: a missing expected item, an
//!   unexpected extra item, or a confidence outside its bounds is a failure.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::boundary_rel::ExtractedRelation;
use crate::inference::ExtractedEntity;

pub const CASE_FILE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceBounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f32>,
}

/// Expected entity: label, exact surface, half-open byte offsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedEntity {
    #[serde(rename = "type")]
    pub entity_type: String,
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<ConfidenceBounds>,
}

/// Expected relation argument: exact surface + half-open byte offsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedArgument {
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// Expected directed relation edge (`head` has `type` with `tail`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedRelation {
    #[serde(rename = "type")]
    pub relation_type: String,
    pub head: ExpectedArgument,
    pub tail: ExpectedArgument,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<ConfidenceBounds>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub text: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub relations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub expect_entities: Vec<ExpectedEntity>,
    #[serde(default)]
    pub expect_relations: Vec<ExpectedRelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseFile {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub cases: Vec<Case>,
}

impl Case {
    pub fn label(&self, index: usize) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("case {}", index + 1))
    }

    pub fn threshold(&self) -> f32 {
        self.threshold.unwrap_or(0.5)
    }
}

fn check_bounds(label: &str, what: &str, b: &ConfidenceBounds) -> Result<()> {
    if let (Some(lo), Some(hi)) = (b.min, b.max) {
        if lo > hi {
            bail!("{label}: {what} confidence bounds min {lo} > max {hi}");
        }
    }
    for v in [b.min, b.max].into_iter().flatten() {
        if !(0.0..=1.0).contains(&v) || v.is_nan() {
            bail!("{label}: {what} confidence bound {v} is outside [0.0, 1.0]");
        }
    }
    Ok(())
}

/// Validate one expected span against the case text: the byte range must be
/// in bounds, non-empty, and slice exactly to the expected surface.
fn check_span(
    label: &str,
    what: &str,
    text: &str,
    expected: &str,
    start: usize,
    end: usize,
) -> Result<()> {
    if start >= end {
        bail!(
            "{label}: {what} has an empty/inverted byte range [{start}..{end}) — offsets are \
             half-open with byte_start < byte_end"
        );
    }
    let slice = text.get(start..end).ok_or_else(|| {
        anyhow::anyhow!(
            "{label}: {what} byte range [{start}..{end}) is not a valid range of the case \
             text ({} bytes; offsets are UTF-8 byte offsets)",
            text.len()
        )
    })?;
    if slice != expected {
        bail!(
            "{label}: {what} expected text {expected:?} does not match the case text at \
             bytes [{start}..{end}) ({slice:?})"
        );
    }
    Ok(())
}

fn trim_types(label: &str, what: &str, types: &[String]) -> Result<Vec<String>> {
    let trimmed: Vec<String> = types.iter().map(|t| t.trim().to_string()).collect();
    if trimmed.iter().any(|t| t.is_empty()) {
        bail!("{label}: {what} contains an empty type name — use comma-free non-empty names");
    }
    Ok(trimmed)
}

/// Parse and validate a case file (unknown fields and inconsistent
/// expectations fail here, not mid-run).
pub fn load(path: &std::path::Path) -> Result<CaseFile> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading case file {}", path.display()))?;
    let file: CaseFile = serde_json::from_str(&raw)
        .with_context(|| format!("{} is not a valid case file (JSON)", path.display()))?;
    if file.version != CASE_FILE_VERSION {
        bail!(
            "{}: unsupported case-file version {} — this build understands version \
             {CASE_FILE_VERSION}",
            path.display(),
            file.version
        );
    }
    if file.cases.is_empty() {
        bail!("{}: case file contains no cases", path.display());
    }
    for (i, case) in file.cases.iter().enumerate() {
        let label = case.label(i);
        let entities = trim_types(&label, "entities", &case.entities)?;
        let relations = trim_types(&label, "relations", &case.relations)?;
        if entities.is_empty() && relations.is_empty() {
            bail!("{label}: needs at least one of \"entities\" / \"relations\"");
        }
        if let Some(t) = case.threshold {
            if !(0.0..=1.0).contains(&t) || t.is_nan() {
                bail!("{label}: threshold {t} is outside [0.0, 1.0]");
            }
        }
        for e in &case.expect_entities {
            check_span(
                &label,
                &format!("expected entity type={:?}", e.entity_type),
                &case.text,
                &e.text,
                e.byte_start,
                e.byte_end,
            )?;
            if let Some(b) = &e.confidence {
                check_bounds(&label, &format!("entity type={:?}", e.entity_type), b)?;
            }
        }
        for r in &case.expect_relations {
            check_span(
                &label,
                &format!("expected relation type={:?} head", r.relation_type),
                &case.text,
                &r.head.text,
                r.head.byte_start,
                r.head.byte_end,
            )?;
            check_span(
                &label,
                &format!("expected relation type={:?} tail", r.relation_type),
                &case.text,
                &r.tail.text,
                r.tail.byte_start,
                r.tail.byte_end,
            )?;
            if let Some(b) = &r.confidence {
                check_bounds(&label, &format!("relation type={:?}", r.relation_type), b)?;
            }
        }
    }
    Ok(file)
}

/// Inference results for one case (the production outputs, kept for the
/// report).
#[derive(Debug, Clone, Default)]
pub struct CaseOutcome {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

fn entity_matches(e: &ExtractedEntity, x: &ExpectedEntity) -> bool {
    e.entity_type == x.entity_type
        && e.text == x.text
        && e.char_start == x.byte_start
        && e.char_end == x.byte_end
}

fn relation_matches(r: &ExtractedRelation, x: &ExpectedRelation) -> bool {
    r.relation_type == x.relation_type
        && r.head.text == x.head.text
        && r.head.char_start == x.head.byte_start
        && r.head.char_end == x.head.byte_end
        && r.tail.text == x.tail.text
        && r.tail.char_start == x.tail.byte_start
        && r.tail.char_end == x.tail.byte_end
}

fn bounds_fail(b: &ConfidenceBounds, confidence: f32) -> bool {
    b.min.is_some_and(|lo| confidence < lo) || b.max.is_some_and(|hi| confidence > hi)
}

fn fmt_bounds(b: &ConfidenceBounds) -> String {
    let lo = b
        .min
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "..".to_string());
    let hi = b
        .max
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "..".to_string());
    format!("[{lo}..{hi}]")
}

/// Compare one case's expectations against the production output. Returns
/// one line per mismatch; an empty vec is a pass. Matching is strict set
/// equality on (type, text, byte offsets), then confidence bounds on each
/// matched pair — so an added or dropped extraction fails even when every
/// expected one is present.
pub fn evaluate(case: &Case, outcome: &CaseOutcome) -> Vec<String> {
    let mut mismatches = Vec::new();

    // Entities: match expected → actual, then flag leftovers on both sides.
    let mut used = vec![false; outcome.entities.len()];
    for x in &case.expect_entities {
        let found = outcome
            .entities
            .iter()
            .enumerate()
            .find(|(i, e)| !used[*i] && entity_matches(e, x))
            .map(|(i, _)| i);
        match found {
            Some(i) => {
                used[i] = true;
                let e = &outcome.entities[i];
                if let Some(b) = &x.confidence {
                    if bounds_fail(b, e.confidence) {
                        mismatches.push(format!(
                            "entity confidence out of bounds: type={} text={:?} bytes [{}..{}) \
                             confidence={:.4} bounds={}",
                            e.entity_type,
                            e.text,
                            e.char_start,
                            e.char_end,
                            e.confidence,
                            fmt_bounds(b)
                        ));
                    }
                }
            }
            None => mismatches.push(format!(
                "missing entity: type={} text={:?} bytes [{}..{})",
                x.entity_type, x.text, x.byte_start, x.byte_end
            )),
        }
    }
    for (i, e) in outcome.entities.iter().enumerate() {
        if !used[i] {
            mismatches.push(format!(
                "unexpected entity: type={} text={:?} bytes [{}..{}) confidence={:.4}",
                e.entity_type, e.text, e.char_start, e.char_end, e.confidence
            ));
        }
    }

    // Relations.
    let mut used = vec![false; outcome.relations.len()];
    for x in &case.expect_relations {
        let found = outcome
            .relations
            .iter()
            .enumerate()
            .find(|(i, r)| !used[*i] && relation_matches(r, x))
            .map(|(i, _)| i);
        match found {
            Some(i) => {
                used[i] = true;
                let r = &outcome.relations[i];
                if let Some(b) = &x.confidence {
                    if bounds_fail(b, r.confidence) {
                        mismatches.push(format!(
                            "relation confidence out of bounds: type={} head={:?} [{}..{}) \
                             -> tail={:?} [{}..{}) confidence={:.4} bounds={}",
                            r.relation_type,
                            r.head.text,
                            r.head.char_start,
                            r.head.char_end,
                            r.tail.text,
                            r.tail.char_start,
                            r.tail.char_end,
                            r.confidence,
                            fmt_bounds(b)
                        ));
                    }
                }
            }
            None => mismatches.push(format!(
                "missing relation: type={} head={:?} [{}..{}) -> tail={:?} [{}..{})",
                x.relation_type,
                x.head.text,
                x.head.byte_start,
                x.head.byte_end,
                x.tail.text,
                x.tail.byte_start,
                x.tail.byte_end
            )),
        }
    }
    for (i, r) in outcome.relations.iter().enumerate() {
        if !used[i] {
            mismatches.push(format!(
                "unexpected relation: type={} head={:?} [{}..{}) -> tail={:?} [{}..{}) \
                 confidence={:.4}",
                r.relation_type,
                r.head.text,
                r.head.char_start,
                r.head.char_end,
                r.tail.text,
                r.tail.char_start,
                r.tail.char_end,
                r.confidence
            ));
        }
    }

    mismatches
}
