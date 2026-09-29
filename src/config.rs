use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::path::Path;

/// Model architecture declared by `config.json`.
///
/// Mirrors Python's `normalize_architecture` (gliner2/configuration.py): a
/// missing (or null / empty) `architecture` field resolves to `Span` for legacy
/// checkpoints; unknown names are a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum Architecture {
    #[default]
    Span,
    Boundary,
}

impl Architecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Architecture::Span => "span",
            Architecture::Boundary => "boundary",
        }
    }

    /// Normalize like Python: trim + lowercase, then match known names.
    /// `None` (missing field) means the legacy `span` architecture.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None => Ok(Architecture::Span),
            Some("") => Ok(Architecture::Span),
            Some(s) => match s.trim().to_ascii_lowercase().as_str() {
                "span" => Ok(Architecture::Span),
                "boundary" => Ok(Architecture::Boundary),
                _ => bail!(
                    "Unknown extractor architecture {s:?}.\n\
                     Expected one of: \"span\", \"boundary\"."
                ),
            },
        }
    }
}

impl<'de> Deserialize<'de> for Architecture {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value: Option<String> = Option::deserialize(deserializer)?;
        Architecture::parse(value.as_deref()).map_err(serde::de::Error::custom)
    }
}

/// GLiNER2-specific model config stored in `config.json` on the HF repo.
/// Mirrors Python's `ExtractorConfig` (post `migrate_config_dict`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Gliner2Config {
    /// HF model ID of the encoder backbone, e.g. "microsoft/deberta-v2-large"
    #[serde(default = "default_model_name")]
    pub model_name: String,

    /// Maximum span width in words (span architecture only; default 8)
    #[serde(default = "default_max_width")]
    pub max_width: usize,

    /// Which counting layer variant: "count_lstm" | "count_lstm_moe" | "count_lstm_v2"
    #[serde(default = "default_counting_layer")]
    pub counting_layer: String,

    /// How to pool subword tokens into word embeddings: "first" | "mean" | "max"
    #[serde(default = "default_token_pooling")]
    pub token_pooling: String,

    /// Optional word-level truncation length
    pub max_len: Option<usize>,

    /// Architecture selector — missing means the legacy span checkpoint.
    #[serde(default)]
    pub architecture: Architecture,

    /// Serialized config schema version (boundary migration input; 0 = legacy)
    #[serde(default)]
    pub config_version: u32,

    /// Boundary-head settings (active for `architecture == "boundary"`).
    /// Deserialized with Python's `BoundaryHeadSettings` defaults for any
    /// missing field.
    #[serde(default)]
    pub boundary_head: BoundaryHeadConfig,
}

fn default_model_name() -> String {
    "microsoft/deberta-v2-large".to_string()
}
fn default_max_width() -> usize {
    8
}
fn default_counting_layer() -> String {
    "count_lstm".to_string()
}
fn default_token_pooling() -> String {
    "first".to_string()
}

/// Boundary-head settings (gliner2/configuration.py `BoundaryHeadSettings` +
/// `validate_boundary_head`).
///
/// Only fields that are active for inference are represented. Training-only
/// knobs (loss weights, focal loss, hard-negative mining, `dropout`,
/// `training_candidate_budget`, `max_gold_per_query`, record/classification
/// loss weights, …) are ignored by design: they never change inference math.
/// Record/classification *tasks* stay out of epic scope entirely.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct BoundaryHeadConfig {
    // ── Dims / boundary encoding (both candidate pools) ─────────────────────
    pub boundary_dim: usize,
    pub pair_dim: usize,
    pub content_dim: usize,
    pub boundary_refinement_layers: usize,
    pub boundary_ffn_multiplier: f64,
    pub boundary_attention_layers: usize,
    pub boundary_attention_heads: usize,
    pub boundary_attention_window: usize,
    pub use_inside_evidence: bool,

    // ── Candidate pool selection / proposals ────────────────────────────────
    pub candidate_pool: String,
    pub pool_boundary_top_k: usize,
    pub pool_size: usize,
    pub min_pool_per_query: usize,
    pub candidate_attention_layers: usize,
    pub candidate_attention_heads: usize,
    pub query_attention_layers: usize,
    pub start_top_k: usize,
    pub end_top_k: usize,
    pub ends_per_start: usize,
    pub starts_per_end: usize,
    pub candidate_budget: usize,
    pub end_block_size: usize,
    pub bidirectional_proposals: bool,
    pub boundary_top_k_alpha: f64,
    pub boundary_top_k_max: usize,
    pub boundary_top_k_bucket: usize,
    pub export_mode: String,
    pub vectorized_pair_elements: usize,

    // ── Scoring flags (which terms/weights are active) ──────────────────────
    pub enable_span_content: bool,
    pub content_soft_max_pool: bool,
    pub enable_rotary_endpoints: bool,
    pub rotary_base: f64,
    pub query_conditioned_inside_weight: bool,
    pub endpoint_difference_features: bool,
    pub reranker_endpoint_compat: bool,
    pub multihead_pair_compat_heads: usize,

    // ── Entity decode ──────────────────────────────────────────────────────
    pub pair_temperature: f64,
    pub adaptive_threshold: bool,
    pub enable_abstention: bool,
    pub abstention_threshold: f64,
    pub enable_count_head: bool,
    pub overlap_policy: String,

    // ── Relations (epic Task 8) ────────────────────────────────────────────
    pub enable_relations: bool,
    pub relation_heads_per_type: usize,
    pub relation_tails_per_type: usize,
    pub relation_pair_cap: usize,
    pub relation_argument_proposal_threshold: f64,
    pub directional_relation_states: bool,
    pub relation_biaffine_content: bool,
    pub relation_temperature: f64,

    // ── Records flag (parsed for awareness; the task itself is out of scope
    //    and is never exposed or run) ───────────────────────────────────────
    pub enable_records: bool,
}

/// Defaults must match Python's `BoundaryHeadSettings` exactly.
impl Default for BoundaryHeadConfig {
    fn default() -> Self {
        Self {
            boundary_dim: 128,
            pair_dim: 128,
            content_dim: 64,
            boundary_refinement_layers: 1,
            boundary_ffn_multiplier: 2.0,
            boundary_attention_layers: 0,
            boundary_attention_heads: 4,
            boundary_attention_window: 0,
            use_inside_evidence: true,
            candidate_pool: "per_query".to_string(),
            pool_boundary_top_k: 64,
            pool_size: 384,
            min_pool_per_query: 8,
            candidate_attention_layers: 2,
            candidate_attention_heads: 4,
            query_attention_layers: 1,
            start_top_k: 16,
            end_top_k: 16,
            ends_per_start: 8,
            starts_per_end: 8,
            candidate_budget: 128,
            end_block_size: 256,
            bidirectional_proposals: true,
            boundary_top_k_alpha: 0.0,
            boundary_top_k_max: 128,
            boundary_top_k_bucket: 8,
            export_mode: "auto".to_string(),
            vectorized_pair_elements: 16_777_216,
            enable_span_content: false,
            content_soft_max_pool: false,
            enable_rotary_endpoints: false,
            rotary_base: 10000.0,
            query_conditioned_inside_weight: false,
            endpoint_difference_features: false,
            reranker_endpoint_compat: true,
            multihead_pair_compat_heads: 8,
            pair_temperature: 1.0,
            adaptive_threshold: false,
            enable_abstention: true,
            abstention_threshold: 0.5,
            enable_count_head: true,
            overlap_policy: "flat".to_string(),
            enable_relations: true,
            relation_heads_per_type: 32,
            relation_tails_per_type: 32,
            relation_pair_cap: 128,
            relation_argument_proposal_threshold: 0.0,
            directional_relation_states: false,
            relation_biaffine_content: false,
            relation_temperature: 1.0,
            enable_records: true,
        }
    }
}

impl BoundaryHeadConfig {
    /// Port of the inference-relevant dimension/range/enum rules of Python's
    /// `validate_boundary_head`. Training-only rules (loss weights, focal
    /// loss, hard negatives, `training_candidate_budget`, `max_gold_per_query`
    /// …) are not ported because those fields are not parsed.
    pub fn validate(&self) -> Result<()> {
        for (key, v) in [
            ("boundary_dim", self.boundary_dim),
            ("pair_dim", self.pair_dim),
            ("start_top_k", self.start_top_k),
            ("end_top_k", self.end_top_k),
            ("ends_per_start", self.ends_per_start),
            ("starts_per_end", self.starts_per_end),
            ("candidate_budget", self.candidate_budget),
            ("end_block_size", self.end_block_size),
        ] {
            if v == 0 {
                bail!("boundary_head.{key} must be > 0, got {v}");
            }
        }
        if self.content_dim == 0 {
            bail!("boundary_head.content_dim must be > 0");
        }
        if self.boundary_ffn_multiplier <= 0.0 {
            bail!("boundary_head.boundary_ffn_multiplier must be > 0");
        }
        if self.rotary_base <= 0.0 {
            bail!("boundary_head.rotary_base must be > 0");
        }
        if self.enable_rotary_endpoints
            && (!self.boundary_dim.is_multiple_of(2) || !self.pair_dim.is_multiple_of(2))
        {
            bail!(
                "boundary_head.enable_rotary_endpoints requires even boundary_dim \
                 and pair_dim, got {} and {}",
                self.boundary_dim,
                self.pair_dim
            );
        }
        if self.boundary_attention_heads == 0 {
            bail!("boundary_head.boundary_attention_heads must be > 0");
        }
        if self.boundary_attention_layers > 0
            && !self
                .boundary_dim
                .is_multiple_of(self.boundary_attention_heads)
        {
            bail!(
                "boundary_head.boundary_dim must be divisible by boundary_attention_heads \
                 when attention is enabled"
            );
        }
        if self.multihead_pair_compat_heads == 0 {
            bail!("boundary_head.multihead_pair_compat_heads must be > 0");
        }
        if !self
            .pair_dim
            .is_multiple_of(self.multihead_pair_compat_heads)
        {
            bail!(
                "boundary_head.pair_dim must be divisible by multihead_pair_compat_heads, \
                 got {} and {}",
                self.pair_dim,
                self.multihead_pair_compat_heads
            );
        }
        if self.boundary_top_k_alpha < 0.0 {
            bail!("boundary_head.boundary_top_k_alpha must be >= 0");
        }
        if self.boundary_top_k_max < self.start_top_k.max(self.end_top_k) {
            bail!("boundary_head.boundary_top_k_max must be >= start_top_k and end_top_k");
        }
        if self.boundary_top_k_bucket == 0 {
            bail!("boundary_head.boundary_top_k_bucket must be > 0");
        }
        match self.candidate_pool.as_str() {
            "per_query" | "shared" => {}
            other => {
                bail!("boundary_head.candidate_pool must be 'per_query' or 'shared', got {other:?}")
            }
        }
        for key in ["pool_boundary_top_k", "pool_size"] {
            let v = if key == "pool_boundary_top_k" {
                self.pool_boundary_top_k
            } else {
                self.pool_size
            };
            if v == 0 {
                bail!("boundary_head.{key} must be > 0");
            }
        }
        if self.min_pool_per_query > self.pool_size {
            bail!("boundary_head.min_pool_per_query must not exceed pool_size");
        }
        if self.candidate_attention_heads == 0 {
            bail!("boundary_head.candidate_attention_heads must be > 0");
        }
        if (self.candidate_attention_layers > 0 || self.query_attention_layers > 0)
            && !self.pair_dim.is_multiple_of(self.candidate_attention_heads)
        {
            bail!(
                "boundary_head.pair_dim must be divisible by candidate_attention_heads \
                 when candidate or query attention is enabled"
            );
        }
        if !(0.0..=1.0).contains(&self.abstention_threshold) {
            bail!("boundary_head.abstention_threshold must be in [0, 1]");
        }
        match self.overlap_policy.as_str() {
            "flat" | "nested" | "longest" => {}
            other => bail!(
                "boundary_head.overlap_policy must be 'flat', 'nested', or 'longest', got {other:?}"
            ),
        }
        for (key, v) in [
            ("pair_temperature", self.pair_temperature),
            ("relation_temperature", self.relation_temperature),
        ] {
            if v <= 0.0 {
                bail!("boundary_head.{key} must be > 0");
            }
        }
        match self.export_mode.as_str() {
            "auto" | "streaming" | "vectorized" => {}
            other => bail!(
                "boundary_head.export_mode must be 'auto', 'streaming', or 'vectorized', \
                 got {other:?}"
            ),
        }
        if self.export_mode == "vectorized" && self.boundary_top_k_alpha > 0.0 {
            bail!(
                "boundary_head.export_mode='vectorized' is incompatible with an adaptive \
                 boundary budget; use 'auto'/'streaming' or set boundary_top_k_alpha=0"
            );
        }
        if self.vectorized_pair_elements == 0 {
            bail!("boundary_head.vectorized_pair_elements must be > 0");
        }
        for key in [
            "relation_heads_per_type",
            "relation_tails_per_type",
            "relation_pair_cap",
        ] {
            let v = match key {
                "relation_heads_per_type" => self.relation_heads_per_type,
                "relation_tails_per_type" => self.relation_tails_per_type,
                _ => self.relation_pair_cap,
            };
            if v == 0 {
                bail!("boundary_head.{key} must be > 0, got {v}");
            }
        }
        if !(0.0..=1.0).contains(&self.relation_argument_proposal_threshold) {
            bail!("boundary_head.relation_argument_proposal_threshold must be in [0, 1]");
        }
        Ok(())
    }

    /// Refuse *active* modes this port cannot implement — silently running
    /// different math is a documented bug class (see epic "Existing Rust
    /// constraints"). Inactive modes (e.g. per_query scoring flags while
    /// `candidate_pool == "shared"`) are not gated here.
    pub fn check_supported(&self) -> Result<()> {
        if self.overlap_policy != "flat" {
            bail!(
                "unsupported boundary_head.overlap_policy {:?}: only \"flat\" is implemented \
                 (nested/longest overlap resolution is not ported); refusing to silently run \
                 different overlap math",
                self.overlap_policy
            );
        }
        if self.candidate_pool == "shared"
            && (self.candidate_attention_layers > 0 || self.query_attention_layers > 0)
        {
            bail!(
                "unsupported boundary_head shared-pool mode: candidate_attention_layers={} \
                 and query_attention_layers={} need attention blocks that are not implemented \
                 yet (epic Task 6); refusing to load a checkpoint whose scoring depends on them",
                self.candidate_attention_layers,
                self.query_attention_layers
            );
        }
        if self.enable_span_content && self.content_soft_max_pool {
            bail!(
                "unsupported boundary_head.content_soft_max_pool=true: smooth-maximum span \
                 content pooling (SpanContentPooler LSE branch) is not implemented (epic Task 6 \
                 ports the mean-pooling path only); refusing to load a checkpoint whose scoring \
                 depends on it"
            );
        }
        Ok(())
    }
}

/// Mirror of Python's `migrate_config_dict` (gliner2/configuration.py).
///
/// Resolves the architecture name and, for boundary configs with
/// `config_version < 3`, defaults a missing `enable_records` /
/// `enable_relations` to false so old checkpoints do not silently gain tasks.
fn migrate_config_dict(mut value: serde_json::Value) -> Result<serde_json::Value> {
    let obj = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("config.json must contain a JSON object"))?;

    let architecture = match obj.get("architecture") {
        None | Some(serde_json::Value::Null) => Architecture::Span,
        Some(serde_json::Value::String(s)) => Architecture::parse(Some(s))?,
        Some(other) => {
            return Err(anyhow!(
                "config.json \"architecture\" must be a string, got {other}"
            ))
        }
    };
    obj.insert(
        "architecture".to_string(),
        serde_json::Value::String(architecture.as_str().to_string()),
    );

    if architecture == Architecture::Boundary {
        let config_version = obj
            .get("config_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        match obj.get("boundary_head") {
            None | Some(serde_json::Value::Null) => {
                obj.insert("boundary_head".to_string(), serde_json::json!({}));
            }
            Some(serde_json::Value::Object(_)) => {}
            Some(other) => {
                return Err(anyhow!(
                    "config.json \"boundary_head\" must be a JSON object, got {other}"
                ))
            }
        }
        let bh = obj
            .get_mut("boundary_head")
            .and_then(|v| v.as_object_mut())
            .expect("boundary_head replaced by an object above");
        if config_version < 3 {
            bh.entry("enable_records")
                .or_insert(serde_json::Value::Bool(false));
            bh.entry("enable_relations")
                .or_insert(serde_json::Value::Bool(false));
        }
    }
    Ok(value)
}

impl Gliner2Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {:?}", path.as_ref()))?;
        Self::from_json_str(&raw).with_context(|| format!("parsing {:?}", path.as_ref()))
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(raw).context("parsing config.json")?;
        let migrated = migrate_config_dict(value)?;
        let cfg: Self = serde_json::from_value(migrated).context("parsing Gliner2Config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        // Active-mode gates shared by both architectures.
        if self.token_pooling != "first" {
            bail!(
                "unsupported token_pooling {:?}: this port only implements \"first\" \
                 (first-subtoken word pooling); refusing to silently run different pooling math",
                self.token_pooling
            );
        }
        match self.architecture {
            Architecture::Span => {
                if self.max_width == 0 {
                    bail!("span max_width must be > 0, got {}", self.max_width);
                }
                if self.counting_layer != "count_lstm" {
                    bail!(
                        "unsupported counting_layer {:?}: this port only implements \
                         \"count_lstm\"; refusing to silently run different counting math",
                        self.counting_layer
                    );
                }
            }
            Architecture::Boundary => {
                self.boundary_head
                    .validate()
                    .context("invalid boundary_head configuration")?;
                self.boundary_head
                    .check_supported()
                    .context("unsupported boundary_head configuration")?;
            }
        }
        Ok(())
    }
}

/// Encoder (DeBERTa V2) config — loaded from `encoder_config/config.json`
/// in the GLiNER2 HF repo, or from the encoder model's own repo.
///
/// We re-use candle-transformers' deserialization for this.
pub use candle_transformers::models::debertav2::Config as EncoderConfig;

pub fn encoder_config_from_file(path: impl AsRef<Path>) -> Result<EncoderConfig> {
    let raw = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("reading encoder config {:?}", path.as_ref()))?;
    serde_json::from_str(&raw).context("parsing EncoderConfig")
}

/// Hidden size extracted from the encoder config.
pub fn hidden_size(enc: &EncoderConfig) -> usize {
    enc.hidden_size
}
