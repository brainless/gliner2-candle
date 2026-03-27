use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// GLiNER2-specific model config stored in `config.json` on the HF repo.
/// Mirrors Python's `ExtractorConfig`.
#[derive(Debug, Deserialize)]
pub struct Gliner2Config {
    /// HF model ID of the encoder backbone, e.g. "microsoft/deberta-v2-large"
    #[serde(default = "default_model_name")]
    pub model_name: String,

    /// Maximum span width in words (default 8)
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

impl Gliner2Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {:?}", path.as_ref()))?;
        serde_json::from_str(&raw).context("parsing Gliner2Config")
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
