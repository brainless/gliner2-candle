/// GLiNER2 model dispatch: `Model` selects the architecture declared by
/// `config.json` **before** any head weights are looked up, then delegates to
/// the span or boundary model struct. The span forward path is unchanged.
///
/// ## Weight key layout in `model.safetensors`
///
/// Based on how Python's `Extractor(PreTrainedModel)` saves its state_dict:
///
///   encoder.embeddings.*            — DeBERTa embeddings
///   encoder.encoder.layer.N.*      — DeBERTa transformer layers
///   span_rep.out_project.{weight,bias}
///   classifier.0.{weight,bias}     — Linear(hidden, 2*hidden)
///   classifier.2.{weight,bias}     — Linear(2*hidden, 1)
///   count_pred.0.{weight,bias}     — Linear(hidden, 2*hidden)
///   count_pred.2.{weight,bias}     — Linear(2*hidden, 20)
///   count_embed.pos_embedding.weight
///   count_embed.gru.*
///   count_embed.projector.{0,2}.*
///
/// If the actual keys differ (download model and inspect with the
/// `--list-weights` flag in main), adjust the `vb.pp(...)` calls below.
/// Boundary-checkpoint key layout lives in `src/boundary.rs` / DEVELOP.md.
use anyhow::{anyhow, Context, Result};
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::debertav2::{DebertaV2Model, DTYPE as DEBERTA_DTYPE};
use std::path::Path;

use crate::{
    boundary::BoundaryModel,
    boundary_rel::ExtractedRelation,
    config::{encoder_config_from_file, hidden_size, Architecture, Gliner2Config},
    count_lstm::CountLSTM,
    inference::{extract_entities, ExtractedEntity},
    processor::{Preprocessor, ProcessedInput},
    span_rep::SpanRepLayer,
};

/// Architecture-dispatched GLiNER2 model. The CLI talks to this enum only;
/// span and boundary internals stay behind their variants.
/// (Held by value once at startup — boxing the variants would gain nothing.)
#[allow(clippy::large_enum_variant)]
pub enum Model {
    Span(SpanModel),
    Boundary(BoundaryModel),
}

/// Legacy `fastino/gliner2-*` span architecture (entity extraction).
pub struct SpanModel {
    encoder: DebertaV2Model,
    span_rep: SpanRepLayer,
    /// classifier MLP: hidden → 2*hidden → 1  (binary span scorer)
    #[allow(dead_code)] // loaded but not wired into scoring (see DEVELOP.md)
    clf1: Linear,
    #[allow(dead_code)]
    clf2: Linear,
    /// count_pred MLP: hidden → 2*hidden → 20
    cp1: Linear,
    cp2: Linear,
    count_embed: CountLSTM,
    preprocessor: Preprocessor,
    #[allow(dead_code)] // kept for introspection; max_width is consumed at load
    pub gliner_cfg: Gliner2Config,
    #[allow(dead_code)] // shapes are read back from tensors in forward()
    hidden_size: usize,
    device: Device,
}

/// One safetensors header entry (name + dtype + shape).
pub struct WeightEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl Model {
    /// Load from a directory that contains:
    ///   - `config.json`           (Gliner2Config)
    ///   - `encoder_config/config.json`  (DeBERTa config)
    ///   - `model.safetensors`     (all weights)
    ///   - `tokenizer.json`        (span preprocessing)
    ///
    /// The architecture is selected from `config.json` before any head
    /// weights are looked up, so a boundary checkpoint never runs span-head
    /// weight loads and vice versa.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let gliner_cfg = Gliner2Config::from_file(dir.join("config.json"))?;
        // Machine-readable dump for cross-checking against oracle/manifest.json.
        tracing::debug!(
            "parsed model config: {}",
            serde_json::to_string(&gliner_cfg).context("serializing parsed config")?
        );
        match gliner_cfg.architecture {
            Architecture::Span => Ok(Self::Span(SpanModel::load(dir, gliner_cfg, device)?)),
            Architecture::Boundary => Ok(Self::Boundary(BoundaryModel::load(
                dir, gliner_cfg, device,
            )?)),
        }
    }

    /// Detected architecture of the loaded checkpoint (printed by the CLI).
    pub fn architecture(&self) -> Architecture {
        match self {
            Self::Span(_) => Architecture::Span,
            Self::Boundary(_) => Architecture::Boundary,
        }
    }

    /// Fail-clearly message for relation requests on the span architecture
    /// (epic Task 8), shared by [`Self::predict_relations`] and
    /// [`Self::check_relation_support`].
    const SPAN_RELATIONS_UNSUPPORTED: &'static str =
        "relation extraction is not supported by the span architecture \
         (this checkpoint has no relation head / enable_relations=false); refusing to \
         return an empty result that would look like a valid negative prediction — \
         use a boundary checkpoint such as fastino/gliner2.5-small-v1";

    /// Pre-flight variant of [`Self::predict_relations`]: the same
    /// fail-clearly errors, so the CLI can reject an unsupported task before
    /// printing any results (epic Task 9).
    pub fn check_relation_support(&self) -> Result<()> {
        match self {
            Self::Span(_) => Err(anyhow!(Self::SPAN_RELATIONS_UNSUPPORTED)),
            Self::Boundary(m) => m.check_relation_support(),
        }
    }

    /// Shared prediction operation used by the CLI regardless of
    /// architecture.
    pub fn predict_entities(
        &self,
        text: &str,
        entity_types: &[String],
        threshold: f32,
    ) -> Result<Vec<ExtractedEntity>> {
        match self {
            Self::Span(m) => m.predict_text(text, entity_types, threshold),
            Self::Boundary(m) => m.predict_entities(text, entity_types, threshold),
        }
    }

    /// Shared relation prediction operation (epic Task 8): entity types are
    /// optional schema context (they change the encoding) — a relation-only
    /// request passes `entity_types = []`.
    ///
    /// The span architecture has no relation head: requesting relations there
    /// fails clearly rather than returning an empty result that would look
    /// like a valid negative prediction (as does a boundary checkpoint with
    /// `enable_relations=false` / no relation scorer). The CLI goes through
    /// [`Self::predict_extractions`]; this entry remains for direct relation
    /// requests and the epic Task 8 tests.
    #[allow(dead_code)]
    pub fn predict_relations(
        &self,
        text: &str,
        entity_types: &[String],
        relation_types: &[String],
        threshold: f32,
    ) -> Result<Vec<ExtractedRelation>> {
        match self {
            Self::Span(_) => Err(anyhow!(Self::SPAN_RELATIONS_UNSUPPORTED)),
            Self::Boundary(m) => m.predict_relations(text, entity_types, relation_types, threshold),
        }
    }

    /// Combined prediction operation (epic Task 9): the single production
    /// inference path behind the CLI's one-off, JSON, and case-file modes.
    /// Entity-only and relation-only requests pass the other list empty; a
    /// combined request decodes entities and relations from **one** forward
    /// over the full schema (matching the Python oracle's single `extract()`
    /// encoding). Span checkpoints reject non-empty relation types via
    /// [`Self::check_relation_support`]'s fail-clearly error.
    pub fn predict_extractions(
        &self,
        text: &str,
        entity_types: &[String],
        relation_types: &[String],
        threshold: f32,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>)> {
        match self {
            Self::Span(_) => {
                if !relation_types.is_empty() {
                    return Err(anyhow!(Self::SPAN_RELATIONS_UNSUPPORTED));
                }
                Ok((
                    self.predict_entities(text, entity_types, threshold)?,
                    Vec::new(),
                ))
            }
            Self::Boundary(m) => {
                m.predict_extractions(text, entity_types, relation_types, threshold)
            }
        }
    }

    /// List weight entries (name, dtype, shape) in the safetensors file —
    /// complete inventory for checkpoint inspection and prefix debugging.
    /// `limit` truncates to the first N entries; `None` lists everything.
    pub fn list_weights(
        model_dir: impl AsRef<Path>,
        limit: Option<usize>,
    ) -> Result<Vec<WeightEntry>> {
        use std::io::Read;
        let path = model_dir.as_ref().join("model.safetensors");
        // Read safetensors header (first 8 bytes = header length, then JSON)
        let mut f = std::fs::File::open(&path)?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header = vec![0u8; header_len];
        f.read_exact(&mut header)?;
        let header_str = std::str::from_utf8(&header)?;
        let v: serde_json::Value = serde_json::from_str(header_str)?;
        let mut entries: Vec<WeightEntry> = v
            .as_object()
            .map(|o| {
                o.iter()
                    .filter(|(k, _)| k.as_str() != "__metadata__")
                    .map(|(k, meta)| WeightEntry {
                        name: k.clone(),
                        dtype: meta["dtype"].as_str().unwrap_or("?").to_string(),
                        shape: meta["shape"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|x| x.as_u64())
                                    .map(|x| x as usize)
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(n) = limit {
            entries.truncate(n);
        }
        Ok(entries)
    }
}

impl SpanModel {
    fn load(model_dir: &Path, gliner_cfg: Gliner2Config, device: &Device) -> Result<Self> {
        let dir = model_dir;

        // ── Configs ──────────────────────────────────────────────────────────
        let enc_cfg = encoder_config_from_file(dir.join("encoder_config").join("config.json"))
            .context("encoder_config/config.json not found — see README")?;
        let h = hidden_size(&enc_cfg);

        // ── Weights ───────────────────────────────────────────────────────────
        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(anyhow!(
                "model.safetensors not found in {:?}. \
                 Download with: huggingface-cli download fastino/gliner2-large-v1 \
                 --local-dir {:?}",
                dir,
                dir
            ));
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DEBERTA_DTYPE, device)?
        };

        // ── Encoder (DeBERTa V2) ──────────────────────────────────────────────
        // GLiNER2 stores the encoder as `self.encoder = AutoModel.from_pretrained(...)`.
        // Its state_dict keys start with `encoder.` (the attribute name).
        // DeBERTa's own state_dict has keys `embeddings.*` and `encoder.*`,
        // so the full path is `encoder.embeddings.*` and `encoder.encoder.*`.
        let encoder = DebertaV2Model::load(vb.pp("encoder"), &enc_cfg)?;

        // ── Custom heads ─────────────────────────────────────────────────────
        // GLiNER2 wraps gliner's SpanRepLayer as self.span_rep.span_rep_layer
        let span_rep = SpanRepLayer::load(
            h,
            gliner_cfg.max_width,
            vb.pp("span_rep").pp("span_rep_layer"),
        )?;

        // classifier Sequential: [Linear(h, 2h), ReLU, Linear(2h, 1)]
        // PyTorch Sequential indices: 0 = first Linear, 2 = second Linear (1 = ReLU, no params)
        let clf1 = linear(h, h * 2, vb.pp("classifier").pp("0"))?;
        let clf2 = linear(h * 2, 1, vb.pp("classifier").pp("2"))?;

        // count_pred Sequential: [Linear(h, 2h), ReLU, Linear(2h, 20)]
        let cp1 = linear(h, h * 2, vb.pp("count_pred").pp("0"))?;
        let cp2 = linear(h * 2, 20, vb.pp("count_pred").pp("2"))?;

        let count_embed = CountLSTM::load(h, 20, vb.pp("count_embed"))?;

        let preprocessor = Preprocessor::from_file(dir.join("tokenizer.json"))?;

        Ok(Self {
            encoder,
            span_rep,
            clf1,
            clf2,
            cp1,
            cp2,
            count_embed,
            preprocessor,
            gliner_cfg,
            hidden_size: h,
            device: device.clone(),
        })
    }

    /// Preprocess + forward + extract entities (CLI entry for span models).
    fn predict_text(
        &self,
        text: &str,
        entity_types: &[String],
        threshold: f32,
    ) -> Result<Vec<ExtractedEntity>> {
        let input = self.preprocessor.process(text, entity_types)?;
        tracing::info!(
            "Input: {} tokens ({} words, {} entity types)",
            input.input_ids.len(),
            input.words.len(),
            input.entity_types.len()
        );
        self.predict(&input, threshold)
    }

    /// Run entity extraction on a preprocessed input.
    ///
    /// Returns raw scores of shape (gold_count, num_entity_types, text_len, max_width).
    pub fn forward(&self, input: &ProcessedInput) -> Result<Tensor> {
        let seq_len = input.input_ids.len();

        // ── Build encoder inputs ───────────────────────────────────────────────
        let ids = Tensor::from_vec(input.input_ids.clone(), (1, seq_len), &self.device)?;
        let mask = Tensor::from_vec(input.attention_mask.clone(), (1, seq_len), &self.device)?;

        // ── Encoder forward ──────────────────────────────────────────────────
        // Output: (1, seq_len, hidden)
        let token_embs = self.encoder.forward(&ids, None, Some(mask))?;
        let token_embs = token_embs.squeeze(0)?; // (seq_len, hidden)

        // ── Extract word-level embeddings (first-subtoken pooling) ────────────
        let text_len = input.word_token_indices.len();
        let word_indices_t = Tensor::from_vec(
            input
                .word_token_indices
                .iter()
                .map(|&i| i as u32)
                .collect::<Vec<_>>(),
            (text_len,),
            &self.device,
        )?;
        let word_embs = token_embs.index_select(&word_indices_t, 0)?; // (text_len, hidden)

        // ── Extract entity-type embeddings (at [E] token positions) ──────────
        let n_types = input.entity_e_positions.len();
        let e_indices_t = Tensor::from_vec(
            input
                .entity_e_positions
                .iter()
                .map(|&i| i as u32)
                .collect::<Vec<_>>(),
            (n_types,),
            &self.device,
        )?;
        let entity_embs = token_embs.index_select(&e_indices_t, 0)?; // (n_types, hidden)

        // ── Count prediction ──────────────────────────────────────────────────
        // Use the [P] token embedding for count prediction
        let p_emb = token_embs.i(input.p_token_pos)?; // (hidden,)
        let count_logits = self
            .cp2
            .forward(&self.cp1.forward(&p_emb.unsqueeze(0)?)?.relu()?)?; // (1, 20)
        let gold_count = count_logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()? as usize;
        let gold_count = gold_count.max(1); // at least 1 extraction step

        // ── CountLSTM ─────────────────────────────────────────────────────────
        // struct_proj: (gold_count, n_types, hidden)
        let struct_proj = self.count_embed.forward(&entity_embs, gold_count)?;

        // ── Span representations ──────────────────────────────────────────────
        // span_reps: (text_len, max_width, hidden)
        let span_reps = self.span_rep.forward(&word_embs)?;

        // ── Scoring (replaces einsum 'lkd,bpd->bplk') ────────────────────────
        // span_reps: (L, K, D) → (L*K, D)
        let (l, k, d) = span_reps.dims3()?;
        let (b, p, _) = struct_proj.dims3()?; // b=gold_count, p=n_types
        let span_flat = span_reps.reshape((l * k, d))?;

        // struct_proj: (B, P, D) → (B*P, D) → (D, B*P)
        let proj_flat = struct_proj.reshape((b * p, d))?.t()?;

        // Raw dot product: (L*K, D) @ (D, B*P) → (L*K, B*P)
        // Matches Python: torch.einsum('lkd,bpd->bplk', span_rep, struct_proj)
        let scores_flat = span_flat.matmul(&proj_flat)?;

        // (L*K, B*P) → (L, K, B, P) → (B, P, L, K)
        let scores = scores_flat.reshape((l, k, b, p))?.permute((2, 3, 0, 1))?;

        Ok(scores)
    }

    /// High-level convenience: preprocess + forward + extract entities.
    pub fn predict(&self, input: &ProcessedInput, threshold: f32) -> Result<Vec<ExtractedEntity>> {
        let scores = self.forward(input)?;
        extract_entities(&scores, input, threshold)
    }
}
