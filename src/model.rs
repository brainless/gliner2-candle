/// GLiNER2 model: DeBERTa V2 encoder + SpanRepLayer + CountLSTM + MLP heads.
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
use anyhow::{anyhow, Context, Result};
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::debertav2::{DebertaV2Model, DTYPE as DEBERTA_DTYPE};

use crate::{
    config::{encoder_config_from_file, hidden_size, Gliner2Config},
    count_lstm::CountLSTM,
    inference::{extract_entities, ExtractedEntity},
    processor::ProcessedInput,
    span_rep::SpanRepLayer,
};

pub struct GLiNER2 {
    encoder: DebertaV2Model,
    span_rep: SpanRepLayer,
    /// classifier MLP: hidden → 2*hidden → 1  (binary span scorer)
    clf1: Linear,
    clf2: Linear,
    /// count_pred MLP: hidden → 2*hidden → 20
    cp1: Linear,
    cp2: Linear,
    count_embed: CountLSTM,
    pub gliner_cfg: Gliner2Config,
    hidden_size: usize,
    device: Device,
}

impl GLiNER2 {
    /// Load from a directory that contains:
    ///   - `config.json`           (GLiNER2Config)
    ///   - `encoder_config/config.json`  (DeBERTa config)
    ///   - `model.safetensors`     (all weights)
    pub fn load(model_dir: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let dir = model_dir.as_ref();

        // ── Configs ──────────────────────────────────────────────────────────
        let gliner_cfg = Gliner2Config::from_file(dir.join("config.json"))?;
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

        Ok(Self {
            encoder,
            span_rep,
            clf1,
            clf2,
            cp1,
            cp2,
            count_embed,
            gliner_cfg,
            hidden_size: h,
            device: device.clone(),
        })
    }

    /// List the first N weight keys in the safetensors file — useful for
    /// debugging prefix mismatches.
    pub fn list_weight_keys(
        model_dir: impl AsRef<std::path::Path>,
        n: usize,
    ) -> Result<Vec<String>> {
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
        let mut keys: Vec<String> = v
            .as_object()
            .map(|o| o.keys().filter(|k| *k != "__metadata__").cloned().collect())
            .unwrap_or_default();
        keys.sort();
        keys.truncate(n);
        Ok(keys)
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

        // L2 normalize vectors before dot product for stable scores
        let span_norms = span_flat.sqr()?.sum_keepdim(1)?.sqrt()?;
        let span_flat = span_flat.broadcast_div(&span_norms)?;

        let proj_norms = proj_flat.sqr()?.sum_keepdim(0)?.sqrt()?;
        let proj_flat = proj_flat.broadcast_div(&proj_norms)?;

        // (L*K, D) @ (D, B*P) → (L*K, B*P)
        let scores_flat = span_flat.matmul(&proj_flat)?;

        // Scale by sqrt(hidden_dim) to match typical attention magnitude
        let scale = (d as f32).sqrt();
        let scores_flat = scores_flat.broadcast_mul(&Tensor::new(scale, scores_flat.device())?)?;

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
