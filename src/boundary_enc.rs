//! Boundary-state encoding (`encoding.py`) and marginal heads (`heads.py`) —
//! epic `gliner-2.5-boundary-support`, Task 5.
//!
//! Rust port, at pinned GLiNER2 commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, of:
//! * `gliner2/models/boundary/encoding.py` — [`BoundaryEncoder`]: left
//!   token/BOS + right token/EOS states, per-side projection, concatenation,
//!   output projection + LayerNorm, then (configured) pre-norm local
//!   self-attention blocks [`BoundaryAttentionBlock`] and residual SwiGLU
//!   refinement blocks [`ResidualSwiGLU`]. Dropout is off at inference, so the
//!   Python `nn.Dropout` calls are not ported. Padding boundaries are masked
//!   (zeroed) exactly where Python zeroes them; the final valid boundary of
//!   each sample is its EOS boundary, including after truncation.
//! * `gliner2/models/boundary/heads.py` — [`BoundaryQueryHead`]: start/end
//!   logits `[B,Q,L+1]` and inside logits `[B,Q,L]` via scaled dot products
//!   (`1/sqrt(boundary_dim)`); invalid positions carry the finite
//!   [`MASK_LOGIT`] sentinel. The inside prefix is an explicitly-fp32
//!   cumulative sum of the per-query mean-centered inside logits (zeros at
//!   invalid tokens, leading zero); interval scoring must restore
//!   `mean * (end - start)` — see [`BoundaryMarginals::inside_interval_sum`].
//!
//! Tensors keep the Python batch dimension (`B = 1` at inference today); see
//! DEVELOP.md "Boundary encoder & marginals" for the oracle comparison and
//! observed tolerances.
use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use crate::config::BoundaryHeadConfig;

/// Finite masking sentinel (`constants.py MASK_LOGIT`): representable in
/// fp16/bf16/fp32 and safe to add before masking (dtype minima can overflow).
pub const MASK_LOGIT: f32 = -1.0e4;

/// `encoding.py BoundaryEncoding`.
pub struct BoundaryEncoding {
    /// `[B, L+1, d]` — padding boundaries zeroed.
    pub states: Tensor,
    /// `[B, L+1]` u8 — boundary `i` valid iff `i <= n_b`.
    pub mask: Tensor,
}

/// `heads.py BoundaryMarginals`.
#[allow(dead_code)] // fields are consumed by epic Task 6 (and the tests below)
pub struct BoundaryMarginals {
    /// `[B, Q, L+1]` — masked positions hold [`MASK_LOGIT`].
    pub start_logits: Tensor,
    /// `[B, Q, L+1]` — masked positions hold [`MASK_LOGIT`].
    pub end_logits: Tensor,
    /// `[B, Q, L]` — masked positions hold [`MASK_LOGIT`].
    pub inside_logits: Tensor,
    /// `[B, Q, L+1]` — fp32 cumulative sum of the mean-centered inside
    /// logits (prefix `k` = sum over tokens `[0, k)`); `prefix[0] == 0`.
    pub inside_prefix: Tensor,
    /// `[B, Q, 1]` — per-query mean of the valid inside logits; interval
    /// scoring restores `mean * (end - start)`.
    pub inside_prefix_mean: Tensor,
}

impl BoundaryMarginals {
    /// Interval reconstruction (Python `scoring.interval_prefix_score`):
    /// `prefix[end] - prefix[start] + mean * (end - start)` for the half-open
    /// word interval `[start, end)`.
    ///
    /// The mean restore is required: omitting it changes every score by
    /// `mean * (end - start)`. Indices are clamped to `[0, L]` like Python.
    /// The identity `= sum(inside_logits[start..end])` holds exactly (up to
    /// fp32 cumsum rounding) for intervals inside a sample's valid text.
    #[allow(dead_code)] // consumed by epic Task 6's pair scorer (tests use it)
    pub fn inside_interval_sum(
        &self,
        batch: usize,
        query: usize,
        start: usize,
        end: usize,
    ) -> Result<f32> {
        let l1 = self.inside_prefix.dim(D::Minus1)?;
        let max_idx = l1 - 1;
        let s = start.min(max_idx);
        let e = end.min(max_idx);
        let prefix = self.inside_prefix.get(batch)?.get(query)?; // [L+1]
        let p_start = prefix.get(s)?.to_vec0::<f32>()?;
        let p_end = prefix.get(e)?.to_vec0::<f32>()?;
        let mean = self.inside_prefix_mean.get(batch)?.get(query)?; // [1]
        let mean = mean.to_vec1::<f32>()?[0];
        Ok(p_end - p_start + mean * (e - s) as f32)
    }
}

/// `logits.masked_fill(~keep, value)` — `keep` is a u8/bool mask broadcastable
/// to `logits`; positions where `keep` is 0 become `value`.
fn masked_fill(logits: &Tensor, keep: &Tensor, value: f32) -> Result<Tensor> {
    let keep = keep.to_dtype(DType::U8)?.broadcast_as(logits.shape())?;
    let fill = Tensor::full(value, logits.shape(), logits.device())?;
    Ok(keep.where_cond(logits, &fill)?)
}

/// `encoding.py BoundaryAttentionBlock`: pre-norm multi-head self-attention
/// over valid boundary positions with an optional local window, residual, and
/// per-block zeroing of padding boundaries.
struct BoundaryAttentionBlock {
    norm: LayerNorm,
    qkv_projection: Linear,
    output_projection: Linear,
    num_heads: usize,
    head_dim: usize,
    /// `0` = full attention; else `|i - j| <= window`.
    window: usize,
}

impl BoundaryAttentionBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, window: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(dim, 1e-5, vb.pp("norm"))?,
            qkv_projection: linear(dim, 3 * dim, vb.pp("qkv_projection"))?,
            output_projection: linear(dim, dim, vb.pp("output_projection"))?,
            num_heads,
            head_dim: dim / num_heads,
            window,
        })
    }

    /// `allowed[b,1,i,j] = mask[j] && |i-j| <= window, or i == j` (the
    /// diagonal is always allowed so padding query rows keep one legal key —
    /// Python `allowed | diagonal`).
    fn allowed_mask(&self, mask: &Tensor, window: usize) -> Result<Tensor> {
        let (b, n) = mask.dims2()?;
        let dev = mask.device();
        let keys = mask
            .reshape((b, 1, 1, n))?
            .broadcast_as((b, 1, n, n))?
            .contiguous()?;
        let mut allowed = keys;
        if window > 0 {
            let pos = Tensor::arange(0i64, n as i64, dev)?;
            let diff = pos.unsqueeze(1)?.broadcast_sub(&pos.unsqueeze(0)?)?.abs()?;
            let local = diff
                .le(window as i64)?
                .reshape((1, 1, n, n))?
                .broadcast_as((b, 1, n, n))?
                .contiguous()?;
            allowed = (&allowed * &local)?;
        }
        let diagonal = Tensor::eye(n, DType::U8, dev)?
            .reshape((1, 1, n, n))?
            .broadcast_as((b, 1, n, n))?
            .contiguous()?;
        Ok((&allowed + &diagonal)?.ne(0u8)?)
    }

    fn forward(&self, states: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.forward_window(states, mask, self.window)
    }

    /// [`Self::forward`] with an explicit window override (diagnostics only).
    fn forward_window(&self, states: &Tensor, mask: &Tensor, window: usize) -> Result<Tensor> {
        let (b, n, d) = states.dims3()?;
        let x = self.norm.forward(states)?; // [B,N,d]
        let qkv = self.qkv_projection.forward(&x)?; // [B,N,3d]
        let qkv = qkv.reshape((b, n, 3, self.num_heads, self.head_dim))?;
        let heads = |idx: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, idx, 1)?
                .squeeze(2)?
                .transpose(1, 2)? // [B,heads,N,head_dim]
                .contiguous()?)
        };
        let q = heads(0)?;
        let k = heads(1)?;
        let v = heads(2)?;

        // Scaled dot-product attention (candle-nn `sdpa` has no CPU path).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * scale)?;
        let scores = masked_fill(
            &scores,
            &self.allowed_mask(mask, window)?,
            f32::NEG_INFINITY,
        )?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = attn.matmul(&v)?; // [B,heads,N,head_dim]
        let attended = attended.transpose(1, 2)?.reshape((b, n, d))?;
        let update = self.output_projection.forward(&attended)?;
        let out = (states + update)?;
        // Zero padding boundaries (Python: `(states + update) * mask`).
        let m = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        Ok(out.broadcast_mul(&m)?)
    }
}

/// `encoding.py ResidualSwiGLU`: pre-norm residual SwiGLU feed-forward block
/// (value * silu(gate), no masking).
struct ResidualSwiGLU {
    norm: LayerNorm,
    input_projection: Linear,
    output_projection: Linear,
    ffn_hidden: usize,
}

impl ResidualSwiGLU {
    fn load(vb: VarBuilder, dim: usize, ffn_hidden: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(dim, 1e-5, vb.pp("norm"))?,
            input_projection: linear(dim, 2 * ffn_hidden, vb.pp("input_projection"))?,
            output_projection: linear(ffn_hidden, dim, vb.pp("output_projection"))?,
            ffn_hidden,
        })
    }

    fn forward(&self, states: &Tensor) -> Result<Tensor> {
        let x = self.input_projection.forward(&self.norm.forward(states)?)?;
        let value = x.narrow(D::Minus1, 0, self.ffn_hidden)?;
        let gate = x.narrow(D::Minus1, self.ffn_hidden, self.ffn_hidden)?;
        let update = (value * candle_nn::ops::silu(&gate)?)?;
        let update = self.output_projection.forward(&update)?;
        Ok((states + update)?)
    }
}

/// `encoding.py BoundaryEncoder`.
pub struct BoundaryEncoder {
    left_projection: Linear,
    right_projection: Linear,
    output_projection: Linear,
    layer_norm: LayerNorm,
    /// `[H]` — learned left state of boundary 0.
    bos_state: Tensor,
    /// `[H]` — learned right state of each sample's final boundary.
    eos_state: Tensor,
    attention_blocks: Vec<BoundaryAttentionBlock>,
    refinement_blocks: Vec<ResidualSwiGLU>,
    #[allow(dead_code)]
    boundary_dim: usize,
    #[allow(dead_code)]
    hidden_size: usize,
}

impl BoundaryEncoder {
    /// Load `boundary_head.boundary_encoder.*`. Layer counts / window come
    /// from the migrated config (`boundary_attention_layers`,
    /// `boundary_attention_heads`, `boundary_attention_window`,
    /// `boundary_refinement_layers`); all are active for
    /// `fastino/gliner2.5-small-v1` (2 + 1 blocks, window 128).
    pub fn load(vb: VarBuilder, cfg: &BoundaryHeadConfig, hidden_size: usize) -> Result<Self> {
        let d = cfg.boundary_dim;
        let ffn = ((d as f64 * cfg.boundary_ffn_multiplier) as usize).max(1);
        let mut attention_blocks = Vec::with_capacity(cfg.boundary_attention_layers);
        for i in 0..cfg.boundary_attention_layers {
            attention_blocks.push(BoundaryAttentionBlock::load(
                vb.pp(format!("attention_blocks.{i}")),
                d,
                cfg.boundary_attention_heads,
                cfg.boundary_attention_window,
            )?);
        }
        let mut refinement_blocks = Vec::with_capacity(cfg.boundary_refinement_layers);
        for i in 0..cfg.boundary_refinement_layers {
            refinement_blocks.push(ResidualSwiGLU::load(
                vb.pp(format!("refinement_blocks.{i}")),
                d,
                ffn,
            )?);
        }
        Ok(Self {
            left_projection: linear(hidden_size, d, vb.pp("left_projection"))?,
            right_projection: linear(hidden_size, d, vb.pp("right_projection"))?,
            output_projection: linear(2 * d, d, vb.pp("output_projection"))?,
            layer_norm: layer_norm(d, 1e-5, vb.pp("layer_norm"))?,
            bos_state: vb.get(hidden_size, "bos_state")?,
            eos_state: vb.get(hidden_size, "eos_state")?,
            attention_blocks,
            refinement_blocks,
            boundary_dim: d,
            hidden_size,
        })
    }

    /// Project and refine left/right token states into per-boundary states.
    ///
    /// `text_states`: `[B, L, H]`; `text_mask`: `[B, L]` u8 (contiguous valid
    /// prefix per sample). Boundary `i` sits between token `i-1` (left) and
    /// token `i` (right); boundary 0 uses BOS as left, each sample's final
    /// boundary `n_b` uses EOS as right (also at `L` — masked out).
    pub fn forward(&self, text_states: &Tensor, text_mask: &Tensor) -> Result<BoundaryEncoding> {
        Ok(self.forward_stages(text_states, text_mask)?.encoding)
    }

    /// [`Self::forward`] keeping the per-stage intermediates: after
    /// `layer_norm`, after each attention block, after each refinement block,
    /// and the final encoding (padding boundaries zeroed). Used by the epic
    /// Task 5 oracle comparisons.
    pub(crate) fn forward_stages(
        &self,
        text_states: &Tensor,
        text_mask: &Tensor,
    ) -> Result<EncoderStages> {
        let (b, l, h) = text_states.dims3()?;
        let dev = text_states.device();
        let text_lengths = text_mask
            .to_dtype(DType::F32)?
            .sum_keepdim(1)?
            .to_dtype(DType::U32)?;
        let bos = self
            .bos_state
            .to_dtype(text_states.dtype())?
            .reshape((1, 1, h))?
            .broadcast_as((b, 1, h))?
            .contiguous()?;
        let left = Tensor::cat(&[&bos, text_states], 1)?;
        let eos = self
            .eos_state
            .to_dtype(text_states.dtype())?
            .reshape((1, 1, h))?
            .broadcast_as((b, 1, h))?
            .contiguous()?;
        let mut right = Tensor::cat(&[text_states, &eos], 1)?;
        let n_b = text_lengths.reshape((b, 1))?.broadcast_as((b, l + 1))?;
        let idx = Tensor::arange(0u32, (l + 1) as u32, dev)?
            .reshape((1, l + 1))?
            .broadcast_as((b, l + 1))?;
        let at_end = idx.eq(&n_b)?.unsqueeze(2)?.broadcast_as((b, l + 1, h))?;
        right = at_end.where_cond(&eos.expand((b, l + 1, h))?, &right)?;

        let left_p = self.left_projection.forward(&left)?;
        let right_p = self.right_projection.forward(&right)?;
        let mut states = self
            .output_projection
            .forward(&Tensor::cat(&[&left_p, &right_p], 2)?)?;
        states = self.layer_norm.forward(&states)?;
        let after_layer_norm = states.clone();

        let mask_idx = Tensor::arange(0u32, (l + 1) as u32, dev)?
            .reshape((1, l + 1))?
            .broadcast_as((b, l + 1))?;
        let mask = mask_idx.le(&n_b)?;

        let mut after_attention = Vec::with_capacity(self.attention_blocks.len());
        for block in &self.attention_blocks {
            states = block.forward(&states, &mask)?;
            after_attention.push(states.clone());
        }
        let mut after_refinement = Vec::with_capacity(self.refinement_blocks.len());
        for block in &self.refinement_blocks {
            states = block.forward(&states)?;
            after_refinement.push(states.clone());
        }
        let m = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        let states = states.broadcast_mul(&m)?;
        Ok(EncoderStages {
            after_layer_norm,
            after_attention,
            after_refinement,
            encoding: BoundaryEncoding { states, mask },
        })
    }

    /// Test-only: run attention block `idx` with an explicit window override
    /// (oracle compares `window=2` against the production `window=128`).
    #[cfg(test)]
    pub(crate) fn attention_block_forward(
        &self,
        idx: usize,
        states: &Tensor,
        mask: &Tensor,
        window: usize,
    ) -> Result<Tensor> {
        self.attention_blocks[idx].forward_window(states, mask, window)
    }
}

/// Per-stage intermediates of [`BoundaryEncoder::forward`] (returned by
/// [`BoundaryEncoder::forward_stages`]).
#[allow(dead_code)] // stage fields are read by the epic Task 5 tests
pub(crate) struct EncoderStages {
    pub(crate) after_layer_norm: Tensor,
    pub(crate) after_attention: Vec<Tensor>,
    pub(crate) after_refinement: Vec<Tensor>,
    pub(crate) encoding: BoundaryEncoding,
}

/// `heads.py BoundaryQueryHead` — start/end/inside marginals per query.
pub struct BoundaryQueryHead {
    start_boundary_projection: Linear,
    start_query_projection: Linear,
    end_boundary_projection: Linear,
    end_query_projection: Linear,
    inside_text_projection: Linear,
    inside_query_projection: Linear,
    boundary_dim: usize,
}

impl BoundaryQueryHead {
    /// Load `boundary_head.boundary_query_head.*`. The query dim equals the
    /// encoder hidden size for this family (query states are marker hidden
    /// states); `query_conditioned_inside_weight` only feeds the inactive
    /// per_query pair scorer and is not part of this head.
    pub fn load(vb: VarBuilder, cfg: &BoundaryHeadConfig, hidden_size: usize) -> Result<Self> {
        let d = cfg.boundary_dim;
        Ok(Self {
            start_boundary_projection: linear(d, d, vb.pp("start_boundary_projection"))?,
            start_query_projection: linear(hidden_size, d, vb.pp("start_query_projection"))?,
            end_boundary_projection: linear(d, d, vb.pp("end_boundary_projection"))?,
            end_query_projection: linear(hidden_size, d, vb.pp("end_query_projection"))?,
            inside_text_projection: linear(hidden_size, d, vb.pp("inside_text_projection"))?,
            inside_query_projection: linear(hidden_size, d, vb.pp("inside_query_projection"))?,
            boundary_dim: d,
        })
    }

    /// `heads.py BoundaryQueryHead.forward`.
    ///
    /// `boundary_states` `[B,L+1,d]`, `boundary_mask` `[B,L+1]`, `text_states`
    /// `[B,L,H]`, `text_mask` `[B,L]`, `query_states` `[B,Q,H]`, `query_mask`
    /// `[B,Q]` (all u8, 1 = valid).
    pub fn forward(
        &self,
        boundary_states: &Tensor,
        boundary_mask: &Tensor,
        text_states: &Tensor,
        text_mask: &Tensor,
        query_states: &Tensor,
        query_mask: &Tensor,
    ) -> Result<BoundaryMarginals> {
        let scale = 1.0 / (self.boundary_dim as f64).sqrt();
        let (b, q, _) = query_states.dims3()?;

        // start/end logits: einsum("bld,bqd->bql") over boundary states.
        let start_b = self.start_boundary_projection.forward(boundary_states)?;
        let start_q = self.start_query_projection.forward(query_states)?;
        let start_logits = start_q.matmul(&start_b.transpose(1, 2)?.contiguous()?)?;
        let start_logits = (start_logits * scale)?;

        let end_b = self.end_boundary_projection.forward(boundary_states)?;
        let end_q = self.end_query_projection.forward(query_states)?;
        let end_logits = end_q.matmul(&end_b.transpose(1, 2)?.contiguous()?)?;
        let end_logits = (end_logits * scale)?;

        // inside logits: einsum over token states -> [B,Q,L].
        let inside_t = self.inside_text_projection.forward(text_states)?;
        let inside_q = self.inside_query_projection.forward(query_states)?;
        let inside_logits = inside_q.matmul(&inside_t.transpose(1, 2)?.contiguous()?)?;
        let inside_logits = (inside_logits * scale)?;

        // Mask invalid boundaries/tokens and invalid queries.
        let qm = query_mask.to_dtype(DType::U8)?.unsqueeze(2)?; // [B,Q,1]
        let b_keep = qm.broadcast_mul(&boundary_mask.to_dtype(DType::U8)?.unsqueeze(1)?)?;
        let t_keep = qm.broadcast_mul(&text_mask.to_dtype(DType::U8)?.unsqueeze(1)?)?;
        let start_logits = masked_fill(&start_logits, &b_keep, MASK_LOGIT)?;
        let end_logits = masked_fill(&end_logits, &b_keep, MASK_LOGIT)?;
        let inside_logits = masked_fill(&inside_logits, &t_keep, MASK_LOGIT)?;

        // Inside prefix: fp32 cumulative sum of the per-query mean-centered
        // inside logits; masked tokens contribute 0 so the prefix difference
        // over [i, j) equals the sum of real inside scores.
        let inside_for_prefix = masked_fill(&inside_logits, &t_keep, 0.0)?;
        let inside_for_prefix = inside_for_prefix.to_dtype(DType::F32)?;
        let t_keep_f32 = t_keep.to_dtype(DType::F32)?;
        let valid_count = t_keep_f32.sum_keepdim(2)?.clamp(1.0f64, f64::INFINITY)?; // [B,Q,1]
        let inside_mean = (inside_for_prefix.sum_keepdim(2)? / &valid_count)?; // [B,Q,1]
        let centered = inside_for_prefix.broadcast_sub(&inside_mean)?;
        let centered = centered.broadcast_mul(&t_keep_f32)?;
        let zeros = Tensor::zeros((b, q, 1), DType::F32, centered.device())?;
        let inside_prefix = Tensor::cat(&[&zeros, &centered.cumsum(2)?], 2)?;

        Ok(BoundaryMarginals {
            start_logits,
            end_logits,
            inside_logits,
            inside_prefix,
            inside_prefix_mean: inside_mean,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Epic Task 5 oracle checks: the synthetic padded batch
    //! (`oracle/synthetic_masks.json`) exercises what the B=1 corpus cannot —
    //! EOS at each sample's own final boundary, validity masks, `MASK_LOGIT`
    //! fills, padding zeroing, the fp32 centered prefix, and the local
    //! attention window. Numerical tolerance: fp32 CPU vs Python fp32, assert
    //! `< 1e-4` abs; observed max diffs are printed and are typically ~1e-6
    //! (see DEVELOP.md "Boundary encoder & marginals").
    use super::*;
    use crate::config::{encoder_config_from_file, hidden_size, Gliner2Config};
    use candle_core::Device;
    use candle_transformers::models::debertav2::DTYPE as DEBERTA_DTYPE;
    use std::path::PathBuf;

    const MODEL_DIR: &str = "models/gliner2.5-small-v1";
    const TOL: f32 = 1e-4;

    fn repo_path(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    #[derive(Debug, serde::Deserialize)]
    struct SyntheticSamples {
        /// Rows `[batch, index, hidden, value]`.
        text_states: Vec<Vec<f64>>,
        /// Rows `[batch, query, hidden, value]`.
        query_states: Vec<Vec<f64>>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct SyntheticOutputs {
        after_layer_norm: Vec<Vec<Vec<f32>>>,
        after_attention_0: Vec<Vec<Vec<f32>>>,
        after_attention_0_window_2: Vec<Vec<Vec<f32>>>,
        after_refinement: Vec<Vec<Vec<f32>>>,
        boundary_states: Vec<Vec<Vec<f32>>>,
        boundary_mask: Vec<Vec<u8>>,
        start_logits: Vec<Vec<Vec<f32>>>,
        end_logits: Vec<Vec<Vec<f32>>>,
        inside_logits: Vec<Vec<Vec<f32>>>,
        inside_prefix: Vec<Vec<Vec<f32>>>,
        inside_prefix_mean: Vec<Vec<f32>>,
        /// Rows `[batch, query, start, end, value]`.
        interval_scores: Vec<Vec<f64>>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct SyntheticFixture {
        input_samples: SyntheticSamples,
        outputs: SyntheticOutputs,
    }

    fn load_fixture() -> SyntheticFixture {
        let path = repo_path("oracle/synthetic_masks.json");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("reading {path:?}: {e} (regenerate with oracle/capture_oracle.py)")
        });
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {path:?}: {e}"))
    }

    fn load_modules() -> (BoundaryEncoder, BoundaryQueryHead) {
        let dir = repo_path(MODEL_DIR);
        let weights = dir.join("model.safetensors");
        assert!(
            weights.exists(),
            "epic Task 5 checks require the boundary checkpoint: \
             hf download fastino/gliner2.5-small-v1 --local-dir ./{MODEL_DIR}"
        );
        let gliner_cfg =
            Gliner2Config::from_file(dir.join("config.json")).expect("parse config.json");
        let enc_cfg = encoder_config_from_file(dir.join("encoder_config").join("config.json"))
            .expect("parse encoder_config/config.json");
        let h = hidden_size(&enc_cfg);
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights], DEBERTA_DTYPE, &device)
                .expect("mmap model.safetensors")
        };
        let head_vb = vb.pp("boundary_head");
        let encoder =
            BoundaryEncoder::load(head_vb.pp("boundary_encoder"), &gliner_cfg.boundary_head, h)
                .expect("load boundary_encoder");
        let head = BoundaryQueryHead::load(
            head_vb.pp("boundary_query_head"),
            &gliner_cfg.boundary_head,
            h,
        )
        .expect("load boundary_query_head");
        (encoder, head)
    }

    /// `oracle/synthetic_masks.json input_formula` — exact fp32 lattice
    /// (multiples of 1/8) rebuilt bit-identically on both sides.
    fn synthetic_text_states(device: &Device) -> Tensor {
        let (bsz, length, hidden) = (2, 3, 384);
        let mut v = Vec::with_capacity(bsz * length * hidden);
        for b in 0..bsz {
            for i in 0..length {
                for h in 0..hidden {
                    v.push((((b * 131 + i * 17 + h * 7) % 32) as i32 - 16) as f32 / 8.0);
                }
            }
        }
        Tensor::from_vec(v, (bsz, length, hidden), device).unwrap()
    }

    fn synthetic_query_states(device: &Device) -> Tensor {
        let (bsz, nq, hidden) = (2, 2, 384);
        let mut v = Vec::with_capacity(bsz * nq * hidden);
        for b in 0..bsz {
            for q in 0..nq {
                for h in 0..hidden {
                    v.push((((b * 17 + q * 13 + h * 11) % 32) as i32 - 16) as f32 / 8.0);
                }
            }
        }
        Tensor::from_vec(v, (bsz, nq, hidden), device).unwrap()
    }

    fn max_diff3(got: &[Vec<Vec<f32>>], want: &[Vec<Vec<f32>>]) -> f32 {
        assert_eq!(got.len(), want.len(), "dim0");
        let mut m = 0f32;
        for (a, b) in got.iter().zip(want) {
            assert_eq!(a.len(), b.len(), "dim1");
            for (x, y) in a.iter().zip(b) {
                assert_eq!(x.len(), y.len(), "dim2");
                for (u, v) in x.iter().zip(y) {
                    m = m.max((u - v).abs());
                }
            }
        }
        m
    }

    fn max_diff2(got: &[Vec<f32>], want: &[Vec<f32>]) -> f32 {
        assert_eq!(got.len(), want.len(), "dim0");
        let mut m = 0f32;
        for (a, b) in got.iter().zip(want) {
            assert_eq!(a.len(), b.len(), "dim1");
            for (u, v) in a.iter().zip(b) {
                m = m.max((u - v).abs());
            }
        }
        m
    }

    fn assert_close3(name: &str, got: &Tensor, want: &[Vec<Vec<f32>>]) -> f32 {
        let m = max_diff3(&got.to_vec3::<f32>().unwrap(), want);
        eprintln!("[synthetic] {name}: max abs diff {:.3e}", m);
        assert!(m < TOL, "{name}: max abs diff {m:.3e} >= {TOL}");
        m
    }

    #[test]
    fn synthetic_padded_batch_matches_python() {
        let fixture = load_fixture();
        let (encoder, head) = load_modules();
        let device = Device::Cpu;

        let text_states = synthetic_text_states(&device);
        let query_states = synthetic_query_states(&device);
        // Verify the lattice formula agreement before comparing outputs.
        for (tensor, rows, name) in [
            (
                &text_states,
                &fixture.input_samples.text_states,
                "text_states",
            ),
            (
                &query_states,
                &fixture.input_samples.query_states,
                "query_states",
            ),
        ] {
            for row in rows {
                let (i0, i1, i2) = (row[0] as usize, row[1] as usize, row[2] as usize);
                let got = tensor.get(i0).unwrap().get(i1).unwrap().get(i2).unwrap();
                let got = got.to_vec0::<f32>().unwrap();
                assert_eq!(got, row[3] as f32, "{name} sample {row:?}");
            }
        }

        let text_mask = Tensor::from_vec(vec![1u8, 1, 1, 1, 1, 0], (2, 3), &device).unwrap();
        let query_mask = Tensor::from_vec(vec![1u8, 1, 1, 0], (2, 2), &device).unwrap();

        let stages = encoder.forward_stages(&text_states, &text_mask).unwrap();
        let marginals = head
            .forward(
                &stages.encoding.states,
                &stages.encoding.mask,
                &text_states,
                &text_mask,
                &query_states,
                &query_mask,
            )
            .unwrap();

        let out = &fixture.outputs;
        assert_close3(
            "after_layer_norm",
            &stages.after_layer_norm,
            &out.after_layer_norm,
        );
        assert_close3(
            "after_attention_0",
            &stages.after_attention[0],
            &out.after_attention_0,
        );
        assert_close3(
            "after_refinement",
            &stages.after_refinement[0],
            &out.after_refinement,
        );
        assert_close3(
            "boundary_states",
            &stages.encoding.states,
            &out.boundary_states,
        );
        assert_eq!(
            stages.encoding.mask.to_vec2::<u8>().unwrap(),
            out.boundary_mask,
            "boundary_mask values"
        );
        assert_close3("start_logits", &marginals.start_logits, &out.start_logits);
        assert_close3("end_logits", &marginals.end_logits, &out.end_logits);
        assert_close3(
            "inside_logits",
            &marginals.inside_logits,
            &out.inside_logits,
        );
        assert_close3(
            "inside_prefix",
            &marginals.inside_prefix,
            &out.inside_prefix,
        );
        let mean = marginals
            .inside_prefix_mean
            .to_vec3::<f32>()
            .unwrap()
            .iter()
            .map(|rows| rows.iter().map(|r| r[0]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let m = max_diff2(&mean, &out.inside_prefix_mean);
        eprintln!("[synthetic] inside_prefix_mean: max abs diff {:.3e}", m);
        assert!(m < TOL, "inside_prefix_mean: max abs diff {m:.3e} >= {TOL}");

        // Interval reconstruction vs Python's `interval_prefix_score`, and the
        // identity with the raw inside-logit sum on [start, end).
        let inside = marginals.inside_logits.to_vec3::<f32>().unwrap();
        let mut m = 0f32;
        for row in &out.interval_scores {
            let (b, q, s, e) = (
                row[0] as usize,
                row[1] as usize,
                row[2] as usize,
                row[3] as usize,
            );
            let got = marginals.inside_interval_sum(b, q, s, e).unwrap();
            m = m.max((got - row[4] as f32).abs());
            let raw: f32 = inside[b][q][s..e].iter().sum();
            assert!(
                (got - raw).abs() < TOL,
                "interval [{s},{e}) q={q}: {got} != inside sum {raw}"
            );
        }
        eprintln!("[synthetic] interval_scores: max abs diff {:.3e}", m);
        assert!(m < TOL, "interval_scores: max abs diff {m:.3e} >= {TOL}");
    }

    /// Production `window=128` is inert at N<=4; re-run attention block 0
    /// with `window=2` on the fixture's layer_norm output and compare.
    #[test]
    fn synthetic_attention_window_matches_python() {
        let fixture = load_fixture();
        let (encoder, _head) = load_modules();
        let device = Device::Cpu;
        let out = &fixture.outputs;

        let ln = Tensor::new(out.after_layer_norm.clone(), &device).unwrap();
        let mask = Tensor::new(out.boundary_mask.clone(), &device).unwrap();
        let got = encoder.attention_block_forward(0, &ln, &mask, 2).unwrap();
        let m = max_diff3(
            &got.to_vec3::<f32>().unwrap(),
            &out.after_attention_0_window_2,
        );
        eprintln!("[synthetic] attention window=2: max abs diff {:.3e}", m);
        assert!(m < TOL, "window=2 block: max abs diff {m:.3e} >= {TOL}");

        // The window must actually bind at N=4 / window=2 (otherwise this
        // comparison would pass trivially against the window=128 run).
        let full = encoder.attention_block_forward(0, &ln, &mask, 0).unwrap();
        let d = max_diff3(
            &full.to_vec3::<f32>().unwrap(),
            &got.to_vec3::<f32>().unwrap(),
        );
        assert!(
            d > 1e-6,
            "window=2 and full attention produce identical outputs"
        );
    }

    /// Focused mask/interval properties (independent of fixture float values).
    #[test]
    fn mask_logit_padding_and_interval_properties() {
        let (encoder, head) = load_modules();
        let device = Device::Cpu;
        let text_states = synthetic_text_states(&device);
        let query_states = synthetic_query_states(&device);
        let text_mask = Tensor::from_vec(vec![1u8, 1, 1, 1, 1, 0], (2, 3), &device).unwrap();
        let query_mask = Tensor::from_vec(vec![1u8, 1, 1, 0], (2, 2), &device).unwrap();
        let stages = encoder.forward_stages(&text_states, &text_mask).unwrap();
        let marginals = head
            .forward(
                &stages.encoding.states,
                &stages.encoding.mask,
                &text_states,
                &text_mask,
                &query_states,
                &query_mask,
            )
            .unwrap();

        // Padding boundary row zeroed (sample 1, boundary 3 > n_b = 2).
        let states = stages.encoding.states.to_vec3::<f32>().unwrap();
        assert!(states[1][3].iter().all(|&v| v == 0.0));
        // Boundary mask: boundary i valid iff i <= n_b.
        assert_eq!(
            stages.encoding.mask.to_vec2::<u8>().unwrap(),
            vec![vec![1, 1, 1, 1], vec![1, 1, 1, 0]]
        );

        // Invalid positions carry the finite sentinel exactly.
        let start = marginals.start_logits.to_vec3::<f32>().unwrap();
        let end = marginals.end_logits.to_vec3::<f32>().unwrap();
        let inside = marginals.inside_logits.to_vec3::<f32>().unwrap();
        assert_eq!(start[1][1][3], MASK_LOGIT, "padded query row");
        assert_eq!(start[1][0][3], MASK_LOGIT, "padded boundary");
        assert_eq!(end[1][1][3], MASK_LOGIT, "padded query row (end)");
        assert_eq!(end[1][0][3], MASK_LOGIT, "padded boundary (end)");
        assert_eq!(inside[1][0][2], MASK_LOGIT, "padded token");
        assert_eq!(inside[1][1][2], MASK_LOGIT, "padded token (query)");

        // Prefix: fp32 origin zero, one-token and edge interval reconstruction
        // against the raw inside sums (mean restore included).
        let prefix = marginals.inside_prefix.to_vec3::<f32>().unwrap();
        for (b, sample) in prefix.iter().enumerate() {
            for (q, row) in sample.iter().enumerate() {
                assert_eq!(row[0], 0.0, "prefix origin b={b} q={q}");
            }
        }
        // The identity holds for intervals inside a sample's valid text of a
        // valid query (masked rows are pure sentinel values; the mean restore
        // is only applied to real intervals, as in pool.py/scoring.py).
        for (b, n, nq_valid) in [(0usize, 3usize, 2usize), (1, 2, 1)] {
            for (q, inside_row) in inside[b].iter().enumerate().take(nq_valid) {
                for (s, e) in [(0, 1), (n - 1, n), (0, n)] {
                    let got = marginals.inside_interval_sum(b, q, s, e).unwrap();
                    let raw: f32 = inside_row[s..e].iter().sum();
                    assert!(
                        (got - raw).abs() < TOL,
                        "one-token/edge interval [{s},{e}) q={q} b={b}: {got} != {raw}"
                    );
                }
            }
        }
    }
}
