/// SpanRepLayer — markerV0 variant as implemented in the gliner package.
///
/// Architecture (derived from actual weight shapes in fastino/gliner2-large-v1):
///
///   project_start(h→4h→h)  applied to start-token embeddings
///   project_end  (h→4h→h)  applied to end-token embeddings
///   out_project  (2h→4h→h) applied to concat(start_proj, end_proj)
///
/// Each "MLP" is a PyTorch Sequential with layout:
///   0: Linear  (has weights)
///   1: GELU    (no params)
///   2: Dropout (no params — skipped at inference)
///   3: Linear  (has weights)
///
/// Weight keys (under whatever prefix the caller supplies, e.g. `span_rep.span_rep_layer`):
///   project_start.0.{weight,bias}  project_start.3.{weight,bias}
///   project_end.0.{weight,bias}    project_end.3.{weight,bias}
///   out_project.0.{weight,bias}    out_project.3.{weight,bias}
///
/// Input:  token_emb  (text_len, hidden_size)
/// Output: span_reps  (text_len, max_width, hidden_size)
///   span_reps[i][w] = representation of span starting at word i with width w+1
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// A 2-layer MLP matching PyTorch `create_mlp(..., activation="gelu")`.
/// Sequential indices: 0=Linear, 1=GELU (no params), 2=Dropout (no params), 3=Linear.
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(in_dim: usize, hidden_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(in_dim, hidden_dim, vb.pp("0"))?;
        let fc2 = linear(hidden_dim, out_dim, vb.pp("3"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.fc2.forward(&self.fc1.forward(x)?.relu()?)?)
    }
}

pub struct SpanRepLayer {
    project_start: Mlp,
    project_end: Mlp,
    out_project: Mlp,
    pub max_width: usize,
}

impl SpanRepLayer {
    pub fn load(hidden_size: usize, max_width: usize, vb: VarBuilder) -> Result<Self> {
        // Each projection: hidden → 4*hidden → hidden
        let project_start = Mlp::load(hidden_size, hidden_size * 4, hidden_size, vb.pp("project_start"))?;
        let project_end   = Mlp::load(hidden_size, hidden_size * 4, hidden_size, vb.pp("project_end"))?;
        // out_project takes concat(start_proj, end_proj) = 2*hidden → 4*hidden → hidden
        let out_project   = Mlp::load(hidden_size * 2, hidden_size * 4, hidden_size, vb.pp("out_project"))?;
        Ok(Self { project_start, project_end, out_project, max_width })
    }

    /// token_emb: (text_len, hidden_size)
    /// Returns:   (text_len, max_width, hidden_size)
    pub fn forward(&self, token_emb: &Tensor) -> Result<Tensor> {
        let (text_len, hidden) = token_emb.dims2()?;
        let n_spans = text_len * self.max_width;
        let device = token_emb.device();

        // Build flat index arrays for start and (clamped) end positions
        let mut start_idx: Vec<u32> = Vec::with_capacity(n_spans);
        let mut end_idx:   Vec<u32> = Vec::with_capacity(n_spans);
        for i in 0..text_len {
            for w in 0..self.max_width {
                start_idx.push(i as u32);
                end_idx.push((i + w).min(text_len - 1) as u32);
            }
        }

        let start_t = Tensor::from_vec(start_idx, (n_spans,), device)?;
        let end_t   = Tensor::from_vec(end_idx,   (n_spans,), device)?;

        // Gather: (n_spans, hidden)
        let start_emb = token_emb.index_select(&start_t, 0)?;
        let end_emb   = token_emb.index_select(&end_t,   0)?;

        // Project start and end independently
        let start_proj = self.project_start.forward(&start_emb)?; // (n_spans, hidden)
        let end_proj   = self.project_end.forward(&end_emb)?;     // (n_spans, hidden)

        // Concatenate, apply ReLU, then out_project.
        // Python: cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
        //         return self.out_project(cat)
        let combined = Tensor::cat(&[&start_proj, &end_proj], 1)?.relu()?; // (n_spans, 2*hidden)
        let projected = self.out_project.forward(&combined)?;                // (n_spans, hidden)

        // Reshape to (text_len, max_width, hidden)
        Ok(projected.reshape((text_len, self.max_width, hidden))?)
    }
}
