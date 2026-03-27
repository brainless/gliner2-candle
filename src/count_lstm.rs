/// CountLSTM — count-aware structure embedding layer.
///
/// Mirrors Python's `CountLSTM` in `gliner2/layers.py` exactly:
///
///   1. pos_embedding(0..gold_count)  → (gold_count, hidden)
///   2. Expand over M fields          → (gold_count, M, hidden)  [PyTorch layout]
///   3. GRU with h0 = field_embs      → output (gold_count, M, hidden)
///   4. Concatenate with field_embs   → (gold_count, M, 2*hidden)
///   5. projector MLP                 → (gold_count, M, hidden)
///
/// Weight keys expected in the safetensors file (under whatever prefix the
/// caller uses, typically `count_embed.`):
///
///   pos_embedding.weight          (max_count, hidden)
///   gru.weight_ih_l0              (3*hidden, hidden)
///   gru.weight_hh_l0              (3*hidden, hidden)
///   gru.bias_ih_l0                (3*hidden,)
///   gru.bias_hh_l0                (3*hidden,)
///   projector.0.weight            (4*hidden, 2*hidden)
///   projector.0.bias              (4*hidden,)
///   projector.2.weight            (hidden, 4*hidden)
///   projector.2.bias              (hidden,)
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{gru, linear, rnn::GRUState, Embedding, GRUConfig, Linear, VarBuilder, RNN};

pub struct CountLSTM {
    pos_embedding: Embedding,
    gru: candle_nn::GRU,
    proj1: Linear, // 2*hidden → 4*hidden
    proj2: Linear, // 4*hidden → hidden
    pub max_count: usize,
}

impl CountLSTM {
    pub fn load(hidden_size: usize, max_count: usize, vb: VarBuilder) -> Result<Self> {
        let pos_embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("pos_embedding"))?;
        let gru = gru(hidden_size, hidden_size, GRUConfig::default(), vb.pp("gru"))?;
        // projector is a Sequential: index 0 = Linear, index 1 = ReLU (no params), index 2 = Linear
        let proj1 = linear(hidden_size * 2, hidden_size * 4, vb.pp("projector").pp("0"))?;
        let proj2 = linear(hidden_size * 4, hidden_size, vb.pp("projector").pp("2"))?;
        Ok(Self {
            pos_embedding,
            gru,
            proj1,
            proj2,
            max_count,
        })
    }

    /// field_embs:  (M, hidden)  — one embedding per entity-type field
    /// gold_count:  predicted (or gold) extraction count, clamped to max_count
    ///
    /// Returns:     (gold_count_clamped, M, hidden)
    pub fn forward(&self, field_embs: &Tensor, gold_count: usize) -> Result<Tensor> {
        let gold_count = gold_count.min(self.max_count);
        if gold_count == 0 {
            let (m, d) = field_embs.dims2()?;
            return Ok(Tensor::zeros(
                (0, m, d),
                field_embs.dtype(),
                field_embs.device(),
            )?);
        }

        let (m, hidden) = field_embs.dims2()?;
        let device = field_embs.device();

        // 1. Positional embeddings: (gold_count, hidden)
        let indices = Tensor::arange(0u32, gold_count as u32, device)?;
        let pos_seq = self.pos_embedding.forward(&indices)?; // (gold_count, hidden)

        // 2. Expand over M fields → (gold_count, M, hidden)
        //    Python: unsqueeze(1).expand(gold_count, M, D)
        let pos_seq = pos_seq
            .unsqueeze(1)? // (gold_count, 1, hidden)
            .broadcast_as((gold_count, m, hidden))?;

        // 3. GRU with h0 = field_embs (M, hidden)
        //    candle GRU: seq_init(input: (batch=M, seq=gold_count, feat), state)
        //    Permute pos_seq from (gold_count, M, hidden) to (M, gold_count, hidden)
        let pos_seq = pos_seq.permute((1, 0, 2))?;
        let init_state = GRUState {
            h: field_embs.clone(),
        };
        let states = self.gru.seq_init(&pos_seq, &init_state)?;

        // Collect GRU outputs: each state.h is (M, hidden) → stack → (M, gold_count, hidden)
        let hs: Vec<Tensor> = states.iter().map(|s| s.h.clone()).collect();
        let gru_out = Tensor::stack(&hs, 1)?; // (M, gold_count, hidden)

        // 4. Concatenate GRU output with original field embeddings
        //    field_embs: (M, hidden) → (M, gold_count, hidden) via broadcast
        let field_expanded = field_embs
            .unsqueeze(1)? // (M, 1, hidden)
            .broadcast_as((m, gold_count, hidden))?;
        let concat = Tensor::cat(&[&gru_out, &field_expanded], 2)?; // (M, gold_count, 2*hidden)

        // 5. Projector MLP: (M, gold_count, 2*hidden) → (M, gold_count, hidden)
        let x = self.proj1.forward(&concat)?.relu()?;
        let output = self.proj2.forward(&x)?; // (M, gold_count, hidden)

        // Permute to (gold_count, M, hidden) to match Python's output layout
        Ok(output.permute((1, 0, 2))?)
    }
}
