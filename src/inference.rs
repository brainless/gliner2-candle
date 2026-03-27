/// Post-processing: turn raw span scores into `ExtractedEntity` results.
///
/// Pipeline:
///   1. sigmoid(scores)
///   2. Keep spans where any count-step score > threshold
///   3. Greedy overlap removal (highest confidence first)
///   4. Map word-span → character offsets
use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use candle_nn::ops::sigmoid;

use crate::processor::ProcessedInput;

#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,
    /// Byte offset in the original string (inclusive)
    pub char_start: usize,
    /// Byte offset in the original string (exclusive)
    pub char_end: usize,
    pub confidence: f32,
}

/// Extract entities from raw model scores.
///
/// `scores` shape: (gold_count, num_entity_types, text_len, max_width)
///
/// For each (entity_type, start_word, width) triple we aggregate over the
/// gold_count dimension by taking the max score across count steps — this
/// produces a single confidence per span, matching inference behaviour in
/// Python's `engine.py`.
pub fn extract_entities(
    scores: &Tensor,
    input: &ProcessedInput,
    threshold: f32,
) -> Result<Vec<ExtractedEntity>> {
    // sigmoid in-place
    let probs = sigmoid(scores)?;

    // Use count step 0 only, matching Python's _extract_entities which does
    // span_scores[0, :, ...].  For entity extraction gold_count is trained as 1,
    // so step 0 carries all the entity-level scores.
    let probs = probs.i(0)?; // (num_types, text_len, max_width)

    let (num_types, text_len, max_width) = probs.dims3()?;
    let probs_vec = probs.to_vec3::<f32>()?;

    // Collect candidates: (confidence, type_idx, word_start, word_end_inclusive)
    let mut candidates: Vec<(f32, usize, usize, usize)> = Vec::new();
    for t in 0..num_types {
        for s in 0..text_len {
            for w in 0..max_width {
                let end_word = s + w;
                if end_word >= text_len {
                    break; // out of bounds
                }
                let conf = probs_vec[t][s][w];
                if conf >= threshold {
                    candidates.push((conf, t, s, end_word));
                }
            }
        }
    }

    // Sort descending by confidence
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy overlap removal: track which word positions are already claimed
    let mut occupied = vec![false; text_len];
    let mut results: Vec<ExtractedEntity> = Vec::new();

    for (conf, type_idx, word_start, word_end) in candidates {
        // Check if any word in this span is already occupied
        if occupied[word_start..=word_end].iter().any(|&x| x) {
            continue;
        }
        // Mark occupied
        for pos in word_start..=word_end {
            occupied[pos] = true;
        }

        // Map word offsets → character offsets
        let char_start = input.word_char_offsets[word_start].0;
        let char_end = input.word_char_offsets[word_end].1;
        let span_text = input.words[word_start..=word_end].join(" ");

        results.push(ExtractedEntity {
            text: span_text,
            entity_type: input.entity_types[type_idx].clone(),
            char_start,
            char_end,
            confidence: conf,
        });
    }

    // Return sorted by position for readability
    results.sort_by_key(|e| e.char_start);
    Ok(results)
}
