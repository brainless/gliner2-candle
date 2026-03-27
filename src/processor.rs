/// Preprocessor for entity extraction — entity task only (minimal version).
///
/// Replicates the relevant parts of Python's `SchemaTransformer` for entity
/// schemas.  The input sequence layout follows GLiNER2:
///
///   ( [P] entities ( [E] type1 [E] type2 ... ) ) [SEP_TEXT]
///   word_1_subtoken_1 word_1_subtoken_2 word_2_subtoken_1 ...
///
/// The tokenizer loaded from the HF repo already has the GLiNER2 special tokens
/// registered in its `added_tokens` list, so they encode as single tokens.
use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

/// Everything the model forward pass needs, for a single text + entity schema.
#[derive(Debug)]
pub struct ProcessedInput {
    /// Combined token ids: schema + [SEP_TEXT] + text subtokens
    pub input_ids: Vec<u32>,
    /// All-ones mask (same length as input_ids)
    pub attention_mask: Vec<u32>,

    /// Token-sequence position of the [SEP_TEXT] token.
    pub sep_text_pos: usize,

    /// Position of the [P] token (for count prediction).
    pub p_token_pos: usize,

    /// For each word in the original text, the index in `input_ids` of its
    /// first subtoken.  Used to extract word-level embeddings from the encoder.
    pub word_token_indices: Vec<usize>,

    /// For each entity type (same order as in the schema), the position in
    /// `input_ids` of its [E] token.  Used to get entity-type embeddings.
    pub entity_e_positions: Vec<usize>,

    /// Original whitespace-split words (for result mapping).
    pub words: Vec<String>,

    /// Character offsets of each word in the original text: (start, end).
    pub word_char_offsets: Vec<(usize, usize)>,

    /// Entity type names in schema order.
    pub entity_types: Vec<String>,
}

pub struct Preprocessor {
    tokenizer: Tokenizer,
    p_id: u32,
    e_id: u32,
    sep_struct_id: u32,
    sep_text_id: u32,
    lp_id: u32, // "("
    rp_id: u32, // ")"
}

impl Preprocessor {
    /// Load from a `tokenizer.json` path (already downloaded from HF hub).
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow!("{e}"))?;
        let p_id = tokenizer
            .token_to_id("[P]")
            .ok_or_else(|| anyhow!("tokenizer missing [P] special token"))?;
        let e_id = tokenizer
            .token_to_id("[E]")
            .ok_or_else(|| anyhow!("tokenizer missing [E] special token"))?;
        let sep_struct_id = tokenizer
            .token_to_id("[SEP_STRUCT]")
            .ok_or_else(|| anyhow!("tokenizer missing [SEP_STRUCT] special token"))?;
        let sep_text_id = tokenizer
            .token_to_id("[SEP_TEXT]")
            .ok_or_else(|| anyhow!("tokenizer missing [SEP_TEXT] special token"))?;
        let lp_id = tokenizer
            .token_to_id("(")
            .ok_or_else(|| anyhow!("tokenizer missing ( token"))?;
        let rp_id = tokenizer
            .token_to_id(")")
            .ok_or_else(|| anyhow!("tokenizer missing ) token"))?;
        Ok(Self {
            tokenizer,
            p_id,
            e_id,
            sep_struct_id,
            sep_text_id,
            lp_id,
            rp_id,
        })
    }

    /// Build the full input sequence for one (text, entity_types) pair.
    pub fn process(&self, text: &str, entity_types: &[String]) -> Result<ProcessedInput> {
        let mut input_ids: Vec<u32> = Vec::new();
        let mut entity_e_positions: Vec<usize> = Vec::new();

        // ── Schema section ────────────────────────────────────────────────────
        // Format: ( [P] entities ( [E] type1 [E] type2 ... ) )
        input_ids.push(self.lp_id); // "("

        let p_token_pos = input_ids.len();
        input_ids.push(self.p_id); // [P]

        // "entities" text
        let enc = self
            .tokenizer
            .encode("entities", false)
            .map_err(|e| anyhow!("{e}"))?;
        input_ids.extend_from_slice(enc.get_ids());

        input_ids.push(self.lp_id); // "("

        // [E] token for each entity type
        for etype in entity_types {
            entity_e_positions.push(input_ids.len());
            input_ids.push(self.e_id); // [E]

            // Tokenize entity type name
            let enc = self
                .tokenizer
                .encode(etype.as_str(), false)
                .map_err(|e| anyhow!("{e}"))?;
            input_ids.extend_from_slice(enc.get_ids());
        }

        input_ids.push(self.rp_id); // ")"
        input_ids.push(self.rp_id); // ")"
        // Note: Python pops the trailing [SEP_STRUCT] for single-schema inputs,
        // so no [SEP_STRUCT] is added here for the entity extraction case.

        // ── [SEP_TEXT] separator ──────────────────────────────────────────────
        let sep_text_pos = input_ids.len();
        input_ids.push(self.sep_text_id);

        // ── Text section ──────────────────────────────────────────────────────
        // Whitespace-split into words, track character offsets
        let (words, word_char_offsets) = whitespace_split(text);
        let mut word_token_indices: Vec<usize> = Vec::with_capacity(words.len());

        for word in &words {
            word_token_indices.push(input_ids.len());
            // Lowercase to match Python's word_splitter(text, lower=True) behaviour.
            let word_lower = word.to_lowercase();
            let enc = self
                .tokenizer
                .encode(word_lower.as_str(), false)
                .map_err(|e| anyhow!("{e}"))?;
            let ids = enc.get_ids();
            if ids.is_empty() {
                // Unknown word — push a single UNK token so indices stay valid
                let unk = self.tokenizer.token_to_id("[UNK]").unwrap_or(0);
                input_ids.push(unk);
            } else {
                input_ids.extend_from_slice(ids);
            }
        }

        let seq_len = input_ids.len();
        let attention_mask = vec![1u32; seq_len];

        Ok(ProcessedInput {
            input_ids,
            attention_mask,
            sep_text_pos,
            p_token_pos,
            word_token_indices,
            entity_e_positions,
            words,
            word_char_offsets,
            entity_types: entity_types.to_vec(),
        })
    }
}

/// Split `text` into tokens and byte offsets, matching Python's `WhitespaceTokenSplitter`:
///   - alphanumeric sequences (plus intra-word hyphens/underscores) become word tokens
///   - every other non-whitespace character becomes its own single-character token
///
/// Tokens are stored with their original casing; callers must lowercase before
/// passing to the tokenizer to match the training-time `lower=True` behaviour.
fn whitespace_split(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut words = Vec::new();
    let mut offsets = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;

    while i < chars.len() {
        let (start_byte, ch) = chars[i];

        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        if ch.is_alphanumeric() || ch == '_' {
            // Collect a word: alphanumeric/underscore, optionally joined by hyphens.
            let mut j = i + 1;
            while j < chars.len() {
                let c = chars[j].1;
                if c.is_alphanumeric() || c == '_' {
                    j += 1;
                } else if c == '-' && j + 1 < chars.len() && chars[j + 1].1.is_alphanumeric() {
                    // Intra-word hyphen (e.g. "well-known") — consume hyphen + next char.
                    j += 2;
                } else {
                    break;
                }
            }
            let end_byte = chars.get(j).map(|(b, _)| *b).unwrap_or(text.len());
            words.push(text[start_byte..end_byte].to_string());
            offsets.push((start_byte, end_byte));
            i = j;
        } else {
            // Non-whitespace, non-alphanumeric: emit as a single-character token
            // (punctuation, symbols, etc.).
            let end_byte = chars.get(i + 1).map(|(b, _)| *b).unwrap_or(text.len());
            words.push(ch.to_string());
            offsets.push((start_byte, end_byte));
            i += 1;
        }
    }

    (words, offsets)
}
