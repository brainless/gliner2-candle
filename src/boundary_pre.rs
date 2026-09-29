//! Boundary-path preprocessing and query routing (epic `gliner-2.5-boundary-support`,
//! Task 4).
//!
//! Rust port, at pinned GLiNER2 commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, of:
//! * the entity/relation schema path of `SchemaTransformer`
//!   (`gliner2/processor.py`): `_collate_batch` text normalization,
//!   `_process_entities`/`_process_relations`/`_transform_schema` schema-token
//!   order, the `WhitespaceTokenSplitter` word split, and
//!   `_format_input_with_mapping` marker/word routing;
//! * the query layout enumeration of `build_boundary_batch_metadata`
//!   (`gliner2/processing/boundary_preprocessing.py`) — every extractive
//!   schema child (`[E]`/`[R]` marker) becomes one boundary query in marker
//!   order;
//! * the relation role routing of `BoundaryExtractorModel._encode_core`
//!   (`gliner2/models/boundary/model.py`) — head/tail role queries per
//!   relation group and the relation query-state mode
//!   (`concat(head,tail)` when `directional_relation_states`, else `mean`).
//!
//! Boundary-specific on purpose: the span path (`src/processor.rs`) keeps its
//! own splitter (byte offsets, no URL/email/@mention recognition) and is not
//! touched by this module.
//!
//! ## Offset conventions
//!
//! * Python records word offsets as Unicode **code-point** indices into the
//!   *normalized* text (`_collate_batch` appends a synthetic `'.'` to input
//!   without terminal `'.'`/`'!'`/`'?'` **before** splitting, and `""` becomes
//!   `"."`; the normalized text is what Python later decodes against). This
//!   module keeps that normalized token sequence for model parity and converts
//!   offsets to UTF-8 **byte** indices immediately.
//! * [`TextMap`] retains the caller's exact input separately and maps
//!   half-open word intervals `[start,end)` (candidate-decoder convention:
//!   `start_mappings[start]`, `end_mappings[end-1]`) to byte ranges in the
//!   caller's input. A candidate whose mapped range reaches into the synthetic
//!   suffix is **rejected** ([`TextMap::candidate_bytes`] returns `None`):
//!   Python decodes against the normalized text and can surface the synthetic
//!   `'.'`, but Rust must never return a surface or byte range outside the
//!   caller's input. This divergence is exercised in the tests below and is
//!   the user-facing difference documented in DEVELOP.md.
// The output vocabulary (BoundaryPrepared/TextMap/QueryEntry/RelationRoute)
// feeds epic Tasks 5–8 (boundary encode/decoders) and the parity tests below;
// until those land the binary crate sees much of it as unused.
#![allow(dead_code)]
use std::collections::HashSet;
use std::ops::Range;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use regex::Regex;
use tokenizers::Tokenizer;

const SEP_STRUCT: &str = "[SEP_STRUCT]";
const SEP_TEXT: &str = "[SEP_TEXT]";
const P_TOKEN: &str = "[P]";
const E_TOKEN: &str = "[E]";
const R_TOKEN: &str = "[R]";
const C_TOKEN: &str = "[C]";

/// Extractive child markers; anything else is prompt text (Python
/// `boundary_preprocessing._EXTRACTIVE_MARKERS`).
const EXTRACTIVE_MARKERS: [&str; 3] = [E_TOKEN, C_TOKEN, R_TOKEN];

/// Relation query-state composition (Python `_encode_core`, gated by
/// `boundary_head.directional_relation_states`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationQueryState {
    /// `mean(head, tail)` — undirected relation query.
    Mean,
    /// `concat(head, tail)` — directed relation query (2 × hidden).
    Concat,
}

impl RelationQueryState {
    /// Oracle/Python spelling of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean(head,tail)",
            Self::Concat => "concat(head,tail)",
        }
    }
}

/// Per-relation-type query routing (Python `_encode_core` `rel_specs` entry).
#[derive(Debug, Clone)]
pub struct RelationRoute {
    /// Group name (relation type) in transformed schema order.
    pub relation_type: String,
    /// Task/group index of the relation schema in [`BoundaryPrepared::task_types`].
    pub group_index: usize,
    /// Global query id of the `[R] head` role marker (single element, as in
    /// Python's `RelationTypeSpec`).
    pub head_query_id: Vec<usize>,
    /// Global query id of the `[R] tail` role marker.
    pub tail_query_id: Vec<usize>,
    /// Configured relation query-state mode.
    pub query_state: RelationQueryState,
    /// Query-state dimension (2 × hidden when concat, else hidden).
    pub query_state_dim: usize,
}

/// One boundary query (Python `QuerySpec` as enumerated by
/// `build_boundary_batch_metadata`).
#[derive(Debug, Clone)]
pub struct QueryEntry {
    pub query_id: usize,
    /// Group index of the owning schema (`task_index` in the oracle fixtures).
    pub task_index: usize,
    /// `"entities"` or `"relations"`.
    pub task_type: String,
    /// Group name (`"entities"` or the relation type).
    pub task_name: String,
    /// Role index within the group (marker order).
    pub field_index: usize,
    /// Output type name — the token after this query's `[E]`/`[R]` marker.
    pub field_name: String,
}

/// Normalized/caller text pair plus word maps (see module docs for the
/// offset conventions).
#[derive(Debug, Clone)]
pub struct TextMap {
    caller_text: String,
    normalized_text: String,
    /// Code-point length of the caller's input.
    caller_len_cp: usize,
    /// Code-point start of the synthetic `'.'` suffix (`None` when the caller's
    /// text already ended with `'.'`/`'!'`/`'?'`).
    suffix_cp_start: Option<usize>,
    /// `(cp_start, cp_end)` of each split word in `normalized_text`.
    word_cp_spans: Vec<(usize, usize)>,
    /// cp index → byte index in `normalized_text` (length = n_cp + 1).
    cp_to_byte: Vec<usize>,
}

impl TextMap {
    /// Exact caller input (for CLI byte offsets and surface slicing).
    pub fn caller_text(&self) -> &str {
        &self.caller_text
    }

    /// Normalized encoding text (caller input plus the synthetic `'.'` when
    /// needed) — the token sequence is built from this, and Python decodes
    /// against it.
    pub fn normalized_text(&self) -> &str {
        &self.normalized_text
    }

    /// Code-point start of the synthetic suffix (`None` if no suffix).
    pub fn suffix_cp_start(&self) -> Option<usize> {
        self.suffix_cp_start
    }

    /// `start_mappings[word]` — code-point start of a word in the normalized
    /// text (Python `start_mappings`).
    pub fn start_mapping(&self, word: usize) -> usize {
        self.word_cp_spans[word].0
    }

    /// `end_mappings[word]` — code-point end of a word in the normalized text
    /// (Python `end_mappings`).
    pub fn end_mapping(&self, word: usize) -> usize {
        self.word_cp_spans[word].1
    }

    /// Map a half-open word interval `[start, end)` onto UTF-8 byte offsets in
    /// the caller's input (Python maps characters via `start_mappings[start]`
    /// and `end_mappings[end-1]`).
    ///
    /// Returns `None` when the interval is empty/invalid or when its mapped
    /// character range reaches into the synthetic `'.'` suffix — such
    /// candidates must be rejected in full (epic §1); Python would decode them
    /// against `normalized_text` and could surface the synthetic `'.'`.
    pub fn candidate_bytes(&self, start: usize, end: usize) -> Option<Range<usize>> {
        if start >= end || end > self.word_cp_spans.len() {
            return None;
        }
        let cp_start = self.word_cp_spans[start].0;
        let cp_end = self.word_cp_spans[end - 1].1;
        if cp_end > self.caller_len_cp {
            return None;
        }
        Some(self.cp_to_byte[cp_start]..self.cp_to_byte[cp_end])
    }

    /// Slice the caller's input at a mapped byte range; the surface is by
    /// construction inside the caller's exact input.
    pub fn surface(&self, range: Range<usize>) -> &str {
        &self.caller_text[range]
    }
}

/// Everything the boundary encode path (epic Task 5+) needs for one request,
/// aligned field-for-field with the `oracle/cases/NN_*.json` preprocessing
/// fixtures.
#[derive(Debug, Clone)]
pub struct BoundaryPrepared {
    /// Schema tokens + `[SEP_TEXT]` + text subtokens.
    pub input_ids: Vec<u32>,
    /// Lowercased split words of the normalized text (after `max_len` truncation).
    pub words: Vec<String>,
    /// Byte offsets `(start, end)` of each word in [`TextMap::normalized_text`].
    pub word_char_spans: Vec<(usize, usize)>,
    /// `input_ids` position of each word's first subtoken (word/first-subtoken
    /// pooling routing; one entry per word even when a word tokenizes to
    /// nothing — a placeholder keeps the 1:1 alignment, as in Python).
    pub first_subtoken_positions: Vec<usize>,
    /// `input_ids` positions of the `[E]`/`[R]` query markers, flattened in
    /// group/marker order (Python `query_marker_indices`).
    pub query_marker_positions: Vec<usize>,
    /// Per group: `input_ids` positions of `[P]` followed by that group's
    /// child markers (Python `schema_special_indices`).
    pub schema_marker_positions: Vec<Vec<usize>>,
    /// Per group: the schema token strings in order.
    pub schema_tokens: Vec<Vec<String>>,
    /// Per group: `"entities"` or `"relations"`.
    pub task_types: Vec<String>,
    /// Query enumeration in transformed marker order.
    pub query_layout: Vec<QueryEntry>,
    /// Relation role routing (`[]` when no relation schema).
    pub relation_role_routing: Vec<RelationRoute>,
    /// Normalized/caller text and word-offset maps.
    pub text_map: TextMap,
}

impl BoundaryPrepared {
    /// Map a candidate half-open word interval to byte offsets in the caller's
    /// input (see [`TextMap::candidate_bytes`]).
    pub fn candidate_bytes(&self, start: usize, end: usize) -> Option<Range<usize>> {
        self.text_map.candidate_bytes(start, end)
    }
}

/// Request-level preparation options.
#[derive(Debug, Clone, Copy)]
pub struct PrepareOptions {
    /// Request `max_len`: maximum number of **word tokens** kept
    /// (Python `collate_fn_inference(max_len=...)` truncates the split words
    /// before schema encoding; `None` = no truncation). The checkpoint
    /// config's `max_len` is not applied at inference — matching Python,
    /// where `extract()` passes `max_len=None`.
    pub max_len: Option<usize>,
    /// `boundary_head.directional_relation_states` — relation query state is
    /// `concat(head,tail)` when true, `mean(head,tail)` when false.
    pub directional_relation_states: bool,
    /// Encoder hidden size (relation query-state dimension).
    pub hidden_size: usize,
}

/// Boundary preprocessor: tokenizer + schema/word routing (port of the
/// `SchemaTransformer` entity/relation path).
pub struct BoundaryPreprocessor {
    tokenizer: Tokenizer,
}

/// One transformed schema group (Python `_process_entities`/`_process_relations`
/// output row: schema tokens + task type + group name).
struct SchemaGroup {
    schema_tokens: Vec<String>,
    task_type: &'static str,
    task_name: String,
}

impl BoundaryPreprocessor {
    /// Load from a checkpoint's `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("loading tokenizer {:?}: {e}", path.as_ref()))?;
        Ok(Self { tokenizer })
    }

    /// Build the full boundary input for one `(text, entity types, relation
    /// types)` request: schema + `[SEP_TEXT]` + normalized-text words, marker
    /// routing, query layout, relation role routing, and the offset map.
    ///
    /// Entity and relation type names follow Python's transformed schema order:
    /// the entities group first (one `[E]` marker per type in declared order),
    /// then one relations group per relation type in declared order (`[R] head`
    /// / `[R] tail` role markers). Duplicate names collapse to their first
    /// appearance (Python dict/group-key semantics).
    pub fn prepare(
        &self,
        text: &str,
        entity_types: &[String],
        relation_types: &[String],
        opts: &PrepareOptions,
    ) -> Result<BoundaryPrepared> {
        let groups = build_schema_groups(entity_types, relation_types);
        let (normalized_text, suffix_cp_start) = normalize_text(text);

        // Python `word_splitter(text, lower=True)`: offsets index the original
        // (normalized) string; only the token value is lower-cased.
        let mut words = Vec::new();
        let mut word_byte_spans = Vec::new();
        for (token, start, end) in split_words(&normalized_text) {
            words.push(token);
            word_byte_spans.push((start, end));
        }
        if let Some(max_len) = opts.max_len {
            words.truncate(max_len);
            word_byte_spans.truncate(max_len);
        }

        let schema_tokens_list: Vec<Vec<String>> =
            groups.iter().map(|g| g.schema_tokens.clone()).collect();
        let formatted = self.format_input(&schema_tokens_list, &words)?;

        // Query layout: every extractive schema child becomes one query, in
        // group/marker order (`build_boundary_batch_metadata`).
        let mut query_layout = Vec::new();
        let mut query_id = 0usize;
        for (task_index, group) in groups.iter().enumerate() {
            for (field_index, field_name) in extractive_fields(&group.schema_tokens) {
                query_layout.push(QueryEntry {
                    query_id,
                    task_index,
                    task_type: group.task_type.to_string(),
                    task_name: group.task_name.clone(),
                    field_index,
                    field_name,
                });
                query_id += 1;
            }
        }

        // Relation role routing (`_encode_core` rel_specs): one route per
        // relations group with ≥ 2 role queries.
        let mut relation_role_routing = Vec::new();
        for (task_index, group) in groups.iter().enumerate() {
            if group.task_type != "relations" {
                continue;
            }
            let role_ids: Vec<usize> = query_layout
                .iter()
                .filter(|q| q.task_index == task_index)
                .map(|q| q.query_id)
                .collect();
            if role_ids.len() < 2 {
                continue;
            }
            let query_state = if opts.directional_relation_states {
                RelationQueryState::Concat
            } else {
                RelationQueryState::Mean
            };
            relation_role_routing.push(RelationRoute {
                relation_type: group.task_name.clone(),
                group_index: task_index,
                head_query_id: vec![role_ids[0]],
                tail_query_id: vec![role_ids[1]],
                query_state,
                query_state_dim: match query_state {
                    RelationQueryState::Concat => 2 * opts.hidden_size,
                    RelationQueryState::Mean => opts.hidden_size,
                },
            });
        }

        let query_marker_positions = formatted
            .schema_special_positions
            .iter()
            .flat_map(|positions| positions.iter().skip(1).copied())
            .collect();

        let text_map = TextMap::new(
            text.to_string(),
            normalized_text,
            suffix_cp_start,
            &word_byte_spans,
        );

        Ok(BoundaryPrepared {
            input_ids: formatted.input_ids,
            words,
            word_char_spans: word_byte_spans,
            first_subtoken_positions: formatted.text_word_first_positions,
            query_marker_positions,
            schema_marker_positions: formatted.schema_special_positions,
            schema_tokens: schema_tokens_list,
            task_types: groups.iter().map(|g| g.task_type.to_string()).collect(),
            query_layout,
            relation_role_routing,
            text_map,
        })
    }

    /// Port of Python `_format_input_with_mapping`: concatenates schema groups
    /// (`[SEP_STRUCT]`-separated), `[SEP_TEXT]`, and the text words; each
    /// combined token is tokenized **in isolation** and concatenated, and the
    /// marker/word routing positions are collected.
    fn format_input(
        &self,
        schema_tokens_list: &[Vec<String>],
        text_tokens: &[String],
    ) -> Result<FormattedInput> {
        let mut combined: Vec<String> = Vec::new();
        for struct_tokens in schema_tokens_list {
            combined.extend(struct_tokens.iter().cloned());
            combined.push(SEP_STRUCT.to_string());
        }
        if !combined.is_empty() {
            // Python pops the trailing [SEP_STRUCT] after the last schema.
            combined.pop();
        }
        combined.push(SEP_TEXT.to_string());
        combined.extend(text_tokens.iter().cloned());

        // Structural marker slots only: [P] at index 1 and child markers at
        // indices 4, 6, … < len-2 of each schema (prompt text and label names
        // are never routed even if they tokenize to special-token ids).
        let mut marker_orig: HashSet<usize> = HashSet::new();
        let mut offset = 0usize;
        for struct_tokens in schema_tokens_list {
            if struct_tokens.len() > 1 {
                marker_orig.insert(offset + 1);
            }
            let mut index = 4usize;
            while index < struct_tokens.len().saturating_sub(2) {
                marker_orig.insert(offset + index);
                index += 2;
            }
            offset += struct_tokens.len() + 1; // schema tokens plus [SEP_STRUCT]
        }

        let num_schemas = schema_tokens_list.len();
        let mut input_ids: Vec<u32> = Vec::new();
        let mut text_word_first_positions: Vec<usize> = Vec::new();
        let mut schema_special_positions: Vec<Vec<usize>> = vec![Vec::new(); num_schemas];

        let mut current_schema = 0usize;
        let mut found_sep = false;
        let mut last_text_orig: Option<usize> = None;

        for (orig_idx, token) in combined.iter().enumerate() {
            let (is_text, schema_idx) = if token == SEP_TEXT {
                found_sep = true;
                (false, num_schemas)
            } else if !found_sep {
                let idx = current_schema;
                if token == SEP_STRUCT {
                    current_schema += 1;
                }
                (false, idx)
            } else {
                (true, num_schemas)
            };

            let subword_pos = input_ids.len();
            let ids = self
                .tokenizer
                .encode(token.as_str(), false)
                .map_err(|e| anyhow!("tokenizing {token:?}: {e}"))?
                .get_ids()
                .to_vec();

            if is_text {
                // One routing row per text word, in order (distinct `orig_idx`
                // per word). A word that tokenizes to nothing would otherwise
                // be dropped while char-offset mappings keep a row, shifting
                // the embedding↔word alignment: keep a placeholder row at the
                // current subword boundary, as Python does.
                if Some(orig_idx) != last_text_orig {
                    last_text_orig = Some(orig_idx);
                    if ids.is_empty() {
                        tracing::warn!(
                            "text word {token:?} (index {orig_idx}) produced no subwords; \
                             inserting a placeholder to preserve word/embedding alignment"
                        );
                    }
                    text_word_first_positions.push(subword_pos);
                }
            } else if marker_orig.contains(&orig_idx) {
                schema_special_positions[schema_idx].push(subword_pos);
            }
            input_ids.extend(ids);
        }

        Ok(FormattedInput {
            input_ids,
            text_word_first_positions,
            schema_special_positions,
        })
    }
}

struct FormattedInput {
    input_ids: Vec<u32>,
    text_word_first_positions: Vec<usize>,
    schema_special_positions: Vec<Vec<usize>>,
}

impl TextMap {
    /// Build the map from an already-split word list (`word_byte_spans` are
    /// UTF-8 byte offsets into `normalized_text`). `pub(crate)` so the Task 7
    /// decoder tests can construct maps without the tokenizer.
    pub(crate) fn new(
        caller_text: String,
        normalized_text: String,
        suffix_cp_start: Option<usize>,
        word_byte_spans: &[(usize, usize)],
    ) -> Self {
        // cp → byte table for `normalized_text` (one entry per cp + sentinel).
        let mut cp_to_byte: Vec<usize> = normalized_text.char_indices().map(|(b, _)| b).collect();
        cp_to_byte.push(normalized_text.len());
        // Splitter offsets are UTF-8 bytes; Python records code points. Every
        // word boundary is a char boundary, so its cp index is the number of
        // boundaries before it.
        let cp_of = |byte: usize| cp_to_byte.partition_point(|&b| b < byte);
        let word_cp_spans = word_byte_spans
            .iter()
            .map(|&(s, e)| (cp_of(s), cp_of(e)))
            .collect();

        let caller_len_cp = caller_text.chars().count();
        Self {
            caller_text,
            normalized_text,
            caller_len_cp,
            suffix_cp_start,
            word_cp_spans,
            cp_to_byte,
        }
    }
}

/// Python `SchemaTransformer._normalize_text`: `""` becomes `"."`; text not
/// ending with `'.'`/`'!'`/`'?'` gets a synthetic `'.'` appended **before**
/// word splitting. Returns the normalized text and the code-point start of the
/// synthetic suffix (`None` when the caller's text was already normalized).
fn normalize_text(text: &str) -> (String, Option<usize>) {
    if text.is_empty() {
        return (".".to_string(), Some(0));
    }
    if text.ends_with(['.', '!', '?']) {
        (text.to_string(), None)
    } else {
        (format!("{text}."), Some(text.chars().count()))
    }
}

/// Python `boundary_preprocessing._group_name`: entities groups are named
/// `"entities"`; other groups take the prompt token (`schema_tokens[2]`) up to
/// any `" [DESCRIPTION] "` suffix.
fn group_name(schema_tokens: &[String], task_type: &str) -> String {
    if task_type == "entities" {
        return "entities".to_string();
    }
    if schema_tokens.len() > 2 {
        return schema_tokens[2]
            .split(" [DESCRIPTION] ")
            .next()
            .unwrap_or(schema_tokens[2].as_str())
            .to_string();
    }
    task_type.to_string()
}

/// Python `boundary_preprocessing._extractive_fields`: the token following
/// each `[E]`/`[C]`/`[R]` marker is that query's output type name.
fn extractive_fields(schema_tokens: &[String]) -> Vec<(usize, String)> {
    let mut fields = Vec::new();
    for (i, token) in schema_tokens.iter().enumerate() {
        if i + 1 >= schema_tokens.len() {
            break;
        }
        if EXTRACTIVE_MARKERS.contains(&token.as_str()) {
            fields.push((fields.len(), schema_tokens[i + 1].clone()));
        }
    }
    fields
}

/// Port of `SchemaTransformer._process_entities` + `_process_relations` +
/// `_transform_schema` (inference mode, no descriptions/examples): the
/// entities group first, then one relations group per relation type in
/// declared order; `[E]`/`[R]` markers in field order; duplicate type names
/// collapse to first appearance (Python dict / group-key semantics).
fn build_schema_groups(entity_types: &[String], relation_types: &[String]) -> Vec<SchemaGroup> {
    let mut groups: Vec<SchemaGroup> = Vec::new();

    let mut seen = HashSet::new();
    let entity_fields: Vec<&String> = entity_types
        .iter()
        .filter(|name| seen.insert((*name).clone()))
        .collect();
    if !entity_fields.is_empty() {
        let mut schema_tokens = vec![
            "(".to_string(),
            P_TOKEN.to_string(),
            "entities".to_string(),
            "(".to_string(),
        ];
        for field in &entity_fields {
            schema_tokens.push(E_TOKEN.to_string());
            schema_tokens.push((*field).clone());
        }
        schema_tokens.push(")".to_string());
        schema_tokens.push(")".to_string());
        let task_name = group_name(&schema_tokens, "entities");
        groups.push(SchemaGroup {
            schema_tokens,
            task_type: "entities",
            task_name,
        });
    }

    let mut seen_relations = HashSet::new();
    for relation in relation_types {
        if !seen_relations.insert(relation.clone()) {
            continue;
        }
        let schema_tokens = vec![
            "(".to_string(),
            P_TOKEN.to_string(),
            relation.clone(),
            "(".to_string(),
            R_TOKEN.to_string(),
            "head".to_string(),
            R_TOKEN.to_string(),
            "tail".to_string(),
            ")".to_string(),
            ")".to_string(),
        ];
        let task_name = group_name(&schema_tokens, "relations");
        groups.push(SchemaGroup {
            schema_tokens,
            task_type: "relations",
            task_name,
        });
    }

    groups
}

/// Port of Python `WhitespaceTokenSplitter` (`gliner2/processing/word_splitter.py`):
///
/// ```text
/// (?:https?://[^\s]+|www\.[^\s]+)
/// |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
/// |@[a-z0-9_]+
/// |\w+(?:[-_]\w+)*
/// |\S
/// ```
///
/// run with `re.VERBOSE | re.IGNORECASE` over the **original** string via
/// `finditer`, then lower-casing only the matched token value (`str.lower()`),
/// so offsets index the source string. Here `\w` is written out as Python's
/// `str` word class (`{L*, Nd, Nl, No, '_'}`) and `\S`/`[^\s]` complement
/// Python's `str` whitespace (which also includes `\x1c`–`\x1f`). Offsets
/// returned are UTF-8 byte ranges into `text`.
fn split_words(text: &str) -> Vec<(String, usize, usize)> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let re = PATTERN.get_or_init(|| {
        Regex::new(concat!(
            r"(?i)(?:https?://[^\s\x1c-\x1f]+|www\.[^\s\x1c-\x1f]+)",
            r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
            r"|@[a-z0-9_]+",
            r"|[\p{L}\p{Nd}\p{Nl}\p{No}_]+(?:[-_][\p{L}\p{Nd}\p{Nl}\p{No}_]+)*",
            r"|[^\s\x1c-\x1f]",
        ))
        .expect("word-split pattern is valid")
    });
    re.find_iter(text)
        .map(|m| {
            let token = m.as_str().to_lowercase();
            (token, m.start(), m.end())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_path(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    const MODEL_DIR: &str = "models/gliner2.5-small-v1";

    /// Oracle fixture fields compared 1:1 against [`BoundaryPrepared`].
    #[derive(Debug, serde::Deserialize)]
    struct OracleCase {
        case_id: String,
        text: String,
        normalized_text: String,
        token_ids: Vec<u32>,
        words: Vec<String>,
        word_char_spans: Vec<(usize, usize)>,
        first_subtoken_positions: Vec<usize>,
        query_marker_positions: Vec<usize>,
        schema_marker_positions: Vec<Vec<usize>>,
        schema_tokens: Vec<Vec<String>>,
        task_types: Vec<String>,
        word_offset: usize,
        text_word_count: usize,
        entity_types: Vec<String>,
        relation_types: Vec<String>,
        query_layout: Vec<OracleQuery>,
        relation_role_routing: Vec<OracleRoute>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct OracleQuery {
        query_id: usize,
        task_index: usize,
        task_type: String,
        task_name: String,
        field_index: usize,
        field_name: String,
    }

    #[derive(Debug, serde::Deserialize)]
    struct OracleRoute {
        relation_type: String,
        group_index: usize,
        head_query_id: Vec<usize>,
        tail_query_id: Vec<usize>,
        query_state: String,
        query_state_dim: usize,
    }

    fn load_oracle_cases() -> Vec<OracleCase> {
        let cases_dir = repo_path("oracle/cases");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&cases_dir)
            .unwrap_or_else(|e| panic!("reading {:?}: {e}", cases_dir))
            .map(|entry| entry.unwrap().path())
            .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
            .collect();
        files.sort();
        assert_eq!(
            files.len(),
            18,
            "expected 18 oracle cases in {:?}",
            cases_dir
        );
        files
            .iter()
            .map(|p| {
                let raw = std::fs::read_to_string(p).unwrap();
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {:?}: {e}", p))
            })
            .collect()
    }

    fn load_prepared() -> (BoundaryPreprocessor, PrepareOptions) {
        let dir = repo_path(MODEL_DIR);
        let tokenizer_path = dir.join("tokenizer.json");
        assert!(
            tokenizer_path.exists(),
            "oracle parity requires the boundary checkpoint: \
             hf download fastino/gliner2.5-small-v1 --local-dir ./{MODEL_DIR}"
        );
        let gliner_cfg = crate::config::Gliner2Config::from_file(dir.join("config.json"))
            .expect("parse boundary config.json");
        let enc_cfg =
            crate::config::encoder_config_from_file(dir.join("encoder_config").join("config.json"))
                .expect("parse encoder_config/config.json");
        let opts = PrepareOptions {
            max_len: None, // Python extract() passes max_len=None (no truncation)
            directional_relation_states: gliner_cfg.boundary_head.directional_relation_states,
            hidden_size: crate::config::hidden_size(&enc_cfg),
        };
        (
            BoundaryPreprocessor::from_file(tokenizer_path).expect("load tokenizer.json"),
            opts,
        )
    }

    /// Fixture `word_char_spans` index `normalized_text` in Unicode code
    /// points; convert to UTF-8 byte offsets for comparison with Rust.
    fn cp_spans_to_byte_spans(normalized: &str, spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut cp_to_byte: Vec<usize> = normalized.char_indices().map(|(b, _)| b).collect();
        cp_to_byte.push(normalized.len());
        spans
            .iter()
            .map(|&(s, e)| (cp_to_byte[s], cp_to_byte[e]))
            .collect()
    }

    /// Case-insensitive surface/word equality (words are lower-cased copies of
    /// the matched source slices).
    fn surface_matches_word(surface: &str, word: &str) -> bool {
        surface.to_lowercase() == word
    }

    #[test]
    fn oracle_parity_all_cases() {
        let (preprocessor, opts) = load_prepared();
        let cases = load_oracle_cases();
        for case in &cases {
            let id = &case.case_id;
            let prepared = preprocessor
                .prepare(&case.text, &case.entity_types, &case.relation_types, &opts)
                .unwrap_or_else(|e| panic!("[{id}] prepare failed: {e}"));

            assert_eq!(prepared.input_ids, case.token_ids, "[{id}] token_ids");
            assert_eq!(
                prepared.first_subtoken_positions, case.first_subtoken_positions,
                "[{id}] first_subtoken_positions"
            );
            assert_eq!(
                prepared.query_marker_positions, case.query_marker_positions,
                "[{id}] query_marker_positions"
            );
            assert_eq!(
                prepared.schema_marker_positions, case.schema_marker_positions,
                "[{id}] schema_marker_positions"
            );
            assert_eq!(
                prepared.schema_tokens, case.schema_tokens,
                "[{id}] schema_tokens"
            );
            assert_eq!(prepared.task_types, case.task_types, "[{id}] task_types");
            assert_eq!(prepared.words, case.words, "[{id}] words");
            assert_eq!(
                prepared.text_map.normalized_text(),
                case.normalized_text,
                "[{id}] normalized_text"
            );
            let expected_spans =
                cp_spans_to_byte_spans(&case.normalized_text, &case.word_char_spans);
            assert_eq!(
                prepared.word_char_spans, expected_spans,
                "[{id}] word_char_spans (fixture code points → UTF-8 bytes)"
            );
            assert_eq!(
                case.word_offset, 0,
                "[{id}] word_offset (no classification prefix)"
            );
            assert_eq!(
                case.text_word_count,
                prepared.words.len(),
                "[{id}] text_word_count"
            );

            assert_eq!(
                prepared.query_layout.len(),
                case.query_layout.len(),
                "[{id}] query_layout length"
            );
            for (got, want) in prepared.query_layout.iter().zip(&case.query_layout) {
                assert_eq!(got.query_id, want.query_id, "[{id}] query_layout.query_id");
                assert_eq!(
                    got.task_index, want.task_index,
                    "[{id}] query_layout.task_index"
                );
                assert_eq!(
                    got.task_type, want.task_type,
                    "[{id}] query_layout.task_type"
                );
                assert_eq!(
                    got.task_name, want.task_name,
                    "[{id}] query_layout.task_name"
                );
                assert_eq!(
                    got.field_index, want.field_index,
                    "[{id}] query_layout.field_index"
                );
                assert_eq!(
                    got.field_name, want.field_name,
                    "[{id}] query_layout.field_name"
                );
            }

            assert_eq!(
                prepared.relation_role_routing.len(),
                case.relation_role_routing.len(),
                "[{id}] relation_role_routing length"
            );
            for (got, want) in prepared
                .relation_role_routing
                .iter()
                .zip(&case.relation_role_routing)
            {
                assert_eq!(
                    got.relation_type, want.relation_type,
                    "[{id}] route.relation_type"
                );
                assert_eq!(
                    got.group_index, want.group_index,
                    "[{id}] route.group_index"
                );
                assert_eq!(
                    got.head_query_id, want.head_query_id,
                    "[{id}] route.head_query_id"
                );
                assert_eq!(
                    got.tail_query_id, want.tail_query_id,
                    "[{id}] route.tail_query_id"
                );
                assert_eq!(
                    got.query_state.as_str(),
                    want.query_state,
                    "[{id}] route.query_state"
                );
                assert_eq!(
                    got.query_state_dim, want.query_state_dim,
                    "[{id}] route.query_state_dim"
                );
            }
        }
    }

    /// Every mapped single word must slice the caller's input to a surface that
    /// case-folds to the word; words lying in the synthetic suffix must map to
    /// `None` (verified over all 18 cases).
    #[test]
    fn surfaces_slice_caller_input() {
        let (preprocessor, opts) = load_prepared();
        for case in load_oracle_cases() {
            let id = case.case_id.clone();
            let prepared = preprocessor
                .prepare(&case.text, &case.entity_types, &case.relation_types, &opts)
                .unwrap();
            let mut mapped = 0usize;
            for (i, word) in prepared.words.iter().enumerate() {
                match prepared.candidate_bytes(i, i + 1) {
                    Some(range) => {
                        let surface = prepared.text_map.surface(range);
                        assert!(
                            surface_matches_word(surface, word),
                            "[{id}] word {i} {word:?} sliced {surface:?} from caller input"
                        );
                        mapped += 1;
                    }
                    None => {
                        // Only words reaching into the synthetic suffix may be
                        // rejected.
                        assert!(
                            prepared.text_map.end_mapping(i)
                                > prepared.text_map.caller_text().chars().count(),
                            "[{id}] word {i} {word:?} rejected but lies inside the caller's input"
                        );
                    }
                }
            }
            if case.case_id != "10_empty_text" {
                assert!(mapped > 0, "[{id}] no word mapped into the caller's input");
            }
        }
    }

    /// Epic §1: a candidate whose mapped range extends into the synthetic
    /// `'.'` suffix is rejected in full. Python decodes against the normalized
    /// text and would surface the synthetic `'.'`; Rust must not return a
    /// surface or byte range outside the caller's input.
    #[test]
    fn suffix_crossing_candidates_are_rejected() {
        let (preprocessor, opts) = load_prepared();

        // 05_edges: "Alice greeted Bob" → normalized "Alice greeted Bob."
        let cases = load_oracle_cases();
        let case = cases.iter().find(|c| c.case_id == "05_edges").unwrap();
        let prepared = preprocessor
            .prepare(&case.text, &case.entity_types, &case.relation_types, &opts)
            .unwrap();
        assert_eq!(prepared.words, vec!["alice", "greeted", "bob", "."]);
        assert_eq!(prepared.text_map.suffix_cp_start(), Some(17));
        assert_eq!(
            prepared.candidate_bytes(2, 4),
            None,
            "[2,4) = \"Bob.\" crosses into the synthetic suffix"
        );
        assert_eq!(
            prepared.candidate_bytes(3, 4),
            None,
            "[3,4) = synthetic \".\" lies outside the caller's input"
        );
        assert_eq!(prepared.candidate_bytes(0, 3), Some(0..17));
        assert_eq!(prepared.text_map.surface(0..17), "Alice greeted Bob");

        // 06_no_terminal_punct: controlled pair with 01 — same normalized
        // text, shorter caller input.
        let case = cases
            .iter()
            .find(|c| c.case_id == "06_no_terminal_punct")
            .unwrap();
        let prepared = preprocessor
            .prepare(&case.text, &case.entity_types, &case.relation_types, &opts)
            .unwrap();
        assert_eq!(prepared.text_map.caller_text().chars().count(), 44);
        assert_eq!(prepared.text_map.suffix_cp_start(), Some(44));
        assert_eq!(
            prepared.candidate_bytes(7, 9),
            None,
            "[7,9) = \"cupertino.\" crosses into the synthetic suffix"
        );
        assert_eq!(prepared.candidate_bytes(7, 8), Some(35..44));
        assert_eq!(prepared.text_map.surface(35..44), "Cupertino");

        // 10_empty_text: "" → normalized "." — every candidate is rejected.
        let case = cases.iter().find(|c| c.case_id == "10_empty_text").unwrap();
        let prepared = preprocessor
            .prepare(&case.text, &case.entity_types, &case.relation_types, &opts)
            .unwrap();
        assert_eq!(prepared.text_map.normalized_text(), ".");
        assert_eq!(prepared.candidate_bytes(0, 1), None);
        assert_eq!(prepared.text_map.suffix_cp_start(), Some(0));
    }

    /// A URL word can swallow the synthetic `'.'` into a single split word
    /// (`https?://[^\s]+`); that single-word candidate must also be rejected.
    #[test]
    fn suffix_swallowing_url_word_is_rejected() {
        let (preprocessor, opts) = load_prepared();
        let entities = vec!["url".to_string()];
        let prepared = preprocessor
            .prepare("see https://example.com", &entities, &[], &opts)
            .unwrap();
        assert_eq!(
            prepared.words,
            vec!["see", "https://example.com."],
            "the URL branch keeps the trailing synthetic dot inside one word"
        );
        assert_eq!(prepared.candidate_bytes(1, 2), None);
        assert_eq!(prepared.candidate_bytes(0, 1), Some(0..3));
    }

    /// Unicode: byte offsets must slice the caller's UTF-8 input exactly
    /// (case 08: diacritics + CJK — code points and bytes disagree).
    #[test]
    fn unicode_byte_offsets_slice_exactly() {
        let (preprocessor, opts) = load_prepared();
        let cases = load_oracle_cases();
        let case = cases.iter().find(|c| c.case_id == "08_unicode").unwrap();
        let prepared = preprocessor
            .prepare(&case.text, &case.entity_types, &case.relation_types, &opts)
            .unwrap();
        for (i, word) in prepared.words.iter().enumerate() {
            if let Some(range) = prepared.candidate_bytes(i, i + 1) {
                assert!(surface_matches_word(prepared.text_map.surface(range), word));
            }
        }
        // "renée" is 5 code points but 6 UTF-8 bytes.
        assert_eq!(prepared.word_char_spans[0], (0, 6));
        assert_eq!(prepared.text_map.surface(0..6), "Renée");
    }

    /// Request `max_len` truncates the split words before schema encoding,
    /// matching Python `collate_fn_inference(max_len=...)`.
    #[test]
    fn max_len_truncates_words() {
        let (preprocessor, mut opts) = load_prepared();
        opts.max_len = Some(2);
        let entities = vec!["person".to_string(), "organization".to_string()];
        let prepared = preprocessor
            .prepare(
                "Apple was founded by Steve Jobs in Cupertino.",
                &entities,
                &[],
                &opts,
            )
            .unwrap();
        assert_eq!(prepared.words, vec!["apple", "was"]);
        assert_eq!(prepared.first_subtoken_positions.len(), 2);
        // Schema markers are unaffected by text truncation.
        assert_eq!(prepared.query_marker_positions.len(), 2);
    }
}
