/// Boundary architecture model (epic `gliner-2.5-boundary-support`):
/// config dispatch target, validated boundary-head weights, preprocessing and
/// query routing (Task 4), the Task 5 forward through the boundary encoder
/// (`encoding.py`) + marginal heads (`heads.py`), the epic Task 6 shared
/// candidate pool (`pool.py` builder + scorer) wired into
/// [`BoundaryModel::forward`] after marginals — producing per-query candidate
/// intervals + raw pair logits — the epic Task 7 entity decoder
/// (`src/boundary_decode.rs`: threshold/abstention/adaptive fill/`flat`
/// overlap resolution + exact caller-text slicing), and the epic Task 8
/// relation path (`src/boundary_rel.rs`: typed/capped pair generation +
/// `SparseRelationScorer` + `_decode_relations`). The `per_query` candidate
/// pool is not ported: loading validates its weights but scoring errors
/// clearly.
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::debertav2::{DebertaV2Model, DTYPE as DEBERTA_DTYPE};
use std::path::Path;

use crate::boundary_decode::BoundaryDecoder;
use crate::boundary_enc::{BoundaryEncoder, BoundaryMarginals, BoundaryQueryHead};
use crate::boundary_pool::{
    CandidatePoolOutput, DocumentCandidatePool, ScoreParts, SharedPoolScorer,
};
use crate::boundary_pre::{
    BoundaryPrepared, BoundaryPreprocessor, PrepareOptions, RelationQueryState,
};
use crate::boundary_rel::{
    decode_relations_from_pairs, generate_relation_pairs, ExtractedRelation, RelationSettings,
    RelationTypeSpec, SparseRelationScorer,
};
use crate::config::{encoder_config_from_file, hidden_size, BoundaryHeadConfig, Gliner2Config};
use crate::inference::ExtractedEntity;

/// Gathered encoder states — the `BoundaryExtractorModel._encode_core` gather
/// stage (first-subtoken text states, `[E]`/`[R]` marker query states).
#[derive(Clone)]
#[allow(dead_code)] // text_lengths is consumed by epic Task 6's pool scorer
pub struct EncodedStates {
    /// `[1, L, H]` — first-subtoken states per text word.
    pub text_states: Tensor,
    /// `[1, L]` u8 (all valid at B=1).
    pub text_mask: Tensor,
    /// Words per sample (B=1 today).
    pub text_lengths: Vec<usize>,
    /// `[1, Q, H]` — one query state per extractive schema marker.
    pub query_states: Tensor,
    /// `[1, Q]` u8 (all valid at B=1).
    pub query_mask: Tensor,
}

/// Everything epic Task 5 produces. The Task 6 shared-pool builder/scorer
/// (`pool.py`) consumes exactly these tensors: boundary states/mask and
/// start/end/inside marginals (+ prefix/mean for inside interval evidence),
/// with the text/query states for content pooling and query projections.
#[allow(dead_code)] // fields are consumed by epic Task 6 (and the tests below)
pub struct BoundaryOutputs {
    pub states: EncodedStates,
    /// `[1, L+1, d]` — boundary states (padding boundaries zeroed).
    pub boundary_states: Tensor,
    /// `[1, L+1]` u8 — boundary `i` valid iff `i <= L` at B=1.
    pub boundary_mask: Tensor,
    pub marginals: BoundaryMarginals,
}

/// Epic Task 6 result: the Task 5 marginals plus the shared candidate pool
/// rows and per-query pair logits (raw, pre `pair_temperature`), and the Task
/// 7 decode heads (`null_projection` / `count_head` over the query states).
#[allow(dead_code)] // fields are consumed by the epic Task 7+ decode paths
pub struct BoundaryForward {
    pub head: BoundaryOutputs,
    pub candidates: CandidatePoolOutput,
    /// `[Q]` raw abstention null logits (`boundary_head.null_projection` on
    /// the query states) — `None` when `enable_abstention` is off.
    pub null_logits: Option<Vec<f32>>,
    /// `[Q]` raw count log rates (`boundary_head.count_head` on the query
    /// states) — `None` when `enable_count_head` is off.
    pub count_log_rates: Option<Vec<f32>>,
}

pub struct BoundaryModel {
    encoder: DebertaV2Model,
    /// `boundary_head.boundary_encoder` (encoding.py), epic Task 5.
    boundary_encoder: BoundaryEncoder,
    /// `boundary_head.boundary_query_head` (heads.py), epic Task 5.
    query_head: BoundaryQueryHead,
    /// `boundary_head.shared_pool_builder` + `shared_pool_scorer` (pool.py),
    /// epic Task 6 — `Some` iff `candidate_pool == "shared"`.
    shared_pool: Option<(DocumentCandidatePool, SharedPoolScorer)>,
    /// `boundary_head.null_projection` (model.py) — abstention null logit,
    /// epic Task 7 — `Some` iff `enable_abstention`.
    null_projection: Option<Linear>,
    /// `boundary_head.count_head` (model.py) — count log rate for
    /// `adaptive_threshold` decoding, epic Task 7 — `Some` iff
    /// `enable_count_head`.
    count_head: Option<Linear>,
    /// `relation_scorer.*` (relations.py `SparseRelationScorer`), epic Task 8
    /// — `Some` iff `enable_relations`.
    relation_scorer: Option<SparseRelationScorer>,
    /// Full migrated config; boundary math reads `boundary_head`.
    pub gliner_cfg: Gliner2Config,
    /// Boundary-path preprocessing and query routing (epic Task 4).
    preprocessor: BoundaryPreprocessor,
    #[allow(dead_code)]
    hidden_size: usize,
    device: Device,
}

impl BoundaryModel {
    /// Load a `architecture == "boundary"` checkpoint: encoder plus validated
    /// boundary-head weights and the Task 5 encoding/marginal modules.
    pub fn load(dir: &Path, gliner_cfg: Gliner2Config, device: &Device) -> Result<Self> {
        let enc_cfg = encoder_config_from_file(dir.join("encoder_config").join("config.json"))
            .context("encoder_config/config.json not found — see README")?;
        let h = hidden_size(&enc_cfg);

        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(anyhow!(
                "model.safetensors not found in {:?}. \
                 Download with: huggingface-cli download fastino/gliner2.5-small-v1 \
                 --local-dir {:?}",
                dir,
                dir
            ));
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DEBERTA_DTYPE, device)?
        };

        // Validate the active boundary head (and relation scorer when enabled)
        // BEFORE loading the large encoder: a wrong checkpoint fails fast.
        let n = validate_boundary_weights(&vb, &gliner_cfg.boundary_head, h)?;
        tracing::debug!(
            "validated {n} boundary/relation weight tensors for candidate_pool={:?}",
            gliner_cfg.boundary_head.candidate_pool
        );

        let boundary_encoder = BoundaryEncoder::load(
            vb.pp("boundary_head").pp("boundary_encoder"),
            &gliner_cfg.boundary_head,
            h,
        )?;
        let query_head = BoundaryQueryHead::load(
            vb.pp("boundary_head").pp("boundary_query_head"),
            &gliner_cfg.boundary_head,
            h,
        )?;

        // Epic Task 6: the shared pool builder + scorer. Loaded only for the
        // checkpoint-selected `candidate_pool == "shared"` path; the per_query
        // modules are inactive here and never constructed (pool.py vs
        // proposal.py/scoring.py — see BoundaryModel::score_candidates).
        let shared_pool = match gliner_cfg.boundary_head.candidate_pool.as_str() {
            "shared" => {
                let pool = DocumentCandidatePool::load(
                    vb.pp("boundary_head").pp("shared_pool_builder"),
                    &gliner_cfg.boundary_head,
                )?;
                let scorer = SharedPoolScorer::load(
                    vb.pp("boundary_head").pp("shared_pool_scorer"),
                    &gliner_cfg.boundary_head,
                    h,
                )?;
                Some((pool, scorer))
            }
            _ => None,
        };

        let encoder = DebertaV2Model::load(vb.pp("encoder"), &enc_cfg)?;

        // Task 7 decode heads: `nn.Linear(hidden, 1)` over the query states
        // (model.py `BoundaryHead.null_projection` / `count_head`), present
        // only when the corresponding flag is enabled.
        let null_projection = if gliner_cfg.boundary_head.enable_abstention {
            Some(linear(h, 1, vb.pp("boundary_head").pp("null_projection"))?)
        } else {
            None
        };
        let count_head = if gliner_cfg.boundary_head.enable_count_head {
            Some(linear(h, 1, vb.pp("boundary_head").pp("count_head"))?)
        } else {
            None
        };

        // Epic Task 8: relation scorer (`SparseRelationScorer`), loaded only
        // when relations are enabled. `expected_boundary_weights` has already
        // pinned its tensors/shapes.
        let relation_scorer = if gliner_cfg.boundary_head.enable_relations {
            Some(SparseRelationScorer::load(
                vb.pp("relation_scorer"),
                &gliner_cfg.boundary_head,
                h,
            )?)
        } else {
            None
        };

        let preprocessor = BoundaryPreprocessor::from_file(dir.join("tokenizer.json"))
            .context("tokenizer.json not found — see DEVELOP.md checkpoint file layout")?;

        Ok(Self {
            encoder,
            boundary_encoder,
            query_head,
            shared_pool,
            null_projection,
            count_head,
            relation_scorer,
            gliner_cfg,
            preprocessor,
            hidden_size: h,
            device: device.clone(),
        })
    }

    /// Boundary-path preprocessing and query routing (epic Task 4): schema +
    /// `[SEP_TEXT]` + normalized-text input, marker/word routing, query layout,
    /// relation role queries, and the caller-text offset map for one request.
    ///
    /// `max_len` is the request-level **word** cap (Python
    /// `collate_fn_inference(max_len=...)`; `None` = no truncation) — the
    /// config's `max_len` is not applied at inference, matching Python's
    /// `extract()`. `token_pooling` must be `"first"` (config load already
    /// rejects other modes; re-checked here so a future caller cannot bypass).
    #[allow(dead_code)] // consumed by the Task 6+ predict path (tests use it now)
    pub fn preprocess(
        &self,
        text: &str,
        entity_types: &[String],
        relation_types: &[String],
        max_len: Option<usize>,
    ) -> Result<BoundaryPrepared> {
        if self.gliner_cfg.token_pooling != "first" {
            return Err(anyhow!(
                "unsupported token_pooling {:?}: boundary preprocessing only implements \
                 first-subtoken pooling",
                self.gliner_cfg.token_pooling
            ));
        }
        self.preprocessor.prepare(
            text,
            entity_types,
            relation_types,
            &PrepareOptions {
                max_len,
                directional_relation_states: self
                    .gliner_cfg
                    .boundary_head
                    .directional_relation_states,
                hidden_size: self.hidden_size,
            },
        )
    }

    /// DeBERTa forward + first-subtoken / query-marker gathers (Python
    /// `BoundaryExtractorModel._encode_core` gather stage). Routing positions
    /// come from the Task 4 preprocessor; `token_pooling == "first"` is
    /// required (enforced by [`Self::preprocess`]).
    pub fn encode(&self, prepared: &BoundaryPrepared) -> Result<EncodedStates> {
        let seq_len = prepared.input_ids.len();
        let device = &self.device;
        let ids = Tensor::from_vec(prepared.input_ids.clone(), (1, seq_len), device)?;
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, device)?;
        let token_embs = self.encoder.forward(&ids, None, Some(attention_mask))?; // [1,S,H]

        let gather = |positions: &[usize]| -> Result<Tensor> {
            let idx = Tensor::from_vec(
                positions.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                (positions.len(),),
                device,
            )?;
            // Python `gather_routed` zeroes masked positions; at B=1 every
            // routed position is valid, so the index_select output is exact.
            Ok(token_embs.index_select(&idx, 1)?)
        };
        let text_states = gather(&prepared.first_subtoken_positions)?; // [1,L,H]
        let query_states = gather(&prepared.query_marker_positions)?; // [1,Q,H]
        let l = prepared.first_subtoken_positions.len();
        let q = prepared.query_marker_positions.len();
        Ok(EncodedStates {
            text_states,
            text_mask: Tensor::ones((1, l), DType::U8, device)?,
            text_lengths: vec![l],
            query_states,
            query_mask: Tensor::ones((1, q), DType::U8, device)?,
        })
    }

    /// `BoundaryHead.forward`'s encoding + marginals stage (epic Task 5 stop
    /// line): boundary encoder over the gathered text states, then the
    /// start/end/inside marginal heads. Proposal/pool scoring (Task 6) is
    /// deliberately not run here.
    pub fn forward_head(&self, states: &EncodedStates) -> Result<BoundaryOutputs> {
        let encoding = self
            .boundary_encoder
            .forward(&states.text_states, &states.text_mask)?;
        let marginals = self.query_head.forward(
            &encoding.states,
            &encoding.mask,
            &states.text_states,
            &states.text_mask,
            &states.query_states,
            &states.query_mask,
        )?;
        Ok(BoundaryOutputs {
            states: states.clone(),
            boundary_states: encoding.states,
            boundary_mask: encoding.mask,
            marginals,
        })
    }

    /// Full forward: preprocessing output → encoder/gather → boundary
    /// encoding → marginals (Task 5) → shared candidate pool + pair scoring
    /// (Task 6) → null/count decode heads (Task 7).
    #[allow(dead_code)] // consumed by the Task 7+ predict path (tests use it now)
    pub fn forward(&self, prepared: &BoundaryPrepared) -> Result<BoundaryForward> {
        let states = self.encode(prepared)?;
        let head = self.forward_head(&states)?;
        let candidates = self.score_candidates(&head)?;
        Ok(BoundaryForward {
            head,
            candidates,
            null_logits: self.query_decode_logits(&states.query_states, &self.null_projection)?,
            count_log_rates: self.query_decode_logits(&states.query_states, &self.count_head)?,
        })
    }

    /// `null_projection`/`count_head` on the query states — model.py
    /// `BoundaryHead.forward`'s `null_logits`/`count_log_rates` (`[1, Q, 1]`
    /// → `[Q]` raw values; sigmoid/rounding happen at decode, epic Task 7).
    fn query_decode_logits(
        &self,
        query_states: &Tensor,
        head: &Option<Linear>,
    ) -> Result<Option<Vec<f32>>> {
        let Some(head) = head else {
            return Ok(None);
        };
        let logits = head.forward(query_states)?.squeeze(2)?; // [1, Q]
        Ok(Some(logits.squeeze(0)?.to_vec1::<f32>()?))
    }

    /// Epic Task 7: decode entities from the Task 6 pool rows + pair logits
    /// via `src/boundary_decode.rs` (per-query threshold/abstention/`flat`
    /// overlap resolution, exact caller-text slicing). `threshold` is the
    /// CLI's single score threshold (Python's per-type schema thresholds are
    /// not exposed by this API).
    #[allow(dead_code)] // consumed by Model::predict_entities and the tests
    pub fn decode_entities(
        &self,
        prepared: &BoundaryPrepared,
        forward: &BoundaryForward,
        threshold: f32,
    ) -> Result<Vec<ExtractedEntity>> {
        let decoder = BoundaryDecoder::new(&self.gliner_cfg.boundary_head);
        let indices3 = forward.candidates.indices.to_vec3::<u32>()?; // [1, C, 2]
        let valid2 = forward.candidates.valid_mask.to_vec2::<u8>()?; // [1, C]
        let pair3 = forward.candidates.pair_logits.to_vec3::<f32>()?; // [1, Q, C]
        let pool_indices: Vec<(u32, u32)> =
            indices3[0].iter().map(|row| (row[0], row[1])).collect();
        let pool_valid: Vec<bool> = valid2[0].iter().map(|&v| v != 0).collect();
        let pair_logits: Vec<Vec<f32>> = pair3[0].to_vec();
        decoder.decode_entities(
            &prepared.text_map,
            &prepared.query_layout,
            &pool_indices,
            &pool_valid,
            &pair_logits,
            forward.null_logits.as_deref(),
            forward.count_log_rates.as_deref(),
            threshold,
        )
    }

    /// High-level entity extraction for the boundary path (epic Task 7):
    /// preprocess (entity schema only — relation requests go through
    /// [`Self::predict_relations`]) → forward → decode. Returns
    /// `ExtractedEntity` values with byte offsets into the caller's exact
    /// input, ordered like Python's `final_output` (declared type order, then
    /// descending confidence within a type).
    #[allow(dead_code)] // consumed by Model::predict_entities
    pub fn predict_entities(
        &self,
        text: &str,
        entity_types: &[String],
        threshold: f32,
    ) -> Result<Vec<ExtractedEntity>> {
        let prepared = self.preprocess(text, entity_types, &[], None)?;
        if !prepared
            .query_layout
            .iter()
            .any(|q| q.task_type == "entities")
        {
            return Ok(Vec::new());
        }
        let forward = self.forward(&prepared)?;
        self.decode_entities(&prepared, &forward, threshold)
    }

    /// Epic Task 8 support gate: relations need `enable_relations`, the loaded
    /// relation scorer, and the shared candidate pool (the pair generator
    /// consumes its query-agnostic rows + per-role-query pair logits; the
    /// `per_query` proposal/scoring math is not ported). Fails clearly instead
    /// of returning an empty result that would look like a valid negative
    /// prediction. `pub(crate)`: also used by the CLI pre-flight
    /// (`Model::check_relation_support`, epic Task 9).
    pub(crate) fn check_relation_support(&self) -> Result<()> {
        if !self.gliner_cfg.boundary_head.enable_relations || self.relation_scorer.is_none() {
            return Err(anyhow!(
                "relation extraction is not available on this checkpoint: \
                 boundary_head.enable_relations is false or the relation_scorer weights are \
                 absent — refusing to return an empty result that would look like a valid \
                 negative prediction (see DEVELOP.md \"Boundary relation extraction\")"
            ));
        }
        if self.shared_pool.is_none() {
            return Err(anyhow!(
                "relation extraction is only implemented for candidate_pool=\"shared\"; \
                 this checkpoint selects {:?} (the per_query proposal/scoring math of \
                 proposal.py/scoring.py is not ported)",
                self.gliner_cfg.boundary_head.candidate_pool
            ));
        }
        Ok(())
    }

    /// Epic Task 8 intermediates for one prepared request: the typed/capped
    /// proposed argument pairs (`relations.py::TypedRelationPairGenerator`)
    /// and the raw `SparseRelationScorer` logits (pre `relation_temperature`).
    #[allow(dead_code)] // consumed by decode_relations and the Task 8 tests
    pub fn propose_relations(
        &self,
        prepared: &BoundaryPrepared,
        forward: &BoundaryForward,
    ) -> Result<(Vec<crate::boundary_rel::RelationPair>, Vec<f32>)> {
        self.check_relation_support()?;
        let routes = &prepared.relation_role_routing;
        if routes.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let specs: Vec<RelationTypeSpec> = routes
            .iter()
            .map(|route| RelationTypeSpec {
                relation_type: route.relation_type.clone(),
                head_query_ids: route.head_query_id.clone(),
                tail_query_ids: route.tail_query_id.clone(),
                // Python `RelationTypeSpec` default; every schema-built spec.
                allow_self: false,
            })
            .collect();

        let indices3 = forward.candidates.indices.to_vec3::<u32>()?; // [1, C, 2]
        let valid2 = forward.candidates.valid_mask.to_vec2::<u8>()?; // [1, C]
        let pair3 = forward.candidates.pair_logits.to_vec3::<f32>()?; // [1, Q, C]
        let pool_indices: Vec<(u32, u32)> =
            indices3[0].iter().map(|row| (row[0], row[1])).collect();
        let pool_valid: Vec<bool> = valid2[0].iter().map(|&v| v != 0).collect();
        let pair_logits: Vec<Vec<f32>> = pair3[0].to_vec();

        let settings = RelationSettings::from_config(&self.gliner_cfg.boundary_head);
        let pairs =
            generate_relation_pairs(&pool_indices, &pool_valid, &pair_logits, &specs, &settings)?;
        if pairs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Relation query states: `concat(head, tail)` role states when
        // `directional_relation_states`, else their mean (Python `_encode_core`).
        let query_rows = forward.head.states.query_states.to_vec3::<f32>()?; // [1, Q, H]
        let n_queries = query_rows[0].len();
        let mut rel_rows: Vec<f32> = Vec::with_capacity(routes.len() * 2 * self.hidden_size);
        let mut rel_dim = 0usize;
        for route in routes {
            let head_id = route.head_query_id.first().copied().ok_or_else(|| {
                anyhow!("relation {:?} has no head role query", route.relation_type)
            })?;
            let tail_id = route.tail_query_id.first().copied().ok_or_else(|| {
                anyhow!("relation {:?} has no tail role query", route.relation_type)
            })?;
            if head_id >= n_queries || tail_id >= n_queries {
                return Err(anyhow!(
                    "relation {:?} role queries ({head_id}, {tail_id}) exceed {} queries",
                    route.relation_type,
                    n_queries
                ));
            }
            let head = &query_rows[0][head_id];
            let tail = &query_rows[0][tail_id];
            match route.query_state {
                RelationQueryState::Concat => {
                    rel_rows.extend_from_slice(head);
                    rel_rows.extend_from_slice(tail);
                    rel_dim = 2 * self.hidden_size;
                }
                RelationQueryState::Mean => {
                    rel_rows.extend(
                        head.iter()
                            .zip(tail)
                            .map(|(a, b)| 0.5 * (a + b))
                            .collect::<Vec<f32>>(),
                    );
                    rel_dim = self.hidden_size;
                }
            }
        }
        let rel_states = Tensor::from_vec(rel_rows, (1, routes.len(), rel_dim), &self.device)?;
        let scorer = self.relation_scorer.as_ref().expect("checked above");
        let logits = scorer.forward(&forward.head.states.text_states, &rel_states, &pairs)?;
        Ok((pairs, logits))
    }

    /// Epic Task 8 decode for one prepared request: typed/capped proposal +
    /// relation scoring + `engine.py::_decode_relations` (temperature →
    /// sigmoid → threshold → caller-text mapping → edge dedup). Relation
    /// arguments are never filtered through the entity decoder.
    #[allow(dead_code)] // consumed by Model::predict_relations and the tests
    pub fn decode_relations(
        &self,
        prepared: &BoundaryPrepared,
        forward: &BoundaryForward,
        threshold: f32,
    ) -> Result<Vec<ExtractedRelation>> {
        let (pairs, logits) = self.propose_relations(prepared, forward)?;
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let routes = &prepared.relation_role_routing;
        let specs: Vec<RelationTypeSpec> = routes
            .iter()
            .map(|route| RelationTypeSpec {
                relation_type: route.relation_type.clone(),
                head_query_ids: route.head_query_id.clone(),
                tail_query_ids: route.tail_query_id.clone(),
                allow_self: false,
            })
            .collect();
        let settings = RelationSettings::from_config(&self.gliner_cfg.boundary_head);
        decode_relations_from_pairs(
            &prepared.text_map,
            &specs,
            &pairs,
            &logits,
            settings.relation_temperature,
            threshold,
        )
    }

    /// High-level relation extraction (epic Task 8): preprocess (relation
    /// schema, optionally with the request's entity schema since it changes
    /// the encoding) → forward → decode. Relation-only requests need no
    /// entity schema. Returns directed `ExtractedRelation` values with byte
    /// offsets into the caller's exact input.
    ///
    /// Fails clearly when the checkpoint cannot extract relations (see
    /// [`Self::check_relation_support`]) instead of returning an empty result
    /// that would look like a valid negative prediction.
    #[allow(dead_code)] // consumed by Model::predict_relations
    pub fn predict_relations(
        &self,
        text: &str,
        entity_types: &[String],
        relation_types: &[String],
        threshold: f32,
    ) -> Result<Vec<ExtractedRelation>> {
        self.check_relation_support()?;
        if relation_types.is_empty() {
            return Err(anyhow!(
                "relation extraction requires at least one relation type; refusing to return \
                 an empty result that would look like a valid negative prediction"
            ));
        }
        let prepared = self.preprocess(text, entity_types, relation_types, None)?;
        let forward = self.forward(&prepared)?;
        self.decode_relations(&prepared, &forward, threshold)
    }

    /// Combined extraction (epic Task 9): **one** forward over the request's
    /// full schema (entity and relation groups together — Python's single
    /// `extract()` encoding) with entities and relations decoded from the
    /// same outputs, so combined results match the Task 7/8 oracle exactly
    /// instead of encoding entities and relations separately. Entity-only and
    /// relation-only requests go through here too: with an empty relation
    /// list this is `predict_entities`, with an empty entity list it is
    /// `predict_relations` (minus the empty-list error — the CLI validates
    /// the request). This is the single production inference path behind the
    /// CLI's one-off, JSON, and case-file modes.
    pub fn predict_extractions(
        &self,
        text: &str,
        entity_types: &[String],
        relation_types: &[String],
        threshold: f32,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>)> {
        if !relation_types.is_empty() {
            self.check_relation_support()?;
        }
        let prepared = self.preprocess(text, entity_types, relation_types, None)?;
        let forward = self.forward(&prepared)?;
        let entities = if prepared
            .query_layout
            .iter()
            .any(|q| q.task_type == "entities")
        {
            self.decode_entities(&prepared, &forward, threshold)?
        } else {
            Vec::new()
        };
        let relations = if relation_types.is_empty() {
            Vec::new()
        } else {
            self.decode_relations(&prepared, &forward, threshold)?
        };
        Ok((entities, relations))
    }

    /// Epic Task 6: run the checkpoint-selected candidate-pool path over the
    /// Task 5 marginals — `pool.py` `DocumentCandidatePool` +
    /// `SharedPoolScorer` — producing query-agnostic pool rows and per-query
    /// pair logits (raw, pre `pair_temperature`). `boundary_proposer` /
    /// `pair_scorer` (the `per_query` path) are never run for a `shared`
    /// checkpoint, and a `per_query` checkpoint errors clearly here.
    #[allow(dead_code)] // consumed by the Task 7+ predict path (tests use it now)
    pub fn score_candidates(&self, head: &BoundaryOutputs) -> Result<CandidatePoolOutput> {
        Ok(self.score_candidates_parts(head)?.0)
    }

    /// [`Self::score_candidates`] keeping the scorer's additive decomposition
    /// ([`ScoreParts`]) — used by the epic "marginals are added once" check.
    pub(crate) fn score_candidates_parts(
        &self,
        head: &BoundaryOutputs,
    ) -> Result<(CandidatePoolOutput, ScoreParts)> {
        let (pool, scorer) = self.shared_pool.as_ref().ok_or_else(|| {
            anyhow!(
                "boundary candidate scoring is only implemented for candidate_pool=\"shared\"; \
                 this checkpoint selects {:?} (the per_query proposer/pair_scorer math of \
                 proposal.py/scoring.py is not ported)",
                self.gliner_cfg.boundary_head.candidate_pool
            )
        })?;
        let pooled = pool.forward(
            &head.boundary_states,
            &head.boundary_mask,
            &head.states.query_mask,
            &head.marginals.start_logits,
            &head.marginals.end_logits,
        )?;
        // Python `BoundaryHead.forward`: `inside_prefix` is passed only when
        // `use_inside_evidence` (else the scorer skips the inside term).
        let (inside_prefix, inside_mean) = if self.gliner_cfg.boundary_head.use_inside_evidence {
            (
                Some(&head.marginals.inside_prefix),
                Some(&head.marginals.inside_prefix_mean),
            )
        } else {
            (None, None)
        };
        let parts = scorer.forward(
            &pooled,
            &head.boundary_states,
            &head.states.query_states,
            &head.states.query_mask,
            &head.marginals.start_logits,
            &head.marginals.end_logits,
            inside_prefix,
            inside_mean,
            &head.states.text_states,
            &head.states.text_mask,
            head.states.text_lengths[0],
        )?;
        // Python `BoundaryHead.forward` shared path: `pooled_logits.transpose(1, 2)`
        // -> per-query `[B, Q, C]` contract (`PooledCandidates.to_candidate_batch`).
        let candidates = CandidatePoolOutput {
            indices: pooled.indices.clone(),
            valid_mask: pooled.mask.clone(),
            proposal_logits: pooled.proposal_logits.clone(),
            compat_logits: pooled.compat_logits.clone(),
            pair_logits: parts.pair_logits.transpose(1, 2)?, // [1, C, Q] -> [1, Q, C]
        };
        Ok((candidates, parts))
    }
}

/// Expected `(tensor name, shape)` for every weight tensor the *active*
/// boundary path reads at inference. Derived from the pinned Python modules
/// (encoding.py / heads.py / pool.py / scoring.py / relations.py) and verified
/// against the `fastino/gliner2.5-small-v1` inventory in DEVELOP.md.
///
/// Inactive modules are deliberately absent and their weights optional:
/// - `boundary_proposer.*` / `pair_scorer.*` when `candidate_pool == "shared"`
/// - `shared_pool_builder.*` / `shared_pool_scorer.*` when `candidate_pool == "per_query"`
/// - `candidate_encoder.*` (records-only) and `record_decoder.*` / `classifier.*`
///   (out of epic scope) are never required.
pub fn expected_boundary_weights(
    cfg: &BoundaryHeadConfig,
    hidden: usize,
) -> Vec<(String, Vec<usize>)> {
    let h = hidden;
    let b = cfg.boundary_dim;
    let p = cfg.pair_dim;
    let c = cfg.content_dim;
    // SpanContentPooler.output_dim = content_dim * (2 if soft-max pool else 1)
    let c_out = if cfg.content_soft_max_pool { c * 2 } else { c };
    // ResidualSwiGLU: hidden_dim = max(1, int(dim * multiplier))
    let ffn = ((b as f64 * cfg.boundary_ffn_multiplier) as usize).max(1);

    let mut spec: Vec<(String, Vec<usize>)> = Vec::new();
    let linear = |spec: &mut Vec<(String, Vec<usize>)>, prefix: &str, out: usize, inn: usize| {
        spec.push((format!("{prefix}.weight"), vec![out, inn]));
        spec.push((format!("{prefix}.bias"), vec![out]));
    };
    let norm1d = |spec: &mut Vec<(String, Vec<usize>)>, prefix: &str, d: usize| {
        spec.push((format!("{prefix}.weight"), vec![d]));
        spec.push((format!("{prefix}.bias"), vec![d]));
    };

    // ── boundary_head.boundary_encoder (encoding.py BoundaryEncoder) ───────
    let enc = "boundary_head.boundary_encoder";
    linear(&mut spec, &format!("{enc}.left_projection"), b, h);
    linear(&mut spec, &format!("{enc}.right_projection"), b, h);
    linear(&mut spec, &format!("{enc}.output_projection"), b, 2 * b);
    norm1d(&mut spec, &format!("{enc}.layer_norm"), b);
    spec.push((format!("{enc}.bos_state"), vec![h]));
    spec.push((format!("{enc}.eos_state"), vec![h]));
    for i in 0..cfg.boundary_attention_layers {
        let blk = format!("{enc}.attention_blocks.{i}");
        norm1d(&mut spec, &format!("{blk}.norm"), b);
        linear(&mut spec, &format!("{blk}.qkv_projection"), 3 * b, b);
        linear(&mut spec, &format!("{blk}.output_projection"), b, b);
    }
    for i in 0..cfg.boundary_refinement_layers {
        let blk = format!("{enc}.refinement_blocks.{i}");
        norm1d(&mut spec, &format!("{blk}.norm"), b);
        linear(&mut spec, &format!("{blk}.input_projection"), 2 * ffn, b);
        linear(&mut spec, &format!("{blk}.output_projection"), b, ffn);
    }

    // ── boundary_head.boundary_query_head (heads.py) — query dim = hidden ──
    let qh = "boundary_head.boundary_query_head";
    linear(&mut spec, &format!("{qh}.start_boundary_projection"), b, b);
    linear(&mut spec, &format!("{qh}.start_query_projection"), b, h);
    linear(&mut spec, &format!("{qh}.end_boundary_projection"), b, b);
    linear(&mut spec, &format!("{qh}.end_query_projection"), b, h);
    linear(&mut spec, &format!("{qh}.inside_text_projection"), b, h);
    linear(&mut spec, &format!("{qh}.inside_query_projection"), b, h);

    // ── Abstention / count heads (model.py BoundaryHead) ────────────────────
    if cfg.enable_abstention {
        linear(&mut spec, "boundary_head.null_projection", 1, h);
    }
    if cfg.enable_count_head {
        linear(&mut spec, "boundary_head.count_head", 1, h);
    }

    match cfg.candidate_pool.as_str() {
        // ── Shared pool (pool.py DocumentCandidatePool + SharedPoolScorer) ──
        "shared" => {
            let bld = "boundary_head.shared_pool_builder";
            linear(&mut spec, &format!("{bld}.start_projection"), b, b);
            linear(&mut spec, &format!("{bld}.end_projection"), b, b);

            let sc = "boundary_head.shared_pool_scorer";
            linear(&mut spec, &format!("{sc}.start_projection"), p, b);
            linear(&mut spec, &format!("{sc}.end_projection"), p, b);
            linear(&mut spec, &format!("{sc}.length_projection"), p, 3);
            linear(&mut spec, &format!("{sc}.prior_projection"), p, 1);
            if cfg.enable_span_content {
                linear(
                    &mut spec,
                    &format!("{sc}.content_pooler.value_projection"),
                    c,
                    h,
                );
                norm1d(&mut spec, &format!("{sc}.content_pooler.layer_norm"), c_out);
                linear(&mut spec, &format!("{sc}.content_projection"), p, c_out);
            }
            norm1d(&mut spec, &format!("{sc}.candidate_norm"), p);
            linear(&mut spec, &format!("{sc}.query_projection"), p, h);
            linear(&mut spec, &format!("{sc}.film"), 2 * p, p);
            // film_output Sequential(Linear(pair_dim, 64), GELU, Dropout, Linear(64, 1)):
            // PyTorch indices 0 and 3.
            linear(&mut spec, &format!("{sc}.film_output.0"), 64, p);
            linear(&mut spec, &format!("{sc}.film_output.3"), 1, 64);
            // candidate_layers/query_layers are gated by check_supported()
            // (candidate/query attention > 0 layers is rejected while unported).
        }
        // ── Per-query pool (proposal.py + scoring.py) ───────────────────────
        "per_query" => {
            let prop = "boundary_head.boundary_proposer";
            linear(&mut spec, &format!("{prop}.start_pair_projection"), b, b);
            linear(&mut spec, &format!("{prop}.end_key_projection"), b, b);
            // Rotary halves the projection width (boundary states are rotated
            // as interleaved even/odd pairs).
            let b_half = if cfg.enable_rotary_endpoints {
                b / 2
            } else {
                b
            };
            linear(
                &mut spec,
                &format!("{prop}.start_query_projection"),
                b_half,
                h,
            );

            let ps = "boundary_head.pair_scorer";
            linear(&mut spec, &format!("{ps}.start_endpoint_projection"), p, b);
            linear(&mut spec, &format!("{ps}.end_endpoint_projection"), p, b);
            let p_half = if cfg.enable_rotary_endpoints {
                p / 2
            } else {
                p
            };
            linear(&mut spec, &format!("{ps}.query_gate"), p_half, h);
            linear(&mut spec, &format!("{ps}.length_query_projection"), 3, h);
            if cfg.query_conditioned_inside_weight {
                linear(&mut spec, &format!("{ps}.inside_weight"), 1, h);
            } else {
                // nn.Parameter(torch.tensor(1.0)) — scalar, no bias.
                spec.push((format!("{ps}.inside_weight"), Vec::new()));
            }
            if cfg.endpoint_difference_features {
                linear(
                    &mut spec,
                    &format!("{ps}.endpoint_difference_projection"),
                    1,
                    2 * p,
                );
            }
            linear(
                &mut spec,
                &format!("{ps}.compat_mix"),
                1,
                cfg.multihead_pair_compat_heads,
            );
            if cfg.enable_span_content {
                // Pair scorer uses content_hidden_size = hidden.
                linear(
                    &mut spec,
                    &format!("{ps}.content_pooler.value_projection"),
                    c,
                    h,
                );
                norm1d(&mut spec, &format!("{ps}.content_pooler.layer_norm"), c_out);
                linear(
                    &mut spec,
                    &format!("{ps}.content_query_projection"),
                    c_out,
                    h,
                );
                linear(&mut spec, &format!("{ps}.content_bias"), 1, c_out);
            }
        }
        other => unreachable!("candidate_pool validated to be per_query|shared, got {other:?}"),
    }

    // ── relation_scorer (relations.py SparseRelationScorer) ────────────────
    // Active only when relations are enabled; the relation query is the
    // concat of head/tail role states (2h) when directional, else mean (h).
    if cfg.enable_relations {
        let rq = if cfg.directional_relation_states {
            2 * h
        } else {
            h
        };
        let in_dim = 4 * h + rq + 2;
        // mlp Sequential(Linear, GELU, Dropout, Linear): indices 0 and 3.
        linear(&mut spec, "relation_scorer.mlp.0", h, in_dim);
        linear(&mut spec, "relation_scorer.mlp.3", 1, h);
        if cfg.relation_biaffine_content {
            linear(&mut spec, "relation_scorer.head_content_projection", h, h);
            linear(&mut spec, "relation_scorer.tail_content_projection", h, h);
            linear(&mut spec, "relation_scorer.relation_content_gate", h, rq);
            linear(&mut spec, "relation_scorer.content_linear", 1, 2 * h + rq);
        }
    }

    spec
}

/// Load every tensor in `expected_boundary_weights` through the VarBuilder:
/// this both proves the weight exists and pins its exact shape. Inactive
/// modules' weights are not touched (they are optional).
pub fn validate_boundary_weights(
    vb: &VarBuilder,
    cfg: &BoundaryHeadConfig,
    hidden: usize,
) -> Result<usize> {
    let spec = expected_boundary_weights(cfg, hidden);
    for (name, shape) in &spec {
        vb.get(shape.clone(), name).with_context(|| {
            format!(
                "boundary checkpoint weight {name:?} is missing or does not have shape {shape:?} \
                 (required by the active config path; see DEVELOP.md \"Checkpoint inventory\" \
                 and epics/gliner-2.5-boundary-support.md)"
            )
        })?;
    }
    Ok(spec.len())
}

#[cfg(test)]
mod tests {
    //! Epic Task 5 acceptance: the Task 5 forward (`encode`, boundary encoder,
    //! marginal heads) is compared against the Python oracle's raw dumps for
    //! `01_apple_readme`, `05_edges`, and `10_empty_text` — boundary states
    //! (including boundary 0/BOS and L/EOS), masks, start/end/inside marginals,
    //! the fp32 centered inside prefix with mean, and interval reconstruction.
    //! Tolerance: fp32 CPU vs Python fp32, assert `< 1e-4` abs; observed max
    //! diffs are printed (typically ~1e-6; see DEVELOP.md).
    //!
    //! Epic Task 6 acceptance: the shared-pool candidates (indices, valid mask,
    //! proposal/compat priors, per-query pair logits) are compared against the
    //! oracle `candidate_*` fields over all 18 cases — indices/mask exactly,
    //! logits `< 1e-4` abs — plus the marginals-added-once decomposition.
    use super::*;
    use crate::boundary_enc::MASK_LOGIT;
    use std::path::PathBuf;

    const MODEL_DIR: &str = "models/gliner2.5-small-v1";
    const TOL: f32 = 1e-4;

    fn repo_path(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    #[derive(Debug, serde::Deserialize)]
    struct RawBoundary {
        boundary_states: Vec<Vec<f32>>,
        boundary_mask: Vec<u8>,
        start_logits: Vec<Vec<f32>>,
        end_logits: Vec<Vec<f32>>,
        inside_logits: Vec<Vec<f32>>,
        inside_prefix: Vec<Vec<f32>>,
        inside_prefix_mean: Vec<f32>,
        /// Rows `[batch, query, start, end, value]` (batch is always 0 here).
        interval_scores: Vec<Vec<f64>>,
        after_layer_norm: Option<Vec<Vec<f32>>>,
        after_attention_0: Option<Vec<Vec<f32>>>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct RawCore {
        text_states: Vec<Vec<f32>>,
        query_states: Vec<Vec<f32>>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct RawCase {
        case_id: String,
        text: String,
        entity_types: Vec<String>,
        relation_types: Vec<String>,
        raw_boundary: RawBoundary,
        raw_core: Option<RawCore>,
    }

    fn load_raw_cases() -> Vec<RawCase> {
        let mut cases = Vec::new();
        for id in ["01_apple_readme", "05_edges", "10_empty_text"] {
            let path = repo_path(&format!("oracle/cases/{id}.json"));
            let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("reading {path:?}: {e} (regenerate with oracle/capture_oracle.py)")
            });
            cases.push(
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {path:?}: {e}")),
            );
        }
        cases
    }

    fn load_model() -> BoundaryModel {
        let dir = repo_path(MODEL_DIR);
        assert!(
            dir.join("model.safetensors").exists(),
            "epic Task 5 checks require the boundary checkpoint: \
             hf download fastino/gliner2.5-small-v1 --local-dir ./{MODEL_DIR}"
        );
        let gliner_cfg =
            Gliner2Config::from_file(dir.join("config.json")).expect("parse boundary config.json");
        BoundaryModel::load(&dir, gliner_cfg, &Device::Cpu).expect("load boundary model")
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

    fn max_diff1(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "len");
        got.iter()
            .zip(want)
            .map(|(u, v)| (u - v).abs())
            .fold(0f32, f32::max)
    }

    fn assert_close2(case: &str, name: &str, got: &Tensor, want: &[Vec<f32>]) {
        let m = max_diff2(&got.to_vec2::<f32>().unwrap(), want);
        eprintln!("[{case}] {name}: max abs diff {:.3e}", m);
        assert!(m < TOL, "[{case}] {name}: max abs diff {m:.3e} >= {TOL}");
    }

    /// Boundary 0 (BOS left state) and L (EOS right state) are the encoding's
    /// structural anchors; compare them explicitly on top of the full tensor.
    fn assert_boundary_edges(case: &str, got: &Tensor, want: &[Vec<f32>]) {
        let states = got.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let last = want.len() - 1;
        let m0 = max_diff1(&states[0], &want[0]);
        let ml = max_diff1(&states[last], &want[last]);
        eprintln!(
            "[{case}] boundary_states[0=BOS] max abs diff {:.3e}, [{last}=EOS] {:.3e}",
            m0, ml
        );
        assert!(m0 < TOL, "[{case}] boundary 0 (BOS): {m0:.3e} >= {TOL}");
        assert!(
            ml < TOL,
            "[{case}] boundary {last} (EOS): {ml:.3e} >= {TOL}"
        );
    }

    #[test]
    fn task5_forward_matches_python_marginals_and_boundary_states() {
        let model = load_model();
        for case in load_raw_cases() {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap_or_else(|e| panic!("[{id}] preprocess failed: {e}"));
            let out = model
                .forward(&prepared)
                .unwrap_or_else(|e| panic!("[{id}] forward failed: {e}"))
                .head;
            let rb = &case.raw_boundary;

            if let Some(rc) = &case.raw_core {
                assert_close2(
                    id,
                    "text_states",
                    &out.states.text_states.squeeze(0).unwrap(),
                    &rc.text_states,
                );
                assert_close2(
                    id,
                    "query_states",
                    &out.states.query_states.squeeze(0).unwrap(),
                    &rc.query_states,
                );
            }

            // Shapes: L+1 boundaries, Q queries, L tokens (fixture widths).
            let l1 = rb.boundary_states.len();
            let q = rb.start_logits.len();
            assert_eq!(
                out.boundary_states.dims(),
                &[1, l1, rb.boundary_states[0].len()]
            );
            assert_eq!(out.marginals.start_logits.dims(), &[1, q, l1]);
            assert_eq!(out.marginals.end_logits.dims(), &[1, q, l1]);
            assert_eq!(
                out.marginals.inside_logits.dims(),
                &[1, q, rb.inside_logits[0].len()]
            );
            assert_eq!(out.marginals.inside_prefix.dims(), &[1, q, l1]);
            assert_eq!(out.marginals.inside_prefix_mean.dims(), &[1, q, 1]);

            assert_eq!(
                out.boundary_mask.to_vec2::<u8>().unwrap(),
                vec![rb.boundary_mask.clone()],
                "[{id}] boundary_mask"
            );
            assert_boundary_edges(id, &out.boundary_states, &rb.boundary_states);
            assert_close2(
                id,
                "boundary_states",
                &out.boundary_states.squeeze(0).unwrap(),
                &rb.boundary_states,
            );
            assert_close2(
                id,
                "start_logits",
                &out.marginals.start_logits.squeeze(0).unwrap(),
                &rb.start_logits,
            );
            assert_close2(
                id,
                "end_logits",
                &out.marginals.end_logits.squeeze(0).unwrap(),
                &rb.end_logits,
            );
            assert_close2(
                id,
                "inside_logits",
                &out.marginals.inside_logits.squeeze(0).unwrap(),
                &rb.inside_logits,
            );
            assert_close2(
                id,
                "inside_prefix",
                &out.marginals.inside_prefix.squeeze(0).unwrap(),
                &rb.inside_prefix,
            );
            let mean = out.marginals.inside_prefix_mean.to_vec3::<f32>().unwrap()[0]
                .iter()
                .map(|r| r[0])
                .collect::<Vec<_>>();
            let m = max_diff1(&mean, &rb.inside_prefix_mean);
            eprintln!("[{id}] inside_prefix_mean: max abs diff {:.3e}", m);
            assert!(m < TOL, "[{id}] inside_prefix_mean: {m:.3e} >= {TOL}");

            if let (Some(ln), Some(attn)) = (&rb.after_layer_norm, &rb.after_attention_0) {
                let enc_stages = model
                    .boundary_encoder
                    .forward_stages(&out.states.text_states, &out.states.text_mask)
                    .unwrap();
                assert_close2(
                    id,
                    "after_layer_norm",
                    &enc_stages.after_layer_norm.squeeze(0).unwrap(),
                    ln,
                );
                assert_close2(
                    id,
                    "after_attention_0",
                    &enc_stages.after_attention[0].squeeze(0).unwrap(),
                    attn,
                );
            }

            // Interval reconstruction vs Python's `interval_prefix_score` and
            // the identity with the raw inside-logit sum on [start, end).
            let inside = out.marginals.inside_logits.to_vec3::<f32>().unwrap();
            let mut m = 0f32;
            for row in &rb.interval_scores {
                let (_b, qi, s, e) = (
                    row[0] as usize,
                    row[1] as usize,
                    row[2] as usize,
                    row[3] as usize,
                );
                let got = out.marginals.inside_interval_sum(0, qi, s, e).unwrap();
                m = m.max((got - row[4] as f32).abs());
                let raw: f32 = inside[0][qi][s..e].iter().sum();
                assert!(
                    (got - raw).abs() < TOL,
                    "[{id}] interval [{s},{e}) q={qi}: {got} != inside sum {raw}"
                );
            }
            eprintln!("[{id}] interval_scores: max abs diff {:.3e}", m);
            assert!(m < TOL, "[{id}] interval_scores: {m:.3e} >= {TOL}");
        }
    }

    /// One-token and edge intervals reconstruct exactly (mean restore
    /// included) on the real forward — epic "one-token/edge intervals".
    #[test]
    fn task5_one_token_and_edge_intervals() {
        let model = load_model();
        let case = load_raw_cases().remove(0); // 01_apple_readme
        let prepared = model
            .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
            .unwrap();
        let out = model.forward(&prepared).unwrap().head;
        let inside = out.marginals.inside_logits.to_vec3::<f32>().unwrap();
        let l1 = out.marginals.inside_prefix.dim(2).unwrap();
        let l = l1 - 1;
        for (qi, inside_row) in inside[0].iter().enumerate() {
            for (s, e) in [(0, 1), (l - 1, l), (0, l)] {
                let got = out.marginals.inside_interval_sum(0, qi, s, e).unwrap();
                let raw: f32 = inside_row[s..e].iter().sum();
                assert!(
                    (got - raw).abs() < TOL,
                    "interval [{s},{e}) q={qi}: {got} != inside sum {raw}"
                );
            }
        }
    }

    // ── epic Task 6: shared candidate pool + pair scoring ──────────────────

    /// Oracle fields the Task 6 comparison consumes (all 18 cases carry them).
    #[derive(Debug, serde::Deserialize)]
    struct PoolCase {
        case_id: String,
        text: String,
        entity_types: Vec<String>,
        relation_types: Vec<String>,
        selected_candidate_indices: Vec<Vec<usize>>,
        candidate_valid_mask: Vec<bool>,
        candidate_proposal_logits: Vec<f32>,
        candidate_compat_logits: Vec<f32>,
        candidate_pair_logits: Vec<Vec<f32>>,
    }

    fn load_pool_cases() -> Vec<PoolCase> {
        let cases_dir = repo_path("oracle/cases");
        let mut paths: Vec<_> = std::fs::read_dir(&cases_dir)
            .unwrap_or_else(|e| panic!("reading {:?}: {e}", cases_dir))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        paths.sort();
        assert_eq!(
            paths.len(),
            18,
            "expected 18 oracle cases in {:?}",
            cases_dir
        );
        paths
            .iter()
            .map(|p| {
                let raw = std::fs::read_to_string(p).unwrap();
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {:?}: {e}", p))
            })
            .collect()
    }

    /// Epic Task 6 acceptance: proposal/selected indices and valid masks match
    /// Python **exactly**; proposal/compat/pair logits match within `TOL`
    /// (fp32 CPU vs Python fp32). Observed max diffs are printed per case.
    #[test]
    fn task6_candidates_match_python_pool_and_pair_logits() {
        let model = load_model();
        let mut agg_prop = 0f32;
        let mut agg_compat = 0f32;
        let mut agg_pair = 0f32;
        for case in load_pool_cases() {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap_or_else(|e| panic!("[{id}] preprocess failed: {e}"));
            let out = model
                .forward(&prepared)
                .unwrap_or_else(|e| panic!("[{id}] forward failed: {e}"));
            let cand = &out.candidates;

            let idx = cand.indices.to_vec3::<u32>().unwrap();
            let valid = cand.valid_mask.to_vec2::<u8>().unwrap();
            let prop = cand.proposal_logits.to_vec2::<f32>().unwrap();
            let comp = cand.compat_logits.to_vec2::<f32>().unwrap();
            let pair = cand.pair_logits.to_vec3::<f32>().unwrap();

            let c = case.selected_candidate_indices.len();
            assert_eq!(idx[0].len(), c, "[{id}] pool row count");
            assert_eq!(valid[0].len(), c, "[{id}] valid mask length");
            assert_eq!(
                pair[0].len(),
                case.candidate_pair_logits.len(),
                "[{id}] query count"
            );
            for row in &case.candidate_pair_logits {
                assert_eq!(row.len(), c, "[{id}] pair logits row width");
            }
            for i in 0..c {
                let (s, e) = (idx[0][i][0] as usize, idx[0][i][1] as usize);
                assert_eq!(
                    [s, e],
                    case.selected_candidate_indices[i][..2],
                    "[{id}] selected_candidate_indices[{i}]"
                );
                assert_eq!(
                    valid[0][i] != 0,
                    case.candidate_valid_mask[i],
                    "[{id}] candidate_valid_mask[{i}]"
                );
            }
            let m_prop = max_diff1(&prop[0], &case.candidate_proposal_logits);
            let m_compat = max_diff1(&comp[0], &case.candidate_compat_logits);
            let mut m_pair = 0f32;
            for (qi, row) in case.candidate_pair_logits.iter().enumerate() {
                m_pair = m_pair.max(max_diff1(&pair[0][qi], row));
            }
            eprintln!(
                "[{id}] candidates: proposal {:.3e} compat {:.3e} pair {:.3e} (indices/mask exact)",
                m_prop, m_compat, m_pair
            );
            assert!(m_prop < TOL, "[{id}] proposal_logits {m_prop:.3e} >= {TOL}");
            assert!(
                m_compat < TOL,
                "[{id}] compat_logits {m_compat:.3e} >= {TOL}"
            );
            assert!(m_pair < TOL, "[{id}] pair_logits {m_pair:.3e} >= {TOL}");
            agg_prop = agg_prop.max(m_prop);
            agg_compat = agg_compat.max(m_compat);
            agg_pair = agg_pair.max(m_pair);
        }
        eprintln!(
            "task6 aggregate max abs diff: proposal {:.3e} compat {:.3e} pair {:.3e}",
            agg_prop, agg_compat, agg_pair
        );
    }

    /// Epic "Explicitly assert that marginals are added once": the pair score
    /// decomposes as `pre_marginal + start + end + inside` with the gathered
    /// start/end marginals at coefficient **exactly 1**, and not the
    /// double-added variant. Also checks the proposal prior carries the
    /// marginals while the scorer's `compat` prior does not (Finding 7).
    #[test]
    fn task6_marginals_are_added_exactly_once() {
        let model = load_model();
        for case in load_pool_cases().iter().take(3) {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap();
            let states = model.encode(&prepared).unwrap();
            let head = model.forward_head(&states).unwrap();
            let (cand, parts) = model.score_candidates_parts(&head).unwrap();

            let pair = parts
                .pair_logits
                .squeeze(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(); // [C, Q]
            let pre = parts
                .pre_marginal
                .squeeze(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let st = parts
                .start_term
                .squeeze(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let et = parts.end_term.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
            let it = parts
                .inside_term
                .squeeze(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let sl = head.marginals.start_logits.to_vec3::<f32>().unwrap(); // [1, Q, N]
            let el = head.marginals.end_logits.to_vec3::<f32>().unwrap();
            let idx = cand.indices.to_vec3::<u32>().unwrap();
            let valid = cand.valid_mask.to_vec2::<u8>().unwrap();
            let prop = cand.proposal_logits.to_vec2::<f32>().unwrap();
            let comp = cand.compat_logits.to_vec2::<f32>().unwrap();
            let q = pair[0].len();

            // Query-max endpoint union (all queries valid at B=1).
            let n = sl[0][0].len();
            let mut union_s = vec![MASK_LOGIT; n];
            let mut union_e = vec![MASK_LOGIT; n];
            for i in 0..n {
                for qq in 0..q {
                    union_s[i] = union_s[i].max(sl[0][qq][i]);
                    union_e[i] = union_e[i].max(el[0][qq][i]);
                }
            }

            let mut checked = 0usize;
            for (ci, &is_valid) in valid[0].iter().enumerate() {
                let (s, e) = (idx[0][ci][0] as usize, idx[0][ci][1] as usize);
                if is_valid == 0 {
                    assert_eq!(prop[0][ci], MASK_LOGIT, "[{id}] invalid proposal sentinel");
                    assert_eq!(comp[0][ci], 0.0, "[{id}] invalid compat zeroed");
                    continue;
                }
                // Finding 7: proposal = compat + union marginals (full prior),
                // while the scorer consumes the marginal-free `compat` prior.
                let want_prop = comp[0][ci] + union_s[s] + union_e[e];
                assert!(
                    (prop[0][ci] - want_prop).abs() < TOL,
                    "[{id}] proposal[{ci}] = compat + union marginals: {} != {want_prop}",
                    prop[0][ci]
                );
                for qq in 0..q {
                    // Coefficient exactly 1.
                    assert_eq!(st[ci][qq], sl[0][qq][s], "[{id}] start marginal once");
                    assert_eq!(et[ci][qq], el[0][qq][e], "[{id}] end marginal once");
                    let once = pre[ci][qq] + st[ci][qq] + et[ci][qq] + it[ci][qq];
                    assert!(
                        (pair[ci][qq] - once).abs() < 1e-6,
                        "[{id}] pair != pre + marginals once at (c={ci}, q={qq})"
                    );
                    if (st[ci][qq] + et[ci][qq]).abs() > 1e-3 {
                        let twice = pre[ci][qq] + 2.0 * st[ci][qq] + 2.0 * et[ci][qq] + it[ci][qq];
                        assert!(
                            (pair[ci][qq] - twice).abs() > 1e-3,
                            "[{id}] marginals double-added at (c={ci}, q={qq})"
                        );
                        checked += 1;
                    }
                }
            }
            assert!(
                checked > 0,
                "[{id}] no non-degenerate marginal rows checked"
            );
            eprintln!("[{id}] marginals-added-once: {checked} rows asserted");
        }
    }

    // ── epic Task 7: boundary entity decode ─────────────────────────────────

    use crate::boundary_decode::threshold_query_candidates;
    use std::collections::BTreeMap;

    /// `final_output.entities` item (code-point offsets into `normalized_text`).
    #[derive(Debug, serde::Deserialize)]
    struct OracleEntityItem {
        text: String,
        confidence: f64,
        start: usize,
        end: usize,
    }

    #[derive(Debug, serde::Deserialize, Default)]
    struct OracleFinalOutput {
        #[serde(default)]
        entities: BTreeMap<String, Vec<OracleEntityItem>>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct OracleThresholded {
        probability: f64,
        start: usize,
        end: usize,
    }

    /// Fixture fields the Task 7 decode comparison consumes.
    #[derive(Debug, serde::Deserialize)]
    struct DecodeCase {
        case_id: String,
        text: String,
        threshold: f32,
        entity_types: Vec<String>,
        relation_types: Vec<String>,
        thresholded_candidates: Vec<Vec<OracleThresholded>>,
        null_logits: Vec<f64>,
        count_log_rates: Vec<f64>,
        final_output: OracleFinalOutput,
    }

    fn load_decode_cases() -> Vec<DecodeCase> {
        let cases_dir = repo_path("oracle/cases");
        let mut paths: Vec<_> = std::fs::read_dir(&cases_dir)
            .unwrap_or_else(|e| panic!("reading {:?}: {e}", cases_dir))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        paths.sort();
        paths
            .iter()
            .map(|p| {
                let raw = std::fs::read_to_string(p).unwrap();
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {:?}: {e}", p))
            })
            .collect()
    }

    fn cp_to_byte_table(s: &str) -> Vec<usize> {
        let mut table: Vec<usize> = s.char_indices().map(|(b, _)| b).collect();
        table.push(s.len());
        table
    }

    /// Task 7 acceptance (intermediates): the decode heads'
    /// `null_logits`/`count_log_rates` and the threshold stage's
    /// `thresholded_candidates` (pool-row order, post `pair_temperature`
    /// sigmoid, `>= 0.5` inclusive, no adaptive fill on this checkpoint) match
    /// Python over all 18 cases — indices exactly, probabilities/logits within
    /// `TOL`.
    #[test]
    fn task7_null_count_and_thresholded_candidates_match_python() {
        let model = load_model();
        let mut agg_null = 0f64;
        let mut agg_count = 0f64;
        let mut agg_prob = 0f64;
        for case in load_decode_cases() {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap_or_else(|e| panic!("[{id}] preprocess failed: {e}"));
            let out = model
                .forward(&prepared)
                .unwrap_or_else(|e| panic!("[{id}] forward failed: {e}"));

            let nulls = out
                .null_logits
                .as_ref()
                .unwrap_or_else(|| panic!("[{id}] null_logits missing (enable_abstention)"));
            let counts = out
                .count_log_rates
                .as_ref()
                .unwrap_or_else(|| panic!("[{id}] count_log_rates missing (enable_count_head)"));
            let m_null = max_diff1(
                nulls,
                &case
                    .null_logits
                    .iter()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
            );
            let m_count = max_diff1(
                counts,
                &case
                    .count_log_rates
                    .iter()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
            );
            assert!(m_null < TOL, "[{id}] null_logits {m_null:.3e} >= {TOL}");
            assert!(
                m_count < TOL,
                "[{id}] count_log_rates {m_count:.3e} >= {TOL}"
            );
            agg_null = agg_null.max(m_null as f64);
            agg_count = agg_count.max(m_count as f64);

            // Threshold stage on every query (entity and relation alike).
            let idx = out.candidates.indices.to_vec3::<u32>().unwrap();
            let valid = out.candidates.valid_mask.to_vec2::<u8>().unwrap();
            let pair = out.candidates.pair_logits.to_vec3::<f32>().unwrap();
            let pool_indices: Vec<(u32, u32)> = idx[0].iter().map(|r| (r[0], r[1])).collect();
            let pool_valid: Vec<bool> = valid[0].iter().map(|&v| v != 0).collect();
            assert_eq!(
                pair[0].len(),
                case.thresholded_candidates.len(),
                "[{id}] query count vs thresholded rows"
            );
            for (qi, want_rows) in case.thresholded_candidates.iter().enumerate() {
                let got_rows = threshold_query_candidates(
                    &pair[0][qi],
                    &pool_valid,
                    &pool_indices,
                    model.gliner_cfg.boundary_head.pair_temperature as f32,
                    case.threshold,
                    None, // adaptive_threshold=false on this checkpoint
                )
                .unwrap_or_else(|e| panic!("[{id}] threshold q={qi}: {e}"));
                assert_eq!(
                    got_rows.len(),
                    want_rows.len(),
                    "[{id}] thresholded count q={qi}: got {:?} want {:?}",
                    got_rows
                        .iter()
                        .map(|s| (s.probability, s.start, s.end))
                        .collect::<Vec<_>>(),
                    want_rows
                );
                for (got, want) in got_rows.iter().zip(want_rows) {
                    assert_eq!(
                        [got.start as usize, got.end as usize],
                        [want.start, want.end],
                        "[{id}] thresholded span q={qi}"
                    );
                    let diff = (got.probability as f64 - want.probability).abs();
                    agg_prob = agg_prob.max(diff);
                    assert!(
                        diff < TOL as f64,
                        "[{id}] thresholded probability q={qi} diff {diff:.3e} >= {TOL}"
                    );
                }
            }
        }
        eprintln!(
            "task7 intermediates aggregate max abs diff: null {:.3e} count {:.3e} thresholded_prob {:.3e}",
            agg_null, agg_count, agg_prob
        );
    }

    /// Task 7 acceptance (final entities): `Model`-level decode output matches
    /// Python's `final_output.entities` on every entity case (01–11, 13, 18)
    /// — labels and surfaces exact, order exact (declared type order, then
    /// descending confidence within a type), code-point offsets converted to
    /// UTF-8 bytes on the caller's text, confidence within `TOL`.
    #[test]
    fn task7_final_entities_match_python() {
        let model = load_model();
        let mut agg_conf = 0f64;
        let mut checked = 0usize;
        for case in load_decode_cases()
            .into_iter()
            .filter(|c| !c.entity_types.is_empty())
        {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap_or_else(|e| panic!("[{id}] preprocess failed: {e}"));
            let out = model
                .forward(&prepared)
                .unwrap_or_else(|e| panic!("[{id}] forward failed: {e}"));
            let got = model
                .decode_entities(&prepared, &out, case.threshold)
                .unwrap_or_else(|e| panic!("[{id}] decode failed: {e}"));

            // Expected flat order = Python's final_output: declared type
            // order, then the fixture's per-type list order.
            let mut expected: Vec<(&str, &OracleEntityItem)> = Vec::new();
            for name in &case.entity_types {
                let items = case
                    .final_output
                    .entities
                    .get(name)
                    .unwrap_or_else(|| panic!("[{id}] final_output missing type {name:?}"));
                expected.extend(items.iter().map(|item| (name.as_str(), item)));
            }
            let covered: usize = case.final_output.entities.values().map(Vec::len).sum();
            assert_eq!(
                covered,
                expected.len(),
                "[{id}] final_output has types outside the declared order"
            );
            assert_eq!(
                got.len(),
                expected.len(),
                "[{id}] entity count: got {:?} want {:?}",
                got.iter()
                    .map(|e| (e.entity_type.clone(), e.text.clone()))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|(_, e)| e.text.clone())
                    .collect::<Vec<_>>()
            );

            let caller_cp = case.text.chars().count();
            let table = cp_to_byte_table(&case.text);
            for (g, (want_type, want)) in got.iter().zip(&expected) {
                assert_eq!(&g.entity_type, want_type, "[{id}] label/order");
                assert_eq!(g.text, want.text, "[{id}] surface for {:?}", want.text);
                // The suffix rule is inert on this corpus: every fixture span
                // lies inside the caller's text (asserted) and converts to the
                // same bytes Rust returns.
                assert!(
                    want.end <= caller_cp,
                    "[{id}] fixture span reaches past the caller text (suffix rule no longer inert)"
                );
                let (b0, b1) = (table[want.start], table[want.end]);
                assert_eq!(
                    (g.char_start, g.char_end),
                    (b0, b1),
                    "[{id}] byte offsets for {:?}",
                    want.text
                );
                // Every returned surface slices the caller's exact input.
                let sliced = case.text[g.char_start..g.char_end].trim();
                assert_eq!(sliced, g.text, "[{id}] surface slices caller input");
                let diff = (g.confidence as f64 - want.confidence).abs();
                agg_conf = agg_conf.max(diff);
                assert!(
                    diff < TOL as f64,
                    "[{id}] confidence for {:?}: diff {diff:.3e} >= {TOL}",
                    want.text
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no entities compared");
        eprintln!(
            "task7 final entities: {checked} entities exact (labels/surfaces/offsets), \
             confidence max abs diff {:.3e}",
            agg_conf
        );
    }

    /// The end-to-end `BoundaryModel::predict_entities` path (the CLI's
    /// entity-only request) returns the same entities as the lower-level
    /// decode; `Model::predict_entities` dispatches to it verbatim.
    #[test]
    fn task7_predict_entities_matches_decode() {
        let model = load_model();
        let cases: Vec<DecodeCase> = load_decode_cases()
            .into_iter()
            .filter(|c| !c.entity_types.is_empty() && c.relation_types.is_empty())
            .collect();
        for case in &cases {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap();
            let out = model.forward(&prepared).unwrap();
            let via_decode = model
                .decode_entities(&prepared, &out, case.threshold)
                .unwrap();
            let via_predict = model
                .predict_entities(&case.text, &case.entity_types, case.threshold)
                .unwrap();
            assert_eq!(
                via_predict.len(),
                via_decode.len(),
                "[{id}] predict vs decode"
            );
            for (a, b) in via_predict.iter().zip(&via_decode) {
                assert_eq!(a.entity_type, b.entity_type, "[{id}] label");
                assert_eq!(a.text, b.text, "[{id}] surface");
                assert_eq!(
                    (a.char_start, a.char_end),
                    (b.char_start, b.char_end),
                    "[{id}] offsets"
                );
                assert_eq!(a.confidence, b.confidence, "[{id}] confidence");
            }
        }
        // One-shot check that the shared dispatch entry runs the boundary path.
        let dispatch = crate::model::Model::Boundary(load_model());
        let first = &cases[0];
        let entities = dispatch
            .predict_entities(&first.text, &first.entity_types, first.threshold)
            .expect("Model::predict_entities on a boundary checkpoint");
        assert!(
            !entities.is_empty(),
            "dispatch returned no entities for {}",
            first.case_id
        );
    }

    // ── epic Task 8: boundary relation extraction ────────────────────────────

    /// `proposed_argument_pairs` item: half-open word indices in the text-word
    /// space (incl. any classification prefix; `word_offset` shifts to
    /// document words — 0 for this corpus).
    #[derive(Debug, serde::Deserialize)]
    struct OracleRelationPair {
        relation_type: String,
        head: Vec<usize>,
        tail: Vec<usize>,
        head_prob: f64,
        tail_prob: f64,
        head_query_id: usize,
        tail_query_id: usize,
    }

    /// `final_output.relation_extraction` argument (code-point offsets into
    /// `normalized_text`).
    #[derive(Debug, serde::Deserialize)]
    struct OracleRelationArg {
        text: String,
        start: usize,
        end: usize,
        confidence: f64,
    }

    #[derive(Debug, serde::Deserialize)]
    struct OracleRelationEdge {
        head: OracleRelationArg,
        tail: OracleRelationArg,
    }

    #[derive(Debug, serde::Deserialize, Default)]
    struct OracleRelationFinal {
        #[serde(default)]
        relation_extraction: BTreeMap<String, Vec<OracleRelationEdge>>,
    }

    /// Fixture fields the Task 8 relation comparison consumes (cases 12–18).
    #[derive(Debug, serde::Deserialize)]
    struct RelationCase {
        case_id: String,
        text: String,
        threshold: f32,
        entity_types: Vec<String>,
        relation_types: Vec<String>,
        word_offset: usize,
        proposed_argument_pairs: Vec<OracleRelationPair>,
        relation_logits: Vec<f64>,
        final_output: OracleRelationFinal,
    }

    fn load_relation_cases() -> Vec<RelationCase> {
        let cases_dir = repo_path("oracle/cases");
        let mut paths: Vec<_> = std::fs::read_dir(&cases_dir)
            .unwrap_or_else(|e| panic!("reading {:?}: {e}", cases_dir))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        paths.sort();
        paths.retain(|p| {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            (12..=18).any(|n| name.starts_with(&format!("{n:02}_")))
        });
        assert_eq!(
            paths.len(),
            7,
            "expected relation cases 12-18 in {cases_dir:?}"
        );
        paths
            .iter()
            .map(|p| {
                let raw = std::fs::read_to_string(p).unwrap();
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {:?}: {e}", p))
            })
            .collect()
    }

    /// Expected flat edge list in Python `final_output.relation_extraction`
    /// order: relation types in declared order (the proposal list is
    /// relation-major, so first-surviving-pair order == declared order), and
    /// within a type the fixture's per-type edge order.
    fn expected_relation_edges(case: &RelationCase) -> Vec<(&str, &OracleRelationEdge)> {
        let mut expected = Vec::new();
        for name in &case.relation_types {
            if let Some(items) = case.final_output.relation_extraction.get(name) {
                expected.extend(items.iter().map(|item| (name.as_str(), item)));
            }
        }
        let covered: usize = case
            .final_output
            .relation_extraction
            .values()
            .map(Vec::len)
            .sum();
        assert_eq!(
            covered,
            expected.len(),
            "[{}] final_output has relation types outside the declared order",
            case.case_id
        );
        expected
    }

    /// Task 8 acceptance (intermediates): the typed/capped proposed argument
    /// pairs match Python exactly (head/tail word indices, role query ids,
    /// relation type) with argument probabilities within `TOL`, and the raw
    /// `SparseRelationScorer` logits within `TOL` over relation cases 12–18.
    #[test]
    fn task8_proposed_pairs_and_relation_logits_match_python() {
        let model = load_model();
        let mut agg_prob = 0f64;
        let mut agg_logit = 0f64;
        for case in load_relation_cases() {
            let id = &case.case_id;
            assert_eq!(
                case.word_offset, 0,
                "[{id}] classification prefixes are not supported (word_offset must be 0)"
            );
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap_or_else(|e| panic!("[{id}] preprocess failed: {e}"));
            let out = model
                .forward(&prepared)
                .unwrap_or_else(|e| panic!("[{id}] forward failed: {e}"));
            let (pairs, logits) = model
                .propose_relations(&prepared, &out)
                .unwrap_or_else(|e| panic!("[{id}] propose failed: {e}"));

            assert_eq!(
                pairs.len(),
                case.proposed_argument_pairs.len(),
                "[{id}] proposed pair count: got {pairs:?}"
            );
            assert_eq!(
                logits.len(),
                case.relation_logits.len(),
                "[{id}] logit count"
            );
            for (i, (pair, want)) in pairs.iter().zip(&case.proposed_argument_pairs).enumerate() {
                let got_type = &prepared.relation_role_routing[pair.relation_index].relation_type;
                assert_eq!(
                    got_type, &want.relation_type,
                    "[{id}] pair {i} relation type"
                );
                assert_eq!(
                    [pair.head_start as usize, pair.head_end as usize],
                    want.head[..2],
                    "[{id}] pair {i} head span"
                );
                assert_eq!(
                    [pair.tail_start as usize, pair.tail_end as usize],
                    want.tail[..2],
                    "[{id}] pair {i} tail span"
                );
                assert_eq!(
                    pair.head_query_id, want.head_query_id,
                    "[{id}] pair {i} head query"
                );
                assert_eq!(
                    pair.tail_query_id, want.tail_query_id,
                    "[{id}] pair {i} tail query"
                );
                let dp = ((pair.head_prob as f64 - want.head_prob).abs())
                    .max((pair.tail_prob as f64 - want.tail_prob).abs());
                agg_prob = agg_prob.max(dp);
                assert!(
                    dp < TOL as f64,
                    "[{id}] pair {i} argument prob diff {dp:.3e}"
                );
            }
            for (i, (got, want)) in logits.iter().zip(&case.relation_logits).enumerate() {
                let diff = (*got as f64 - *want).abs();
                agg_logit = agg_logit.max(diff);
                assert!(
                    diff < TOL as f64,
                    "[{id}] relation_logit[{i}] diff {diff:.3e} >= {TOL}"
                );
            }
        }
        eprintln!(
            "task8 intermediates: proposed pairs exact (indices/queries/types), \
             arg prob max abs diff {:.3e}, relation logits max abs diff {:.3e}",
            agg_prob, agg_logit
        );
    }

    /// Task 8 acceptance (final edges): `decode_relations` output matches
    /// Python's `final_output.relation_extraction` on cases 12–18 — exact
    /// types, **direction** (head is the head-role argument), surfaces, and
    /// code-point→byte offsets on the caller's text; confidence within `TOL`.
    #[test]
    fn task8_final_relations_match_python() {
        let model = load_model();
        let mut agg_conf = 0f64;
        let mut checked = 0usize;
        for case in load_relation_cases() {
            let id = &case.case_id;
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap();
            let out = model.forward(&prepared).unwrap();
            let got = model
                .decode_relations(&prepared, &out, case.threshold)
                .unwrap_or_else(|e| panic!("[{id}] decode failed: {e}"));
            let expected = expected_relation_edges(&case);
            assert_eq!(
                got.len(),
                expected.len(),
                "[{id}] edge count: got {:?} want {:?}",
                got.iter()
                    .map(|r| (&r.relation_type, &r.head.text, &r.tail.text))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|(t, e)| (*t, &e.head.text, &e.tail.text))
                    .collect::<Vec<_>>()
            );

            let caller_cp = case.text.chars().count();
            let table = cp_to_byte_table(&case.text);
            for (g, (want_type, want)) in got.iter().zip(&expected) {
                assert_eq!(&g.relation_type, want_type, "[{id}] relation type/order");
                // Direction: the head is the head-role argument even when its
                // mention follows the tail's in the text (case 15).
                assert_eq!(g.head.text, want.head.text, "[{id}] head surface");
                assert_eq!(g.tail.text, want.tail.text, "[{id}] tail surface");
                for (arg, w) in [(&g.head, &want.head), (&g.tail, &want.tail)] {
                    assert!(
                        w.end <= caller_cp,
                        "[{id}] fixture span reaches past the caller text"
                    );
                    let (b0, b1) = (table[w.start], table[w.end]);
                    assert_eq!(
                        (arg.char_start, arg.char_end),
                        (b0, b1),
                        "[{id}] byte offsets for {:?}",
                        w.text
                    );
                    let sliced = case.text[arg.char_start..arg.char_end].trim();
                    assert_eq!(sliced, arg.text, "[{id}] surface slices caller input");
                }
                let diff = (g.confidence as f64 - want.head.confidence).abs();
                agg_conf = agg_conf.max(diff);
                assert!(
                    diff < TOL as f64,
                    "[{id}] confidence for {:?}→{:?}: diff {diff:.3e}",
                    g.head.text,
                    g.tail.text
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no relation edges compared");
        eprintln!(
            "task8 final edges: {checked} edges exact (types/direction/surfaces/offsets), \
             confidence max abs diff {:.3e}",
            agg_conf
        );
    }

    /// Direction preservation: case 14 (head mention precedes tail) and case
    /// 15 (tail mention precedes head) both keep the semantic founder as the
    /// directed head.
    #[test]
    fn task8_direction_is_preserved() {
        let model = load_model();
        let cases = load_relation_cases();
        let run = |id: &str| -> Vec<ExtractedRelation> {
            let case = cases.iter().find(|c| c.case_id == id).unwrap();
            let prepared = model
                .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
                .unwrap();
            let out = model.forward(&prepared).unwrap();
            model
                .decode_relations(&prepared, &out, case.threshold)
                .unwrap()
        };
        let fwd = run("14_direction_fwd");
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].head.text, "Steve Jobs");
        assert_eq!(fwd[0].tail.text, "Apple");
        assert!(
            fwd[0].head.char_start < fwd[0].tail.char_start,
            "case 14: head first"
        );
        let rev = run("15_direction_rev");
        assert_eq!(rev.len(), 1);
        assert_eq!(rev[0].head.text, "Steve Jobs");
        assert_eq!(rev[0].tail.text, "Apple");
        assert!(
            rev[0].head.char_start > rev[0].tail.char_start,
            "case 15: head mention follows the tail — direction is semantic, not textual"
        );
    }

    /// No-relation case 17 is near-threshold sensitive: the founder edge is
    /// emitted (Python parity — sigmoid of the raw logit 0.042 is just above
    /// 0.5), so this checks both the parity and how close the decision sits
    /// to the threshold (a small logit drift flips it).
    #[test]
    fn task8_no_relation_case_is_near_threshold_sensitive() {
        let model = load_model();
        let case = load_relation_cases()
            .into_iter()
            .find(|c| c.case_id == "17_no_relation")
            .unwrap();
        let prepared = model
            .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
            .unwrap();
        let out = model.forward(&prepared).unwrap();
        let got = model
            .decode_relations(&prepared, &out, case.threshold)
            .unwrap();
        assert_eq!(
            got.len(),
            1,
            "Python emits the near-threshold edge (parity): {got:?}"
        );
        let conf = got[0].confidence;
        assert!(
            conf > 0.5 && conf < 0.52,
            "case 17 confidence {conf} must sit within 0.02 of the 0.5 threshold"
        );
        // The decision is threshold-sensitive: a slightly higher threshold
        // flips it to the empty/negative result.
        let stricter = model.decode_relations(&prepared, &out, 0.52).unwrap();
        assert!(
            stricter.is_empty(),
            "threshold 0.52 must drop the near-threshold edge"
        );
    }

    /// End-to-end relation-only request (no entity schema) through both
    /// `BoundaryModel::predict_relations` and the shared `Model` dispatch,
    /// plus the fail-clearly behavior for checkpoints without a relation head
    /// (span architecture, `enable_relations=false`, empty type list).
    #[test]
    fn task8_predict_relations_and_fail_clearly() {
        let mut model = load_model();

        // Relation-only request: the two role queries + candidate pool are
        // enough; no entity schema is requested.
        let case = load_relation_cases()
            .into_iter()
            .find(|c| c.case_id == "12_relation_only")
            .unwrap();
        let via_predict = model
            .predict_relations(&case.text, &[], &case.relation_types, case.threshold)
            .expect("relation-only predict_relations");
        let prepared = model
            .preprocess(&case.text, &case.entity_types, &case.relation_types, None)
            .unwrap();
        let out = model.forward(&prepared).unwrap();
        let via_decode = model
            .decode_relations(&prepared, &out, case.threshold)
            .unwrap();
        assert_eq!(
            via_predict, via_decode,
            "predict_relations == decode_relations"
        );
        assert_eq!(via_predict.len(), 1);
        assert_eq!(via_predict[0].relation_type, "founder");
        assert_eq!(via_predict[0].head.text, "Steve Jobs");
        assert_eq!(via_predict[0].tail.text, "Apple");

        // Shared dispatch (Task 3's prediction operation) reaches the same
        // relation path.
        let dispatch = crate::model::Model::Boundary(load_model());
        let dispatched = dispatch
            .predict_relations(&case.text, &[], &case.relation_types, case.threshold)
            .expect("Model::predict_relations on a boundary checkpoint");
        assert_eq!(dispatched, via_predict);

        // Empty relation type list fails clearly (not a silent empty result).
        let empty = model
            .predict_relations(&case.text, &[], &[], 0.5)
            .expect_err("empty relation types must error");
        assert!(
            empty.to_string().contains("at least one relation type"),
            "unhelpful error: {empty}"
        );

        // A boundary checkpoint with enable_relations=false must fail clearly.
        model.gliner_cfg.boundary_head.enable_relations = false;
        let disabled = model
            .predict_relations(&case.text, &[], &case.relation_types, case.threshold)
            .expect_err("enable_relations=false must error");
        assert!(
            disabled
                .to_string()
                .contains("relation extraction is not available"),
            "unhelpful error: {disabled}"
        );

        // The span architecture has no relation head: fail clearly there too.
        let span_dir = repo_path("models/gliner2-large-v1");
        assert!(
            span_dir.join("model.safetensors").exists(),
            "the fail-clearly span check requires ./models/gliner2-large-v1"
        );
        let span = crate::model::Model::load(&span_dir, &Device::Cpu).expect("load span model");
        let span_err = span
            .predict_relations(&case.text, &[], &case.relation_types, case.threshold)
            .expect_err("span checkpoints must reject relation requests");
        assert!(
            span_err
                .to_string()
                .contains("not supported by the span architecture"),
            "unhelpful error: {span_err}"
        );
    }
}
