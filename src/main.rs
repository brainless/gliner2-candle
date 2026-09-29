mod boundary;
mod boundary_decode;
mod boundary_enc;
mod boundary_pool;
mod boundary_pre;
mod boundary_rel;
mod cases;
mod config;
mod count_lstm;
mod hub;
mod inference;
mod model;
mod processor;
mod span_rep;

use anyhow::{bail, Context, Result};
use clap::Parser;
use model::Model;
use std::path::{Path, PathBuf};
use tracing::info;

const ABOUT: &str = "GLiNER2 entity and relation extraction — Candle backend";

const AFTER_HELP: &str = "\
One-off extraction needs only a text, the type lists, and a model (ID or
local directory):
  gliner2-candle --model-id fastino/gliner2.5-small-v1 \\
    --text \"Steve Jobs founded Apple.\" --entities person,organization \\
    --relations founder

  # Relations only (boundary checkpoints; span checkpoints fail clearly):
  gliner2-candle --model-id fastino/gliner2.5-small-v1 \\
    --text \"Steve Jobs founded Apple.\" --relations founder

  # Local model directory (no download):
  gliner2-candle --model-dir ./models/gliner2.5-small-v1 \\
    --text \"Apple was founded by Steve Jobs in Cupertino.\" \\
    --entities person,organization,location

  # Machine-readable JSON on stdout:
  gliner2-candle --model-id fastino/gliner2.5-small-v1 --json \\
    --text \"Apple was founded by Steve Jobs.\" --entities person,organization

  # Versioned agent case file (pass/fail report, nonzero exit on failure):
  gliner2-candle --model-id fastino/gliner2.5-small-v1 --cases cases/cli-cases.json

Offsets in output and case files are half-open UTF-8 byte offsets into --text.
--model-id downloads config/tokenizer/encoder config/weights into
--models-root/<name>/ (e.g. ./models/gliner2.5-small-v1/) on first use and
reuses the identity-verified cache afterwards; --model-dir takes precedence
and never downloads.";

#[derive(Parser, Debug)]
#[command(name = "gliner2-candle", about = ABOUT, version, after_help = AFTER_HELP)]
struct Args {
    /// Path to a local model directory (config.json, model.safetensors,
    /// tokenizer.json, encoder_config/config.json). Takes precedence over
    /// --model-id and never triggers a download.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// HuggingFace model ID to download/reuse (e.g.
    /// fastino/gliner2.5-small-v1). Files land in --models-root/<name>/.
    #[arg(long, default_value = "fastino/gliner2-large-v1")]
    model_id: String,

    /// Hub revision (commit sha, tag, or branch) for --model-id. Defaults to
    /// the pinned revision documented in README.md for the fastino
    /// checkpoints, otherwise "main" resolved at download time.
    #[arg(long)]
    revision: Option<String>,

    /// Directory under which model-specific cache directories live.
    #[arg(long, default_value = "./models")]
    models_root: PathBuf,

    /// Text to extract from (one-off mode)
    #[arg(long)]
    text: Option<String>,

    /// Comma-separated entity types to extract
    #[arg(long, value_delimiter = ',')]
    entities: Vec<String>,

    /// Comma-separated relation types to extract (boundary checkpoints only)
    #[arg(long, value_delimiter = ',')]
    relations: Vec<String>,

    /// Score threshold in [0.0, 1.0] (one-off mode; cases carry their own)
    #[arg(long)]
    threshold: Option<f32>,

    /// Run a versioned JSON case file through the production inference path:
    /// per-case pass/fail report, nonzero exit if any case fails
    #[arg(long)]
    cases: Option<PathBuf>,

    /// Emit machine-readable JSON on stdout (one-off results or case report)
    #[arg(long)]
    json: bool,

    /// Device: "cpu" or "cuda:N"
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Print weight keys (name, dtype, shape) and exit. Without a value lists
    /// all entries; with N lists the first N (for debugging prefix mismatches)
    #[arg(long, num_args = 0..=1)]
    list_weights: Option<Option<usize>>,
}

fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    match run(&args) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::from(1)
        }
    }
}

/// Expand a leading `~` in a user-supplied path.
fn expand_tilde(p: &Path) -> PathBuf {
    PathBuf::from(shellexpand::tilde(&p.to_string_lossy()).into_owned())
}

/// Trimmed, validated type list (`--entities` / `--relations`).
fn clean_types(what: &str, types: &[String]) -> Result<Vec<String>> {
    let trimmed: Vec<String> = types.iter().map(|t| t.trim().to_string()).collect();
    if trimmed.iter().any(|t| t.is_empty()) {
        bail!("{what} contains an empty type name — use comma-separated non-empty names");
    }
    Ok(trimmed)
}

/// One-off request validation: text, type lists, threshold range.
fn validate_one_off(
    text: &Option<String>,
    entity_types: &[String],
    relation_types: &[String],
    threshold: Option<f32>,
) -> Result<(String, Vec<String>, Vec<String>, f32)> {
    let text = match text {
        Some(t) => t.clone(),
        None => {
            bail!("no input text — pass --text \"...\" (or --cases FILE for the case-file mode)")
        }
    };
    let entity_types = clean_types("--entities", entity_types)?;
    let relation_types = clean_types("--relations", relation_types)?;
    if entity_types.is_empty() && relation_types.is_empty() {
        bail!(
            "no extraction types — pass --entities and/or --relations with at least one type \
             name"
        );
    }
    let threshold = threshold.unwrap_or(0.5);
    if !(0.0..=1.0).contains(&threshold) || threshold.is_nan() {
        bail!("threshold {threshold} is outside [0.0, 1.0]");
    }
    Ok((text, entity_types, relation_types, threshold))
}

/// Run one production inference request (the shared path for one-off, JSON,
/// and case-file modes).
struct Extraction {
    entities: Option<Vec<inference::ExtractedEntity>>,
    relations: Option<Vec<boundary_rel::ExtractedRelation>>,
}

fn run_extraction(
    model: &Model,
    text: &str,
    entity_types: &[String],
    relation_types: &[String],
    threshold: f32,
) -> Result<Extraction> {
    let (entities, relations) =
        model.predict_extractions(text, entity_types, relation_types, threshold)?;
    Ok(Extraction {
        entities: (!entity_types.is_empty()).then_some(entities),
        relations: (!relation_types.is_empty()).then_some(relations),
    })
}

// ── JSON output (stable field names and ordering) ──────────────────────────

#[derive(serde::Serialize)]
struct JsonModel {
    id: Option<String>,
    revision: Option<String>,
    dir: String,
    architecture: String,
}

#[derive(serde::Serialize)]
struct JsonEntity {
    #[serde(rename = "type")]
    entity_type: String,
    text: String,
    byte_start: usize,
    byte_end: usize,
    confidence: f32,
}

#[derive(serde::Serialize)]
struct JsonArgument {
    text: String,
    byte_start: usize,
    byte_end: usize,
}

#[derive(serde::Serialize)]
struct JsonRelation {
    #[serde(rename = "type")]
    relation_type: String,
    confidence: f32,
    head: JsonArgument,
    tail: JsonArgument,
}

#[derive(serde::Serialize)]
struct JsonExtraction {
    schema_version: u32,
    kind: &'static str,
    model: JsonModel,
    tasks: Vec<String>,
    threshold: f32,
    entity_types: Vec<String>,
    relation_types: Vec<String>,
    entities: Vec<JsonEntity>,
    relations: Vec<JsonRelation>,
}

#[derive(serde::Serialize)]
struct JsonCaseResult {
    name: String,
    status: &'static str,
    mismatches: Vec<String>,
    entities: Vec<JsonEntity>,
    relations: Vec<JsonRelation>,
}

#[derive(serde::Serialize)]
struct JsonCaseReport {
    schema_version: u32,
    kind: &'static str,
    model: JsonModel,
    cases: Vec<JsonCaseResult>,
    passed: usize,
    failed: usize,
    total: usize,
}

const SCHEMA_VERSION: u32 = 1;

fn json_entity(e: &inference::ExtractedEntity) -> JsonEntity {
    JsonEntity {
        entity_type: e.entity_type.clone(),
        text: e.text.clone(),
        byte_start: e.char_start,
        byte_end: e.char_end,
        confidence: e.confidence,
    }
}

fn json_relation(r: &boundary_rel::ExtractedRelation) -> JsonRelation {
    JsonRelation {
        relation_type: r.relation_type.clone(),
        confidence: r.confidence,
        head: JsonArgument {
            text: r.head.text.clone(),
            byte_start: r.head.char_start,
            byte_end: r.head.char_end,
        },
        tail: JsonArgument {
            text: r.tail.text.clone(),
            byte_start: r.tail.char_start,
            byte_end: r.tail.char_end,
        },
    }
}

fn json_model(resolved: &hub::ResolvedModel, architecture: &str) -> JsonModel {
    JsonModel {
        id: resolved.model_id.clone(),
        revision: resolved.revision.clone(),
        dir: resolved.dir.display().to_string(),
        architecture: architecture.to_string(),
    }
}

// ── Readable output ────────────────────────────────────────────────────────

fn print_model_header(resolved: &hub::ResolvedModel, architecture: &str) {
    let id = resolved
        .model_id
        .clone()
        .unwrap_or_else(|| resolved.dir.display().to_string());
    let mut extras = vec![format!("dir {}", resolved.dir.display())];
    if let Some(rev) = &resolved.revision {
        extras.push(format!("revision {rev}"));
    }
    println!("Model:        {id} ({})", extras.join(", "));
    println!("Architecture: {architecture}");
}

fn print_extraction(extraction: &Extraction, threshold: f32) {
    if let Some(entities) = &extraction.entities {
        if entities.is_empty() {
            println!();
            println!("No entities found above threshold {threshold}.");
        } else {
            println!();
            println!("Entities ({}):", entities.len());
            for e in entities {
                println!(
                    "  {:<16} {:>5.1}%  {:?}  bytes [{}..{})",
                    e.entity_type,
                    e.confidence * 100.0,
                    e.text,
                    e.char_start,
                    e.char_end
                );
            }
        }
    }
    if let Some(relations) = &extraction.relations {
        if relations.is_empty() {
            println!();
            println!("No relations found above threshold {threshold}.");
        } else {
            println!();
            println!("Relations ({}):", relations.len());
            for r in relations {
                println!(
                    "  {:<16} {:>5.1}%  head {:?} bytes [{}..{}) -> tail {:?} bytes [{}..{})",
                    r.relation_type,
                    r.confidence * 100.0,
                    r.head.text,
                    r.head.char_start,
                    r.head.char_end,
                    r.tail.text,
                    r.tail.char_start,
                    r.tail.char_end
                );
            }
        }
    }
}

/// `run` returns Ok(true) on success, Ok(false) when the run completed but
/// reported failures (case mode), Err on any user/input/model error.
fn run(args: &Args) -> Result<bool> {
    let device = parse_device(&args.device)?;

    // ── Resolve model directory ────────────────────────────────────────────
    // --model-dir: local files only, takes precedence, never downloads.
    let resolved = if let Some(dir) = &args.model_dir {
        hub::local_model(&expand_tilde(dir))?
    } else {
        hub::ensure_model(
            &args.model_id,
            args.revision.as_deref(),
            &expand_tilde(&args.models_root),
        )?
    };
    info!("Using model directory: {:?}", resolved.dir);

    // ── Debug: list weight keys ────────────────────────────────────────────
    if let Some(n) = args.list_weights {
        let entries = Model::list_weights(&resolved.dir, n)?;
        println!(
            "Weight keys in model.safetensors (showing {}):",
            entries.len()
        );
        for e in &entries {
            let shape = e
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            println!("  {}  {}  [{}]", e.name, e.dtype, shape);
        }
        return Ok(true);
    }

    // ── Mode selection & request validation (before loading the model) ─────
    let case_path = args.cases.as_deref().map(expand_tilde);
    if case_path.is_some()
        && (args.text.is_some()
            || !args.entities.is_empty()
            || !args.relations.is_empty()
            || args.threshold.is_some())
    {
        bail!(
            "--cases takes text, type lists, and thresholds from the case file — do not \
             combine it with --text/--entities/--relations/--threshold"
        );
    }
    let case_file = match &case_path {
        Some(p) => Some(cases::load(p)?),
        None => {
            // Fail fast on a bad one-off request before the slow model load.
            validate_one_off(&args.text, &args.entities, &args.relations, args.threshold)?;
            None
        }
    };

    // Case files may pin the checkpoint they were written for.
    if let Some(cf) = &case_file {
        if let (Some(want), Some(have)) = (&cf.model_id, &resolved.model_id) {
            if want != have {
                bail!(
                    "case file targets model ID {want:?} but {have:?} is in use — run it \
                     with --model-id {want:?} (or a matching --model-dir)"
                );
            }
        }
    }

    // ── Load model (architecture dispatched from config.json) ──────────────
    info!("Loading model…");
    let model = Model::load(&resolved.dir, &device)
        .with_context(|| format!("loading model from {}", resolved.dir.display()))?;
    let architecture = model.architecture().as_str().to_string();

    // ── Pre-flight: fail clearly on unsupported tasks before any output ────
    let wants_relations = args.relations.iter().any(|r| !r.trim().is_empty())
        || case_file
            .as_ref()
            .is_some_and(|cf| cf.cases.iter().any(|c| !c.relations.is_empty()));
    if wants_relations {
        model.check_relation_support()?;
    }

    match &case_file {
        Some(cf) => run_case_mode(args, &resolved, &architecture, &model, cf),
        None => {
            let (text, entity_types, relation_types, threshold) =
                validate_one_off(&args.text, &args.entities, &args.relations, args.threshold)?;
            let extraction =
                run_extraction(&model, &text, &entity_types, &relation_types, threshold)?;
            let mut tasks = Vec::new();
            if !entity_types.is_empty() {
                tasks.push("entities".to_string());
            }
            if !relation_types.is_empty() {
                tasks.push("relations".to_string());
            }
            if args.json {
                let out = JsonExtraction {
                    schema_version: SCHEMA_VERSION,
                    kind: "extraction",
                    model: json_model(&resolved, &architecture),
                    tasks,
                    threshold,
                    entity_types,
                    relation_types,
                    entities: extraction
                        .entities
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(json_entity)
                        .collect(),
                    relations: extraction
                        .relations
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(json_relation)
                        .collect(),
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).context("serializing JSON output")?
                );
            } else {
                print_model_header(&resolved, &architecture);
                println!("Tasks:        {}", tasks.join(", "));
                println!("Threshold:    {threshold}");
                print_extraction(&extraction, threshold);
            }
            Ok(true)
        }
    }
}

fn run_case_mode(
    args: &Args,
    resolved: &hub::ResolvedModel,
    architecture: &str,
    model: &Model,
    case_file: &cases::CaseFile,
) -> Result<bool> {
    let mut results = Vec::new();
    for (i, case) in case_file.cases.iter().enumerate() {
        let label = case.label(i);
        let threshold = case.threshold();
        let outcome = run_extraction(
            model,
            &case.text,
            &case.entities,
            &case.relations,
            threshold,
        );
        let (outcome, mismatches) = match outcome {
            Ok(o) => {
                let m = cases::evaluate(
                    case,
                    &cases::CaseOutcome {
                        entities: o.entities.clone().unwrap_or_default(),
                        relations: o.relations.clone().unwrap_or_default(),
                    },
                );
                (o, m)
            }
            Err(e) => (
                Extraction {
                    entities: None,
                    relations: None,
                },
                vec![format!("inference error: {e:#}")],
            ),
        };
        results.push((label, outcome, mismatches));
    }

    let failed = results.iter().filter(|(_, _, m)| !m.is_empty()).count();
    let passed = results.len() - failed;

    if args.json {
        let out = JsonCaseReport {
            schema_version: SCHEMA_VERSION,
            kind: "case_report",
            model: json_model(resolved, architecture),
            cases: results
                .iter()
                .map(|(label, outcome, mismatches)| JsonCaseResult {
                    name: label.clone(),
                    status: if mismatches.is_empty() {
                        "pass"
                    } else {
                        "fail"
                    },
                    mismatches: mismatches.clone(),
                    entities: outcome
                        .entities
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(json_entity)
                        .collect(),
                    relations: outcome
                        .relations
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(json_relation)
                        .collect(),
                })
                .collect(),
            passed,
            failed,
            total: results.len(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&out).context("serializing case report")?
        );
    } else {
        print_model_header(resolved, architecture);
        println!(
            "Case file:    {} ({} cases)",
            args.cases.as_ref().unwrap().display(),
            results.len()
        );
        println!();
        for (label, _, mismatches) in &results {
            if mismatches.is_empty() {
                println!("PASS  {label}");
            } else {
                println!("FAIL  {label}");
                for m in mismatches {
                    println!("      {m}");
                }
            }
        }
        println!("---");
        println!("{passed} passed, {failed} failed ({} cases)", results.len());
    }

    Ok(failed == 0)
}

fn parse_device(s: &str) -> Result<candle_core::Device> {
    match s {
        "cpu" => Ok(candle_core::Device::Cpu),
        s if s.starts_with("cuda:") => {
            let idx: usize = s[5..].parse().context("invalid CUDA device index")?;
            Ok(candle_core::Device::new_cuda(idx)?)
        }
        other => anyhow::bail!("unknown device {other:?}; use 'cpu' or 'cuda:N'"),
    }
}
