mod config;
mod count_lstm;
mod inference;
mod model;
mod processor;
mod span_rep;

use anyhow::{Context, Result};
use clap::Parser;
use model::GLiNER2;
use processor::Preprocessor;
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    name = "gliner2-candle",
    about = "GLiNER2 entity extraction — Candle backend",
    version
)]
struct Args {
    /// Path to a local model directory (config.json, model.safetensors, tokenizer.json,
    /// encoder_config/config.json).  Takes priority over --model-id.
    /// Example: --model-dir ~/Projects/gliner2-burn/model
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// HuggingFace model ID — used to download files when --model-dir is not given.
    #[arg(long, default_value = "fastino/gliner2-large-v1")]
    model_id: String,

    /// Text to extract entities from
    #[arg(long, default_value = "")]
    text: String,

    /// Comma-separated entity types to extract
    #[arg(long, value_delimiter = ',')]
    entities: Vec<String>,

    /// Confidence threshold (0.0–1.0)
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Device: "cpu" or "cuda:N"
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Print the first N weight keys and exit (for debugging prefix mismatches)
    #[arg(long)]
    list_weights: Option<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let device = parse_device(&args.device)?;

    // ── Resolve model directory ───────────────────────────────────────────────
    let model_dir: PathBuf = if let Some(dir) = args.model_dir {
        // Local path given directly — expand ~ if present
        let expanded = shellexpand::tilde(&dir.to_string_lossy()).into_owned();
        PathBuf::from(expanded)
    } else {
        // Download via HF Hub
        download_from_hub(&args.model_id)?
    };

    info!("Using model directory: {:?}", model_dir);

    // ── Debug: list weight keys ───────────────────────────────────────────────
    if let Some(n) = args.list_weights {
        let keys = GLiNER2::list_weight_keys(&model_dir, n)?;
        println!("First {n} weight keys in model.safetensors:");
        for k in &keys {
            println!("  {k}");
        }
        return Ok(());
    }

    // ── Load model ───────────────────────────────────────────────────────────
    info!("Loading model…");
    let model = GLiNER2::load(&model_dir, &device)?;

    // ── Preprocess ───────────────────────────────────────────────────────────
    let tokenizer_path = model_dir.join("tokenizer.json");
    let preprocessor = Preprocessor::from_file(&tokenizer_path)?;
    let input = preprocessor.process(&args.text, &args.entities)?;

    info!(
        "Input: {} tokens ({} words, {} entity types)",
        input.input_ids.len(),
        input.words.len(),
        input.entity_types.len()
    );

    // ── Inference ────────────────────────────────────────────────────────────
    info!("Running inference…");
    let entities = model.predict(&input, args.threshold)?;

    // ── Results ──────────────────────────────────────────────────────────────
    if entities.is_empty() {
        println!("No entities found above threshold {:.2}", args.threshold);
    } else {
        println!("Extracted entities:");
        for e in &entities {
            println!(
                "  [{:>5}:{:<5}]  {:20}  {:>6.1}%  \"{}\"",
                e.char_start,
                e.char_end,
                e.entity_type,
                e.confidence * 100.0,
                e.text
            );
        }
    }

    Ok(())
}

fn download_from_hub(model_id: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;

    info!("Fetching {} from HuggingFace Hub…", model_id);
    let api = Api::new().context("failed to initialise HF Hub client")?;
    let repo = api.model(model_id.to_string());

    let sentinel = repo
        .get("config.json")
        .with_context(|| format!("could not fetch config.json from {model_id}"))?;

    let dir = sentinel
        .parent()
        .expect("config.json has no parent dir")
        .to_path_buf();

    for file in &[
        "tokenizer.json",
        "model.safetensors",
        "encoder_config/config.json",
    ] {
        repo.get(file)
            .with_context(|| format!("could not fetch {file} from {model_id}"))?;
    }

    Ok(dir)
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
