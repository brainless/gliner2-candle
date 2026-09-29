//! Project-local model download and cache contract for `--model-id`
//! (epic `gliner-2.5-boundary-support`, Task 9).
//!
//! Contract:
//! - A model ID maps to exactly one cache directory:
//!   `<models_root>/<name>` where `<name>` is the ID's repo name
//!   (`fastino/gliner2.5-small-v1` → `./models/gliner2.5-small-v1/`).
//! - Each cache directory holds one checkpoint's `config.json`,
//!   `tokenizer.json`, `encoder_config/config.json`, and `model.safetensors`,
//!   plus a `.gliner2-cache.json` entry recording the resolved Hub revision
//!   and each file's identity (size + sha256).
//! - Downloads complete atomically: every file is streamed to a temporary
//!   name and renamed only after a full write, and the cache entry is written
//!   last — an interrupted fetch can never be mistaken for a valid
//!   checkpoint (a directory without a cache entry is rejected unless all
//!   required files are present and adoptable).
//! - Reuse verifies the recorded model ID and revision before touching the
//!   files, then verifies every file's size and sha256. Files from two
//!   different model IDs are never mixed and a different checkpoint is never
//!   silently overwritten: mismatches error with instructions.
//! - `--model-dir` bypasses all of this (local files only, no download).
//!
//! Revision pinning: `--revision <ref>` (commit sha, tag, or branch) or the
//! documented pinned default for the two `fastino` checkpoints (see
//! [`pinned_revision`]); other IDs without `--revision` resolve `main` to a
//! commit sha at download time and record it, so later runs stay reproducible.
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Identity file recorded inside each model-specific cache directory.
pub const CACHE_META_FILE: &str = ".gliner2-cache.json";

/// Files every checkpoint directory must contain (config, tokenizer, encoder
/// config, weights — the four artifacts the loader reads).
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "encoder_config/config.json",
    "model.safetensors",
    "tokenizer.json",
];

/// Documented pinned Hub revisions (DEVELOP.md "Checkpoints and revisions").
/// The legacy span checkpoint is pinned to the revision the regression
/// checks were captured against, not to the moving `main`.
pub fn pinned_revision(model_id: &str) -> Option<&'static str> {
    match model_id {
        "fastino/gliner2-large-v1" => Some("5312584a6fd5543e457ba5f309ac5db226431d1a"),
        "fastino/gliner2.5-small-v1" => Some("7132dc4561c3f94563c6147e75ffa8ef34c4964a"),
        _ => None,
    }
}

/// One recorded file identity: byte size and sha256 (lowercase hex).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    pub size: u64,
    pub sha256: String,
}

/// `.gliner2-cache.json`: the cache entry for one checkpoint directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheMeta {
    pub cache_version: u32,
    pub model_id: String,
    /// Resolved Hub revision (commit sha).
    pub revision: String,
    /// "downloaded" (fetched by this tool) or "adopted" (pre-existing files,
    /// e.g. from a manual `hf download`, hashed locally on first use).
    pub origin: String,
    pub files: BTreeMap<String, FileRecord>,
}

/// A resolved model directory plus the identity it carries, if known.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub dir: PathBuf,
    /// Hub model ID (absent for a plain `--model-dir` without cache entry).
    pub model_id: Option<String>,
    /// Resolved Hub revision (absent when unknown).
    pub revision: Option<String>,
}

const CACHE_VERSION: u32 = 1;

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .build()
}

fn authorized(req: ureq::Request) -> ureq::Request {
    match std::env::var("HF_TOKEN") {
        Ok(tok) if !tok.is_empty() => req.set("Authorization", &format!("Bearer {tok}")),
        _ => req,
    }
}

fn is_commit_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Repo-name part of a model ID (`fastino/gliner2-large-v1` →
/// `gliner2-large-v1`); the stable ID → directory mapping.
pub fn model_dir_name(model_id: &str) -> Result<&str> {
    let name = model_id.rsplit('/').next().unwrap_or("");
    if model_id.is_empty()
        || name.is_empty()
        || model_id.starts_with('/')
        || model_id.ends_with('/')
        || name == "."
        || name == ".."
    {
        bail!(
            "invalid model ID {model_id:?} — expected a HuggingFace ID like \
             \"fastino/gliner2.5-small-v1\""
        );
    }
    Ok(name)
}

/// Hash a file's content (sha256, lowercase hex) and size.
pub fn hash_file(path: &Path) -> Result<FileRecord> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("opening {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let size = std::io::copy(&mut f, &mut hasher)
        .with_context(|| format!("reading {} for hashing", path.display()))?;
    Ok(FileRecord {
        size,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn write_cache_meta(dir: &Path, meta: &CacheMeta) -> Result<()> {
    let json = serde_json::to_vec_pretty(meta).context("serializing cache metadata")?;
    let final_path = dir.join(CACHE_META_FILE);
    let tmp = dir.join(format!(".{CACHE_META_FILE}.tmp-{}", std::process::id()));
    std::fs::write(&tmp, &json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &final_path)
        .with_context(|| format!("committing {}", final_path.display()))?;
    Ok(())
}

fn read_cache_meta(dir: &Path) -> Result<CacheMeta> {
    let path = dir.join(CACHE_META_FILE);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let meta: CacheMeta = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "cache metadata {} is corrupt — remove {} to re-download",
            path.display(),
            dir.display()
        )
    })?;
    if meta.cache_version != CACHE_VERSION {
        bail!(
            "cache metadata {} has unsupported cache_version {} (this build writes version \
             {CACHE_VERSION}) — remove {} to re-download",
            path.display(),
            meta.cache_version,
            dir.display()
        );
    }
    Ok(meta)
}

/// Resolve a revision reference (commit sha / tag / branch) to a commit sha
/// via the Hub API. Commit shas pass through without a network call.
pub fn resolve_revision(model_id: &str, rev_ref: &str) -> Result<String> {
    if is_commit_sha(rev_ref) {
        return Ok(rev_ref.to_string());
    }
    let url = format!(
        "{}/api/models/{}/revision/{}",
        hf_endpoint(),
        model_id,
        rev_ref
    );
    let resp = match authorized(agent().get(&url)).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(404, _)) => bail!(
            "unknown revision {rev_ref:?} for model {model_id} (HTTP 404) — pass a valid \
             --revision (commit sha, tag, or branch)"
        ),
        Err(ureq::Error::Status(code, _)) => {
            bail!("could not resolve revision {rev_ref:?} of {model_id}: HTTP {code}")
        }
        Err(ureq::Error::Transport(t)) => {
            bail!("network error resolving revision {rev_ref:?} of {model_id}: {t}")
        }
    };
    let value: serde_json::Value =
        serde_json::from_reader(resp.into_reader()).context("parsing Hub revision response")?;
    let sha = value
        .get("sha")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("Hub revision response for {model_id}@{rev_ref} has no \"sha\""))?;
    if !is_commit_sha(sha) {
        bail!("Hub returned a non-sha revision for {model_id}@{rev_ref}: {sha:?}");
    }
    Ok(sha.to_string())
}

/// Stream `resolve/{revision}/{rel}` into `dest` atomically (temp file +
/// rename after a full write), returning the file's identity.
fn download_file(model_id: &str, revision: &str, rel: &str, dest: &Path) -> Result<FileRecord> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        hf_endpoint(),
        model_id,
        revision,
        rel
    );
    let resp = match authorized(agent().get(&url)).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(404, _)) => {
            bail!("file {rel} is not available at revision {revision} of {model_id} (HTTP 404)")
        }
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => bail!(
            "access denied fetching {rel} from {model_id}@{revision} (HTTP {code}) — set \
             HF_TOKEN if the model is private"
        ),
        Err(ureq::Error::Status(code, _)) => {
            bail!("could not fetch {rel} from {model_id}@{revision}: HTTP {code}")
        }
        Err(ureq::Error::Transport(t)) => {
            bail!("network error fetching {rel} from {model_id}@{revision}: {t}")
        }
    };

    let parent = dest
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", dest.display()))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}-{}",
        dest.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let result = (|| -> Result<FileRecord> {
        let mut reader = resp.into_reader();
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?,
        );
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut size: u64 = 0;
        loop {
            let n = reader
                .read(&mut buf)
                .with_context(|| format!("downloading {rel} from {model_id}@{revision}"))?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut file, &buf[..n])
                .with_context(|| format!("writing {}", tmp.display()))?;
            hasher.update(&buf[..n]);
            size += n as u64;
        }
        std::io::Write::flush(&mut file).with_context(|| format!("flushing {}", tmp.display()))?;
        file.get_ref()
            .sync_all()
            .with_context(|| format!("syncing {}", tmp.display()))?;
        drop(file);
        std::fs::rename(&tmp, dest).with_context(|| format!("committing {}", dest.display()))?;
        Ok(FileRecord {
            size,
            sha256: format!("{:x}", hasher.finalize()),
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Verify every required file against the recorded cache entry (existence,
/// byte size, and content hash).
fn verify_files(dir: &Path, meta: &CacheMeta) -> Result<()> {
    for rel in REQUIRED_FILES {
        let path = dir.join(rel);
        let record = meta.files.get(*rel).ok_or_else(|| {
            anyhow!(
                "cache entry {} has no identity recorded for {rel} — remove {} to re-download",
                dir.join(CACHE_META_FILE).display(),
                dir.display()
            )
        })?;
        if !path.exists() {
            bail!(
                "cached checkpoint {} is incomplete: {rel} is missing (interrupted download?) \
                 — remove {} to re-download",
                meta.model_id,
                dir.display()
            );
        }
        let actual = hash_file(&path)?;
        if actual.size != record.size {
            bail!(
                "cached file {} has {} bytes but {} were recorded (truncated or modified) — \
                 remove {} to re-download",
                path.display(),
                actual.size,
                record.size,
                dir.display()
            );
        }
        if actual.sha256 != record.sha256 {
            bail!(
                "cached file {} does not match its recorded sha256 (corrupted or modified) — \
                 remove {} to re-download",
                path.display(),
                dir.display()
            );
        }
    }
    Ok(())
}

/// `--model-id` entry point: return a complete, identity-verified model
/// directory, downloading it first if needed.
///
/// `revision` is the explicit `--revision` value, if any. For the documented
/// `fastino` checkpoints the pinned default is enforced on reuse as well; an
/// unknown ID without `--revision` is downloaded at `main` and pinned to the
/// resolved commit sha by its cache entry.
pub fn ensure_model(
    model_id: &str,
    revision: Option<&str>,
    models_root: &Path,
) -> Result<ResolvedModel> {
    model_dir_name(model_id)?;
    let dir = models_root.join(model_dir_name(model_id)?);
    let meta_path = dir.join(CACHE_META_FILE);

    if meta_path.exists() {
        let meta = read_cache_meta(&dir)?;
        if meta.model_id != model_id {
            bail!(
                "model directory {} holds the checkpoint for {:?} but --model-id {model_id:?} \
                 was requested — refusing to mix checkpoints from different IDs (use a \
                 different --models-root or remove the directory)",
                dir.display(),
                meta.model_id
            );
        }
        // The requested revision must match the recorded one: an explicit
        // --revision, or the documented pin when there is one.
        let requested = revision
            .map(|r| r.to_string())
            .or_else(|| pinned_revision(model_id).map(|r| r.to_string()));
        if let Some(req) = requested {
            if !is_commit_sha(&req) {
                let req = resolve_revision(model_id, &req)?;
                // A moving ref cannot be verified offline; compare after
                // resolution only for exact recorded shas.
                if req != meta.revision {
                    bail!(
                        "cached checkpoint {} is at revision {} but {req} resolves from the \
                         Hub now — remove {} (or pass --revision {}) to switch",
                        meta.model_id,
                        meta.revision,
                        dir.display(),
                        meta.revision
                    );
                }
            } else if req != meta.revision {
                bail!(
                    "cached checkpoint {} is at revision {} but revision {req} was requested \
                     — remove {} to re-download, or pass --revision {}",
                    meta.model_id,
                    meta.revision,
                    dir.display(),
                    meta.revision
                );
            }
        }
        verify_files(&dir, &meta)?;
        return Ok(ResolvedModel {
            dir,
            model_id: Some(meta.model_id),
            revision: Some(meta.revision),
        });
    }

    // No cache entry.
    let have: Vec<&&str> = REQUIRED_FILES
        .iter()
        .filter(|f| dir.join(f).exists())
        .collect();
    if !have.is_empty() {
        if have.len() != REQUIRED_FILES.len() {
            let missing: Vec<&str> = REQUIRED_FILES
                .iter()
                .filter(|f| !dir.join(f).exists())
                .copied()
                .collect();
            bail!(
                "model directory {} has no cache entry and is incomplete (missing: {}) — \
                 interrupted download or foreign files? remove {} and retry",
                dir.display(),
                missing.join(", "),
                dir.display()
            );
        }
        // Adopt a complete pre-existing directory (e.g. manual `hf download`):
        // record its identity so later runs verify it, without re-downloading.
        let revision = revision
            .map(|r| r.to_string())
            .or_else(|| pinned_revision(model_id).map(|r| r.to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        let mut files = BTreeMap::new();
        for rel in REQUIRED_FILES {
            files.insert((*rel).to_string(), hash_file(&dir.join(rel))?);
        }
        let meta = CacheMeta {
            cache_version: CACHE_VERSION,
            model_id: model_id.to_string(),
            revision: revision.clone(),
            origin: "adopted".to_string(),
            files,
        };
        write_cache_meta(&dir, &meta)?;
        tracing::info!(
            "adopted pre-existing model directory {} for {model_id}@{revision} (identity \
             recorded locally; the files were not re-fetched from the Hub)",
            dir.display()
        );
        return Ok(ResolvedModel {
            dir,
            model_id: Some(model_id.to_string()),
            revision: Some(revision),
        });
    }

    // Fresh download into an empty (or absent) directory.
    let revision = match revision {
        Some(r) => resolve_revision(model_id, r)?,
        None => match pinned_revision(model_id) {
            Some(r) => r.to_string(),
            None => resolve_revision(model_id, "main")?,
        },
    };
    let preexisting = dir.exists();
    if !preexisting {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut downloaded: Vec<PathBuf> = Vec::new();
    let mut files = BTreeMap::new();
    for rel in REQUIRED_FILES {
        let dest = dir.join(rel);
        tracing::info!(
            "downloading {model_id}@{revision}:{rel} → {}",
            dest.display()
        );
        match download_file(model_id, &revision, rel, &dest) {
            Ok(record) => {
                downloaded.push(dest);
                files.insert((*rel).to_string(), record);
            }
            Err(e) => {
                for p in &downloaded {
                    let _ = std::fs::remove_file(p);
                }
                return Err(e).with_context(|| {
                    format!(
                        "downloading {model_id} into {} failed — the directory was left \
                         empty; fix the error and retry",
                        dir.display()
                    )
                });
            }
        }
    }
    let meta = CacheMeta {
        cache_version: CACHE_VERSION,
        model_id: model_id.to_string(),
        revision: revision.clone(),
        origin: "downloaded".to_string(),
        files,
    };
    write_cache_meta(&dir, &meta)?;
    Ok(ResolvedModel {
        dir,
        model_id: Some(model_id.to_string()),
        revision: Some(revision),
    })
}

/// `--model-dir` entry point: local files only — never downloads, never
/// hashes. Fails clearly when required files are missing; reads the cache
/// entry for identity display when present.
pub fn local_model(dir: &Path) -> Result<ResolvedModel> {
    let missing: Vec<&str> = REQUIRED_FILES
        .iter()
        .filter(|f| !dir.join(f).exists())
        .copied()
        .collect();
    if !missing.is_empty() {
        bail!(
            "model directory {} is missing: {} — expected config.json, \
             encoder_config/config.json, model.safetensors, and tokenizer.json",
            dir.display(),
            missing.join(", ")
        );
    }
    let meta = if dir.join(CACHE_META_FILE).exists() {
        Some(read_cache_meta(dir)?)
    } else {
        None
    };
    Ok(ResolvedModel {
        dir: dir.to_path_buf(),
        model_id: meta.as_ref().map(|m| m.model_id.clone()),
        revision: meta.as_ref().map(|m| m.revision.clone()),
    })
}
