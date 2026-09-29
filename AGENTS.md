# AGENTS.md

Instructions for coding agents (Claude Code, Codex, opencode, etc.) working in this repo.
For architecture and usage docs, see `DEVELOP.md` and `README.md` — read those first.

## What this is

A pure-Rust port of GLiNER2 entity extraction using Candle, targeting the
`fastino/gliner2-*` model family. Single-crate binary, no workspace, no tests directory yet.

## Build / run / check

```bash
# Toolchain is pinned to stable via rust-toolchain.toml — no action needed, cargo picks it up.
cargo build --release
cargo check                 # fast iteration
cargo clippy --all-targets
cargo fmt

# Requires a downloaded model to actually run (not vendored):
hf download fastino/gliner2-large-v1 --local-dir ./model
cargo run --release -- --model-dir ./model --text "Apple was founded by Steve Jobs in Cupertino." --entities "person,organization,location"
```

There is no test suite. Verify changes by running inference against `./model` (download once,
reuse across sessions) and checking the extracted spans look correct, not just that the binary
exits 0.

## Working on this codebase

- This is a from-scratch numerical port of a PyTorch model. Bugs here are silent — wrong tensor
  shapes, wrong index_select axes, or off-by-one span boundaries produce a program that runs and
  emits *plausible but wrong* entities. Never assume a change is correct because it compiles and
  produces output; run it against the quick-start example and sanity-check the spans against
  what the sentence actually says.
- When touching `src/model.rs`, `src/span_rep.rs`, `src/count_lstm.rs`, or `src/processor.rs`,
  cross-check against the weight key layout and MLP indexing conventions documented in
  `DEVELOP.md` ("Weight key prefixes" / MLP Sequential layout section). Getting the `0`/`3` vs
  `0`/`2` Linear-layer index wrong is a documented past bug class.
- If you change how weights are loaded or renamed, verify with
  `cargo run -- --model-dir ./model --list-weights 50` before assuming the mapping still holds.
- `clf1`/`clf2` (binary span classifier) is loaded but intentionally not wired into scoring —
  don't wire it in as a "fix" unless the task specifically asks for it.
- Known incomplete areas (see DEVELOP.md "Known limitations"): only entity extraction is
  implemented (no relations/classification/JSON structures), no batching, no GPU testing. Don't
  silently expand scope into these unless asked.
- Keep `DEVELOP.md` in sync when you change architecture, source layout, or weight-key
  assumptions — it's the primary reference for future agent sessions, not just humans.

## Git

- No CI configured. `cargo fmt` and `cargo clippy` before committing is on you.
- Commit messages in this repo are terse and describe the fix/change directly (see `git log`).
