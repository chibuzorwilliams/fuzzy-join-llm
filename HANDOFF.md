# Handoff / Status — fuzzy-join-llm paper

_Last updated: 2026-06-19_

Working notes for picking this up on another computer. Two locations are involved:

- **Analysis + figures (this git repo):** `~/gh/research/fuzzy-join-llm/`
- **Paper LaTeX (synced folder, not git):** `~/sync/00_current/llm-fuzzy-joins/source/bare_jrnl.tex`

## Authoritative data
- **`results/cv_results.csv`** — threshold-optimized, cross-validated F1/precision/recall (τ swept 0.05–0.95, τ chosen to max F1). **Cite these.** Matches paper Tables S1–S4 and backs every figure.
- **`results/summary.csv`** — fixed-threshold; NOT F1-optimal (TF-IDF lands at P0.92/R0.39). Keep it — it has the only `tp/fp/fn` counts and `cost_usd`/`wall_time_seconds`. Do not use for threshold-sensitive figures/tables.

## Done this round
1. **New results integrated** into the paper from `cv_results.csv` (Claude Haiku, GPT-4o, Claude Sonnet, lexical "token-fallback" prompt).
2. **Positioning:** GPT-4o-mini stays primary; "eight methods" core + Tables S1–S4 unchanged. Claude Haiku = secondary corroborating LLM in the two main figures. Message: "even with another model, the ablation collapse holds."
3. **Paper edits** (`bare_jrnl.tex`): abstract + intro robustness sentences; Methods III-A note + III-C "LLM robustness variants" paragraph; Fig. 2 & 3 captions updated; Results overview now names both LLMs; **new subsection IV-E "Robustness Across Models and Prompts"** with two new figures (Fig. 4 prompt comparison, Fig. 5 model robustness); Discussion + Conclusion limitation paragraph rewritten (prompt-framing threat now tested & refuted). Compiles clean (13 pp, no undefined refs).
4. **`scripts/charts.R` Plot 2 (precision–recall scatter) switched from `summary.csv` → `cv_results.csv`** so its operating points match the F1 figure/tables.

## To do next
- [ ] **Regenerate figures:** `Rscript scripts/charts.R` from repo root. ⚠️ Needs the **Roboto Condensed** font; headless `Rscript` errors `invalid font type` with the default PNG device. Render via `ragg::agg_png` or run in RStudio. Install Roboto Condensed on the new machine first.
- [ ] **Copy regenerated plots** into the paper folder: `cp results/plots/{f1_comparison,precision_recall_scatter,llm_prompt_comparison,llm_robustness}.png ~/sync/00_current/llm-fuzzy-joins/source/` then recompile `bare_jrnl.tex`.
- [ ] Consider whether to mention LLM cost/runtime (available in `summary.csv`) in the paper.
- [ ] Optional: fold Claude Haiku into Tables S1–S4 if reviewers want it as a full method (currently secondary by design).
- [ ] Decide whether `summary.csv` should be renamed to make its fixed-threshold nature obvious (or just rely on this note + README).

## Verifying the paper compiles
```
cd ~/sync/00_current/llm-fuzzy-joins/source
pdflatex bare_jrnl && bibtex bare_jrnl && pdflatex bare_jrnl && pdflatex bare_jrnl
```
