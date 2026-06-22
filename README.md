# Matching Without Meaning: LLM Entity Matchers Rely on Non-Essential Semantic Priors

**Systematic comparison of 8 entity matching methods across controlled semantic ablations**

---

## Research Question

**Does an LLM entity matcher rely on the features that actually distinguish entities, or on semantic priors the task does not require?**

We use *controlled semantic ablation*: structure-preserving transformations that remove meaning while keeping the discriminative tokens (and their cross-catalog identity) fully intact. If a method depends only on token distinctiveness, it should be unaffected; if it depends on meaning, it should collapse.

1. **Original** - Baseline
2. **Ciphered Letters** - Consistent character substitution applied to both catalogs (e.g., 'a'->'x'). Preserves token identity exactly.
3. **Ciphered Words** - Consistent vocabulary replacement (e.g., 'Sony'->'Goat' in both catalogs).
4. **Scrambled** - Deterministic per-word letter scrambling (first/last letter kept), identical across catalogs.

---

## Key Findings

**The LLM's clean-text advantage depends on semantic content the task does not require.** Once token-based methods are properly thresholded, the LLM's edge on clean data is modest (Abt-Buy) or negligible (Amazon-Google), and under ablation the LLM collapses while token methods stay flat. A method's fragility tracks how much it depends on pre-trained meaning, not its architecture or scale -- the smallest neural model (SentenceTransformer) is the *most* fragile, and the far larger LLM is no more robust.

### F1 by transformation (all 8 methods)

**Abt-Buy** (rich title + description):
| Method | Original | Ciphered Letters | Ciphered Words | Scrambled |
|---|---|---|---|---|
| LLM (GPT-4o-mini) | **0.928** | **0.779** | 0.511 | 0.603 |
| OpenAI Embeddings | 0.799 | 0.409 | 0.422 | 0.508 |
| TF-IDF | 0.761 | 0.747 | **0.751** | 0.746 |
| Soft TF-IDF | 0.742 | 0.731 | 0.743 | **0.763** |
| SentenceTransformer | 0.628 | 0.164 | 0.376 | 0.362 |
| Monge-Elkan | 0.381 | 0.381 | 0.289 | 0.382 |
| Levenshtein | 0.164 | 0.164 | 0.120 | 0.173 |
| Jaro-Winkler | 0.085 | 0.085 | 0.052 | 0.080 |

**Amazon-Google** (short title only):
| Method | Original | Ciphered Letters | Ciphered Words | Scrambled |
|---|---|---|---|---|
| LLM (GPT-4o-mini) | **0.822** | 0.585 | 0.576 | 0.556 |
| TF-IDF | 0.808 | **0.808** | **0.807** | **0.808** |
| OpenAI Embeddings | 0.802 | 0.639 | 0.701 | 0.725 |
| SentenceTransformer | 0.755 | 0.619 | 0.683 | 0.676 |
| Monge-Elkan | 0.740 | 0.740 | 0.700 | 0.735 |
| Soft TF-IDF | 0.650 | 0.650 | 0.678 | 0.683 |
| Levenshtein | 0.563 | 0.563 | 0.535 | 0.569 |
| Jaro-Winkler | 0.488 | 0.488 | 0.483 | 0.489 |

### Precision / Recall / F1 on original (clean) text

**Abt-Buy:**
| Method | Precision | Recall | F1 |
|---|---|---|---|
| LLM (GPT-4o-mini) | 0.967 | 0.892 | **0.928** |
| OpenAI Embeddings | 0.799 | 0.798 | 0.799 |
| TF-IDF | 0.762 | 0.759 | 0.761 |
| Soft TF-IDF | 0.742 | 0.742 | 0.742 |
| SentenceTransformer | 0.629 | 0.628 | 0.628 |
| Monge-Elkan | 0.525 | 0.299 | 0.381 |
| Levenshtein | 0.171 | 0.156 | 0.164 |
| Jaro-Winkler | 0.266 | 0.051 | 0.085 |

**Amazon-Google:**
| Method | Precision | Recall | F1 |
|---|---|---|---|
| LLM (GPT-4o-mini) | 0.876 | 0.774 | **0.822** |
| TF-IDF | 0.808 | 0.808 | 0.808 |
| OpenAI Embeddings | 0.803 | 0.801 | 0.802 |
| SentenceTransformer | 0.759 | 0.751 | 0.755 |
| Monge-Elkan | 0.793 | 0.694 | 0.740 |
| Soft TF-IDF | 0.650 | 0.650 | 0.650 |
| Levenshtein | 0.684 | 0.478 | 0.563 |
| Jaro-Winkler | 0.564 | 0.430 | 0.488 |

### Failure is in reasoning, not retrieval

TF-IDF blocking keeps the true match in the candidate set for ~99% of records *even under corruption*, and a purely lexical scorer still solves those cases -- yet the LLM cannot.

- **Blocking recall** (top-50 candidates): Abt-Buy **99.7%** (1,078/1,081), Amazon-Google **99.0%** (1,102/1,113). Stays high under every ablation (99.5-99.7% / 98.5-99.0%). `k = 50` chosen by sweep (Abt-Buy: k=25 -> 97.2%, k=50 -> 99.7%, k=100 -> 99.8%).
- **LLM recall under Ciphered Words** drops to **0.418** (Abt-Buy) and **0.456** (Amazon-Google) despite blocking recall staying near 0.99 -- the candidates are present; the LLM fails to select them once words are no longer meaningful.

### Robustness holds across models (clean-text F1)

Larger/stronger models score slightly higher on clean text but are *no more robust* under ablation.

| Model | Abt-Buy | Amazon-Google |
|---|---|---|
| GPT-4o-mini (primary) | 0.928 | 0.822 |
| GPT-4o | 0.955 | 0.904 |
| Claude Haiku 3.5 | 0.961 | 0.895 |
| Claude Sonnet 3.5 | 0.954 | -- (run abandoned, billing) |

### Prompt sensitivity (token-fallback prompt)

A prompt that explicitly instructs the model to match on identifier tokens gives mixed results and never closes the gap to TF-IDF:
- **GPT-4o-mini** improves on short titles (Amazon-Google Scrambled 0.555 -> 0.707) and helps unevenly on Abt-Buy (Ciphered Words 0.510 -> 0.557, Scrambled 0.601 -> 0.632) but *hurts* under Ciphered Letters (0.779 -> 0.703).
- **Claude Haiku** collapses to near-zero F1 on Ciphered Words/Scrambled under the token-fallback prompt.
- Even the best case (GPT-4o-mini, 0.633-0.707) stays well below TF-IDF (0.807-0.808).

### Cost & latency (per dataset)

| Model | Cost | Wall time |
|---|---|---|
| GPT-4o-mini | <$1 (~$0.50 Abt-Buy, ~$1.00 Amazon-Google) | 30-80 min |
| Claude Haiku 3.5 | $3-9 | hours |
| GPT-4o | $7-22 | 1-6 h |
| Claude Sonnet 3.5 | $10-13 | 1-6 h |
| All non-LLM methods | $0 | seconds |

The cost-benefit case for LLM matching is narrow: it is justified only on clean, description-rich text where the modest F1 gain (0.928 vs 0.761) outweighs orders-of-magnitude differences in cost and latency.

> **Note on results:** earlier versions of this repo reported lower TF-IDF/Soft TF-IDF F1 because (a) the threshold sweep started at 0.50, which is above the optimal operating point for cosine/edit-distance methods, and (b) the "Soft TF-IDF" method lacked IDF weighting. Both are fixed (`scripts/methods.py`); the threshold sweep now runs the full 0.05-0.95 grid and Soft TF-IDF is the real IDF-weighted, Jaro-Winkler version. The previous `summary.csv` is preserved as `summary_OLD_threshold0.50_brokenSoftTFIDF.csv`.

---

## Methods Evaluated

### Character-Based
- **Jaro-Winkler**: Edit distance with prefix weighting
- **Levenshtein**: Minimum edit distance
- **Monge-Elkan**: Token-level Jaro-Winkler averaging

### Token-Based
- **TF-IDF**: Cosine similarity on term frequency vectors
- **Soft TF-IDF**: IDF-weighted TF-IDF with Jaro-Winkler fuzzy token matching (threshold >= 0.9)

### Neural Embeddings
- **SentenceTransformer**: all-MiniLM-L6-v2
- **OpenAI Embeddings**: text-embedding-3-small

### Large Language Model
- **GPT-4o-mini** (primary): TF-IDF blocking (top-50 candidates), temperature = 0
- Additional models tested for robustness: GPT-4o, Claude Sonnet 3.5, Claude Haiku 3.5

---

## Datasets

Both datasets are filtered to records with at least one ground-truth match, following standard benchmark practice.

- **Abt-Buy** (1,081 matched pairs): Title + full product description. Rich, heterogeneous text.
- **Amazon-Google** (1,113 matched pairs): Title only. Short, keyword-dense text.

---

## Evaluation Protocol

- **Threshold optimization**: Each method's decision threshold is swept over tau in {0.05, 0.10, ..., 0.95}; the threshold maximizing F1 is reported.
- **Cross-validation**: 5-fold CV confirms threshold selection does not overfit -- threshold is chosen on 4 training folds and evaluated on the held-out fold (seed = 42). Results are in `results/cv_results.csv` with mean and std columns.
- **LLM blocking**: TF-IDF cosine similarity retrieves top-50 candidates per record before LLM classification. Blocking recall is ~99% on both datasets under all conditions.

---

## Reproducing the Results

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy scikit-learn tqdm
pip install jellyfish sentence-transformers openai anthropic

# R (for figures)
# Requires: ggplot2, cowplot, readr, dplyr, showtext (+ Roboto Condensed font)
```

API keys needed for LLM/embedding methods:

```bash
export OPENAI_API_KEY=...      # GPT-4o-mini, GPT-4o, OpenAI embeddings
export ANTHROPIC_API_KEY=...   # Claude Sonnet, Claude Haiku
```

### Step 1: Prepare transformed datasets

```bash
python scripts/prepare_datasets.py --dataset all
```

Creates `data/{dataset}/data_test/` with ciphered_letters, ciphered_words, and scrambled variants.

### Step 2: Run experiments

```bash
# All methods on all transformations and datasets
python scripts/run_experiments.py --dataset all --methods all --transformations all

# Or run selectively
python scripts/run_experiments.py --dataset abt-buy --methods llm,tfidf --transformations original
```

Results are saved as parquet files in `results/{dataset}/{method}/`.

### Step 3: Cross-validate thresholds

```bash
python scripts/cv_threshold.py
```

Produces `results/cv_results.csv` (F1 mean/std, precision, recall for all 128 configurations).

### Step 4: Generate summary table

```bash
python scripts/analyze_results.py
```

Produces `results/summary.csv` with TP/FP/FN counts and (for LLM methods) cost and wall time.

### Step 5: Generate figures

```bash
Rscript scripts/charts.R
```

Produces PNG plots in `results/plots/`. Requires the Roboto Condensed font; use `ragg::agg_png` or run in RStudio if headless rendering fails.

---

## Project Structure

```
fuzzy-join-llm/
├── data/
│   ├── abt-buy/
│   │   ├── data_original/           # Source CSVs + ground-truth mapping
│   │   └── data_test/               # Transformed variants
│   └── amazon-google/
│       ├── data_original/
│       └── data_test/
│
├── scripts/
│   ├── prepare_datasets.py          # Generate text transformations
│   ├── methods.py                   # All 8 matching methods
│   ├── run_experiments.py           # Main experiment runner
│   ├── cv_threshold.py              # 5-fold cross-validated threshold selection
│   ├── analyze_results.py           # Aggregate results + summary CSV
│   ├── charts.R                     # Publication figures (ggplot2)
│   ├── diagnose_blocking.py         # Blocking quality analysis
│   ├── run_sonnet_robustness.py     # Extended LLM robustness runs
│   └── test_haiku_mini.py           # Haiku testing script
│
├── results/
│   ├── {dataset}/{method}/          # Per-method parquet files
│   ├── summary.csv                  # Fixed-threshold results with TP/FP/FN and cost
│   ├── cv_results.csv               # Cross-validated F1 (authoritative for figures/tables)
│   └── plots/                       # Generated figures
│
└── README.md
```

---

## Key Results Files

- **`results/cv_results.csv`** -- Threshold-optimized, cross-validated F1/precision/recall. **Use this for figures and tables.**
- **`results/summary.csv`** -- Fixed-threshold results. Has TP/FP/FN counts and LLM `cost_usd`/`wall_time_seconds`, but thresholds are not F1-optimal. Use for cost/runtime analysis only.
