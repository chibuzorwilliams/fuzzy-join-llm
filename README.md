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

**The LLM's clean-text advantage depends on semantic content the task does not require.** Once token-based methods are properly thresholded, the LLM's edge on clean data is modest (Abt-Buy) or negligible (Amazon-Google), and under ablation the LLM collapses while token methods stay flat.

**Abt-Buy F1:**
| Transformation | LLM | TF-IDF | Soft TF-IDF |
|---|---|---|---|
| Original | **0.928** | 0.761 | 0.742 |
| Ciphered Letters | 0.779 | 0.747 | 0.731 |
| Ciphered Words | 0.511 | **0.751** | 0.743 |
| Scrambled | 0.603 | 0.746 | **0.763** |

**Amazon-Google F1:**
| Transformation | LLM | TF-IDF |
|---|---|---|
| Original | **0.822** | 0.808 |
| Ciphered Letters | 0.585 | **0.808** |
| Ciphered Words | 0.576 | **0.807** |
| Scrambled | 0.556 | **0.808** |

**Key insight:** TF-IDF blocking keeps the true match in the candidate set for ~99% of records *even under corruption*, and a purely lexical scorer still solves those cases -- yet the LLM cannot. The matching evidence is present and preserved; the LLM fails because it relies on meaning the task does not require.

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
