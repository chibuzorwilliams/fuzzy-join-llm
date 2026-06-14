# Entity Matching: A Comprehensive Evaluation of Traditional, Neural, and LLM Based Methods

**Systematic comparison of 8 entity matching methods across different text transformations**

---

##  Research Question

**Does an LLM entity matcher rely on the features that actually distinguish entities, or on semantic priors the task does not require?**

We use *controlled semantic ablation*: structure-preserving transformations that remove meaning while keeping the discriminative tokens (and their cross-catalog identity) fully intact. If a method depends only on token distinctiveness, it should be unaffected; if it depends on meaning, it should collapse.

1. **Original** - Baseline
2. **Ciphered letters** - Consistent character substitution applied to both catalogs (e.g., 'a'→'x'). Preserves token identity exactly.
3. **Ciphered words** - Consistent vocabulary replacement (e.g., 'Sony'→'Goat' in both catalogs).
4. **Scrambled** - Deterministic per-word letter scrambling (first/last letter kept), identical across catalogs.

---

##  Key Findings

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

**Key insight:** TF-IDF blocking keeps the true match in the candidate set for ~99% of records *even under corruption*, and a purely lexical scorer still solves those cases — yet the LLM cannot. The matching evidence is present and preserved; the LLM fails because it relies on meaning the task does not require.

> **Note on results:** earlier versions of this repo reported lower TF-IDF/Soft TF-IDF F1 because (a) the threshold sweep started at 0.50, which is above the optimal operating point for cosine/edit-distance methods, and (b) the "Soft TF-IDF" method lacked IDF weighting. Both are fixed (`scripts/methods.py`); the threshold sweep now runs the full 0.05–0.95 grid and Soft TF-IDF is the real IDF-weighted, Jaro-Winkler version. The previous `summary.csv` is preserved as `summary_OLD_threshold0.50_brokenSoftTFIDF.csv`.

---

##  Methods Evaluated

### **1. Character Based Methods**
- **Jaro-Winkler**: Edit distance with prefix weighting
- **Levenshtein**: Minimum edit distance
- **Monge-Elkan**: Token level Jaro-Winkler averaging

### **2. Token-Based Methods**
- **TF-IDF**: Cosine similarity on term frequency vectors
- **Soft TF-IDF**: TF-IDF with fuzzy token matching

### **3. Neural Embedding Methods**
- **SentenceTransformer**: all-MiniLM-L6-v2 
- **OpenAI Embeddings**: text-embedding-3-small (paid)

### **4. Large Language Model**
- **GPT-4o-mini**: With TF-IDF blocking 

---

##  Project Structure

```
fuzzy-join-llm/
├── data/
│   └── abt-buy/
│       ├── data_original/           # Original Abt-Buy dataset
│       ├── data_ciphered_letters/   # Consistent character substitution
│       ├── data_ciphered_words/     # Vocabulary replacement
│       └── data_scrambled/          # Letter scrambling
│
├── scripts/
│   ├── prepare_datasets.py          # Data transformation pipeline
│   ├── methods.py                   # All 8 matching methods
│   ├── run_experiments.py           # Main experiment runner
│   ├── analyze_results.py           # Results analysis & visualization
│   └── diagnose_blocking.py         # Blocking quality diagnostic
│
├── results/
│   └── abt-buy/
│       ├── jaro_winkler/            # Results per method
│       ├── levenshtein/
│       ├── monge_elkan/
│       ├── tfidf/
│       ├── soft_tfidf/
│       ├── sentence_transformer/
│       ├── openai_embeddings/
│       └── llm/                     # LLM results (GPT-4o-mini)
│
└── README.md
```

---

##  Quick Start

### **Prerequisites**

```bash
# Python 3.8+
pip install pandas numpy scikit-learn tqdm
pip install jellyfish sentence-transformers openai
```

### **Step 1: Prepare Data**

Transform the original dataset into privacy-preserving variants:

```bash
python scripts/prepare_datasets.py
```

**Creates:**
- `data_ciphered_letters/` - Consistent character cipher
- `data_ciphered_words/` - Word-level vocabulary replacement
- `data_scrambled/` - Letter scrambling within words

### **Step 2: Run Experiments**

**Option A: Run ALL methods on ALL transformations**
```bash
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods jaro_winkler,levenshtein,monge_elkan,soft_tfidf,tfidf,sentence_transformer,openai_embeddings,llm \
  --transformations original,ciphered_letters,ciphered_words,scrambled
```

**Option B: Run specific methods (e.g., just LLM)**
```bash
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods llm \
  --transformations original,ciphered_letters,ciphered_words,scrambled
```

**Option C: Test LLM on one transformation**
```bash
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods llm \
  --transformations original
```

### **Step 3: Analyze Results**

Generate summary statistics and visualizations:

```bash
python scripts/analyze_results.py
```

**Outputs:**
- `results/summary.csv` - Complete results table
- Plots comparing methods across transformations

---
