# Entity Matching: A Comprehensive Evaluation of Traditional, Neural, and LLM Based Methods

**Systematic comparison of 8 entity matching methods across different text transformations**

---

##  Research Question

**How do different entity matching techniques perform under systematically degraded text conditions**

We evaluate methods on:
1. **Original data** - Baseline performance
2. **Ciphered letters** - Consistent character substitution (e.g., 'a'→'x', 'b'→'y')
3. **Ciphered words** - Vocabulary replacement (e.g., 'Sony'→'Bear')
4. **Scrambled letters** - Letters shuffled within words (e.g., 'Sony'→'nSyo')

---

##  Key Findings

### **Best Method: LLM (GPT-4o-mini) + TF-IDF Blocking** compared to the next best method

| Transformation       | LLM F1    | OpenAI Embeddings (F1)| Improvement |
|----------------------|-----------|-----------------------|-------------|
| **Original**         | **0.928** | 0.799                 | **+16%**    |
| **Ciphered Letters** | **0.779** | 0.409                 | **+90%**    |
| **Scrambled**        | **0.603** | 0.508                 | **+19%**    |  
| **Ciphered Words**   | **0.511** | 0.422                 | **+21%**    |

**Key Insight:** LLMs maintain semantic understanding even under adversarial transformations, outperforming pure embedding methods by 90% on ciphered data.

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
