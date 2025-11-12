# Entity Matching Evaluation

Comparing traditional string matching, embeddings, and LLMs for entity matching.

## Research Question
How do different entity matching techniques perform on product data when tested against:
1. Original data
2. Ciphered data (letter substitution)
3. Letter-scrambled data (letters shuffled within words)


## Data
- **Dataset**: Abt-Buy product matching
- **Source**: Downloaded from entity matching benchmark
- **Ground Truth**: 1,097 known matching pairs

## Methods Tested
1. **Traditional String-Based**: Jaro-Winkler, Levenshtein
2. **Token-Based**: TF-IDF, Soft TF-IDF
3. **Hybrid**: Monge-Elkan
4. **Neural Embeddings**: Sentence Transformers (free), OpenAI (paid)
5. **LLM**: GPT-4o-mini

## Usage

### Step 1: Prepare Data
```bash
python src/data_prep/prepare_datasets.py
```

This creates:
- `data/processed/abt_ciphered.csv`, `buy_ciphered.csv`
- `data/processed/abt_scrambled.csv`, `buy_scrambled.csv`


### Step 2: Run Evaluation
```bash
python src/evaluation/run_evaluation.py
```

### Step 3: Generate Results
```bash
python src/evaluation/calculate_accuracy.py
```

## Results
Results saved to `results/`:
- `accuracy_by_dataset.csv` - Main results table
