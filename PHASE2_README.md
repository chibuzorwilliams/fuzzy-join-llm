# Phase 2: Automated Experiment Runner - COMPLETE

## 🎉 What You Got

### 1. `scripts/run_experiments.py` ✅
**Complete** automated runner that executes all methods across all transformations.

### 2. `scripts/methods.py` ✅  
**Complete** with ALL 8 method implementations extracted from your notebook v6:
- ✅ Jaro-Winkler
- ✅ Levenshtein
- ✅ Monge-Elkan
- ✅ TF-IDF
- ✅ Soft TF-IDF
- ✅ SentenceTransformer
- ✅ OpenAI Embeddings
- ✅ LLM (with improved blocking, confidence thresholds)

## 📦 Installation

```bash
cd ~/fuzzy-join-llm

# Move files to scripts/
mv ~/Downloads/run_experiments.py scripts/
mv ~/Downloads/methods.py scripts/

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Test (Single Method, Single Transformation)
```bash
# Test Jaro-Winkler on original data
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods jaro_winkler \
  --transformations original
```

### Run All Methods on One Dataset
```bash
# Run all 8 methods on all 4 transformations for abt-buy
python scripts/run_experiments.py --dataset abt-buy

# This generates 32 parquet files (8 methods × 4 transformations)
```

### Run Specific Methods
```bash
# Fast methods only (no LLM, no OpenAI)
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods jaro_winkler,levenshtein,monge_elkan,tfidf,soft_tfidf,sentence_transformer

# Just the expensive methods
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods openai_embeddings,llm
```

### Run Specific Transformations
```bash
# Just original and ciphered_letters
python scripts/run_experiments.py \
  --dataset abt-buy \
  --transformations original,ciphered_letters
```

### Run All Datasets (when ready)
```bash
python scripts/run_experiments.py --dataset all
```

## 📊 Output Structure

After running, you'll have:

```
results/
└── abt-buy/
    ├── jaro_winkler/
    │   ├── original.parquet           ✅
    │   ├── ciphered_letters.parquet   ✅
    │   ├── ciphered_words.parquet     ✅
    │   └── scrambled.parquet          ✅
    ├── levenshtein/
    │   ├── original.parquet
    │   └── ...
    ├── monge_elkan/
    ├── tfidf/
    ├── soft_tfidf/
    ├── sentence_transformer/
    ├── openai_embeddings/
    └── llm/
```

Each parquet file contains:
- `id_left`: Left record ID
- `id_right`: Right record ID (matched)
- `similarity`: Similarity score
- `is_match`: Binary prediction (0 or 1)
- `ground_truth`: True label (0 or 1)
- `method`: Method name
- `transformation`: Transformation type
- `dataset`: Dataset name
- `timestamp`: When experiment ran

## ⏱️ Estimated Runtime

| Method | Time per Transformation |
|--------|------------------------|
| Jaro-Winkler | 2-5 mins |
| Levenshtein | 2-5 mins |
| Monge-Elkan | 5-10 mins |
| TF-IDF | 30 secs - 2 mins |
| Soft TF-IDF | 5-10 mins |
| SentenceTransformer | 1-3 mins |
| OpenAI Embeddings | 10-20 mins |
| LLM | 30-60 mins |

**Total for abt-buy (8 methods × 4 transformations): ~3-5 hours**

## 💰 Cost Estimate

- **String methods** (Jaro, Levenshtein, Monge-Elkan, TF-IDF, Soft-TFIDF): FREE
- **SentenceTransformer**: FREE (local model)
- **OpenAI Embeddings**: ~$0.02-0.05 per transformation
- **LLM**: ~$0.50-2.00 per transformation

**Total for abt-buy (all 8 methods, 4 transformations): ~$2-10**

## 🔧 Configuration

Edit `scripts/run_experiments.py` to adjust:
- **Thresholds**: Modify threshold values in each method
- **LLM parameters**: Change `top_k`, `blocking_threshold`, `confidence_threshold`
- **Batch sizes**: Adjust for OpenAI API calls

Edit `scripts/methods.py` to:
- Add new methods
- Modify existing methods
- Change similarity functions

## 📋 Next Steps (Phase 1)

After session resets (~2 hours), I'll create:
1. ✅ Clean notebook v7 with updated paths
2. ✅ Helper functions for loading results
3. ✅ Analysis and visualization code

## 🎯 Quick Validation

Test that everything works:

```bash
# 1. Test imports
python -c "from scripts.methods import METHODS; print(list(METHODS.keys()))"

# 2. Run quick test (should take ~2 mins)
python scripts/run_experiments.py \
  --dataset abt-buy \
  --methods jaro_winkler \
  --transformations original

# 3. Check output
ls -lh results/abt-buy/jaro_winkler/
```

## 💡 Pro Tips

1. **Run overnight**: LLM method takes longest, run it overnight
2. **Save costs**: Test with fast methods first, then run expensive methods
3. **Incremental**: Run one transformation at a time if needed
4. **Monitor**: Check terminal output for progress and costs
5. **Results**: Load parquet files with `pd.read_parquet()` for analysis

## ⚠️ Important Notes

- **OpenAI API Key**: Make sure `.env` file has `OPENAI_API_KEY=your_key_here`
- **Memory**: SentenceTransformer and embedding methods need ~2GB RAM
- **Disk Space**: All parquet files total ~50-100MB
- **Progress**: Progress bars show real-time status
- **Errors**: Script continues even if one method fails

## 🐛 Troubleshooting

**Import Error:**
```bash
# Make sure you're in project root
cd ~/fuzzy-join-llm
python scripts/run_experiments.py --dataset abt-buy
```

**OpenAI Error:**
```bash
# Check API key
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Key found!' if os.getenv('OPENAI_API_KEY') else 'No key!')"
```

**Memory Error:**
```bash
# Run methods one at a time
python scripts/run_experiments.py --dataset abt-buy --methods jaro_winkler
```

---

## ✅ Phase 2 Status: COMPLETE

You now have:
- ✅ Full automated experiment runner
- ✅ All 8 methods implemented
- ✅ Ready to generate 32 parquet files
- ✅ Production-ready code

**Ready to run experiments!** 🚀
