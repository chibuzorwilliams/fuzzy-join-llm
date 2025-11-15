"""
FREE BLOCKING DIAGNOSTIC - No API Costs!
=========================================

This script checks how well TF-IDF blocking works BEFORE running expensive LLM.

Key Question: "How often is the TRUE match in the TF-IDF top-K candidates?"

If blocking_recall >= 70%: LLM will likely work well (F1 ~0.70-0.80)
If blocking_recall < 50%: LLM will fail (F1 ~0.30-0.40) - don't run!
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

def prepare_text(df):
    """Prepare text column by concatenating and normalizing"""
    text_cols = []
    for col in df.columns[1:]:  # Skip ID column
        if df[col].dtype == 'object':
            text_cols.append(col)
    
    df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    df['text'] = df['text'].str.lower().str.strip()
    return df

def diagnose_blocking(dataset_name='abt-buy', transformation='original', top_k=50):
    """
    Diagnose TF-IDF blocking quality
    
    Returns:
        blocking_recall: % of true matches found in top-K
        details: DataFrame with per-record analysis
    """
    
    print("="*80)
    print(f"BLOCKING DIAGNOSTIC: {dataset_name} ({transformation})")
    print("="*80)
    
    # Try multiple possible data paths
    possible_paths = [
        Path(f"data/{dataset_name}/data_{transformation}"),  # data_original, data_ciphered_letters, etc.
        Path(f"data/{dataset_name}/{transformation}"),
        Path(f"data/{dataset_name}"),
        Path("."),
        Path("data"),
    ]
    
    # Find the correct path
    data_path = None
    for path in possible_paths:
        if (path / "abt.csv").exists() or (path / "Abt.csv").exists():
            data_path = path
            break
    
    if data_path is None:
        print("ERROR: Could not find data files!")
        print("\nSearching in:")
        for path in possible_paths:
            print(f"  {path.absolute()}")
        print("\nPlease run this from your fuzzy-join-llm directory")
        print("Or tell me where your CSV files are located")
        sys.exit(1)
    
    print(f"\nFound data in: {data_path.absolute()}")
    
    # Try different file names (case sensitivity)
    try:
        df_left = pd.read_csv(data_path / "abt.csv", encoding='latin1')
    except FileNotFoundError:
        try:
            df_left = pd.read_csv(data_path / "Abt.csv", encoding='latin1')
        except FileNotFoundError:
            df_left = pd.read_csv(data_path / "abt.csv")  # Try without encoding
    
    try:
        df_right = pd.read_csv(data_path / "buy.csv", encoding='latin1')
    except FileNotFoundError:
        try:
            df_right = pd.read_csv(data_path / "Buy.csv", encoding='latin1')
        except FileNotFoundError:
            df_right = pd.read_csv(data_path / "buy.csv")
    
    try:
        df_mapping = pd.read_csv(data_path / "abt_buy_perfect_mapping.csv")
    except FileNotFoundError:
        try:
            df_mapping = pd.read_csv(data_path / "Abt_Buy_perfectMapping.csv")
        except FileNotFoundError:
            df_mapping = pd.read_csv(data_path / "abt_buy_perfect_mapping.csv", encoding='latin1')
    
    print(f"\nLoaded:")
    print(f"  Left: {len(df_left)} records")
    print(f"  Right: {len(df_right)} records")
    print(f"  Ground truth: {len(df_mapping)} matches")
    
    # Prepare text
    df_left = prepare_text(df_left)
    df_right = prepare_text(df_right)
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    
    # Create ground truth set
    ground_truth = set(zip(
        df_mapping.iloc[:, 0].astype(str),
        df_mapping.iloc[:, 1].astype(str)
    ))
    
    # Count how many left IDs have ground truth
    left_with_gt = set(df_mapping.iloc[:, 0].astype(str))
    print(f"  Left records with ground truth: {len(left_with_gt)}")
    
    # Compute TF-IDF similarity matrix
    print(f"\nComputing TF-IDF similarity matrix...")
    vectorizer = TfidfVectorizer()
    all_texts = df_left['text'].tolist() + df_right['text'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    left_matrix = tfidf_matrix[:len(df_left)]
    right_matrix = tfidf_matrix[len(df_left):]
    
    print(f"Computing cosine similarities...")
    sim_matrix = cosine_similarity(left_matrix, right_matrix)
    
    # For each left record, get top-K candidates
    print(f"\nAnalyzing top-{top_k} candidates for each record...")
    
    results = []
    found_in_topk = 0
    total_with_gt = 0
    
    for idx, row_left in df_left.iterrows():
        id_left = str(row_left[id_col_left])
        
        # Check if this record has ground truth
        true_matches = [right for (left, right) in ground_truth if left == id_left]
        
        if not true_matches:
            continue  # Skip records without ground truth
        
        total_with_gt += 1
        
        # Get top-K candidates by TF-IDF similarity
        similarities = sim_matrix[idx]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Get candidate IDs
        candidate_ids = [str(df_right.iloc[i][id_col_right]) for i in top_indices]
        
        # Check if ANY true match is in top-K
        found = any(true_id in candidate_ids for true_id in true_matches)
        
        if found:
            found_in_topk += 1
        
        # Find rank of best true match (if any)
        best_rank = None
        best_sim = 0
        for true_id in true_matches:
            if true_id in candidate_ids:
                rank = candidate_ids.index(true_id) + 1
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_sim = similarities[df_right[df_right[id_col_right].astype(str) == true_id].index[0]]
        
        results.append({
            'id_left': id_left,
            'text_left': row_left['text'][:100] + "...",
            'num_true_matches': len(true_matches),
            'found_in_topk': found,
            'best_rank': best_rank,
            'best_similarity': best_sim if best_rank else None
        })
    
    # Calculate blocking recall
    blocking_recall = found_in_topk / total_with_gt if total_with_gt > 0 else 0
    
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print("="*80)
    print(f"\nRecords with ground truth: {total_with_gt}")
    print(f"True matches found in top-{top_k}: {found_in_topk}")
    print(f"\n🎯 BLOCKING RECALL: {blocking_recall:.1%}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if blocking_recall >= 0.80:
        print(f"\n✅ EXCELLENT! {blocking_recall:.1%} of true matches are in top-{top_k}")
        print(f"   → LLM should work well!")
        print(f"   → Expected LLM F1: 0.75-0.85")
        print(f"   → RECOMMENDATION: Run full LLM experiment confidently!")
    elif blocking_recall >= 0.65:
        print(f"\n⚠️  GOOD but not great. {blocking_recall:.1%} of true matches are in top-{top_k}")
        print(f"   → LLM will miss {100-blocking_recall*100:.0f}% of matches before it even tries")
        print(f"   → Expected LLM F1: 0.60-0.75")
        print(f"   → RECOMMENDATION: Run LLM, but consider embedding blocking for better results")
    elif blocking_recall >= 0.50:
        print(f"\n⚠️  MEDIOCRE. Only {blocking_recall:.1%} of true matches are in top-{top_k}")
        print(f"   → LLM will miss half the matches!")
        print(f"   → Expected LLM F1: 0.45-0.60")
        print(f"   → RECOMMENDATION: Use embedding blocking instead of TF-IDF")
    else:
        print(f"\n❌ POOR! Only {blocking_recall:.1%} of true matches are in top-{top_k}")
        print(f"   → LLM won't work - it never sees most true matches!")
        print(f"   → Expected LLM F1: 0.30-0.45")
        print(f"   → RECOMMENDATION: DO NOT run LLM with TF-IDF blocking!")
        print(f"   → Use embedding blocking or increase top_k to 100+")
    
    # Show examples
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("EXAMPLES")
    print("="*80)
    
    print("\n✅ FOUND IN TOP-K (first 5):")
    found = df_results[df_results['found_in_topk'] == True].head(5)
    for _, row in found.iterrows():
        print(f"\n  ID: {row['id_left']}")
        print(f"  Text: {row['text_left']}")
        print(f"  True match rank: {row['best_rank']}/{top_k}")
        print(f"  Similarity: {row['best_similarity']:.3f}")
    
    print("\n\n❌ NOT FOUND IN TOP-K (first 5):")
    not_found = df_results[df_results['found_in_topk'] == False].head(5)
    for _, row in not_found.iterrows():
        print(f"\n  ID: {row['id_left']}")
        print(f"  Text: {row['text_left']}")
        print(f"  True match NOT in top-{top_k}! (missed by blocking)")
    
    return blocking_recall, df_results


if __name__ == "__main__":
    # You can change these parameters
    dataset = "abt-buy"
    transformation = "original"
    top_k = 50  # Same as LLM will use
    
    print("\n🔍 Starting blocking diagnostic...\n")
    print("This checks: 'If LLM only sees top-50 TF-IDF candidates,")
    print("             how many true matches will it have a chance to find?'\n")
    
    blocking_recall, details = diagnose_blocking(dataset, transformation, top_k)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBlocking Recall: {blocking_recall:.1%}")
    print(f"Top-K: {top_k}")
    print(f"Transformation: {transformation}")
    
    print("\n💰 COST ESTIMATE FOR FULL LLM:")
    print(f"   Cost: ~$0.30")
    print(f"   Time: ~40 minutes")
    print(f"   Expected F1: {blocking_recall * 0.95:.2f} (assuming LLM perfect on candidates)")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if blocking_recall >= 0.70:
        print("\n✅ GO FOR IT!")
        print("   Run: python scripts/run_experiments.py --dataset abt-buy --methods llm --transformations original")
    else:
        print("\n⚠️  CONSIDER ALTERNATIVES:")
        print("   1. Use embedding blocking instead of TF-IDF")
        print("   2. Increase top_k to 100 (costs 2x more)")
        print("   3. Stick with embeddings alone (F1=0.799)")
    
    print("\n")
