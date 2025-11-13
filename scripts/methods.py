"""
ENTITY MATCHING METHODS - CORRECT IMPLEMENTATION
================================================
Extracted from complete_entity_matching_v6.ipynb with:
- Threshold optimization (0.50 to 0.95, step 0.05)
- Proper output schema with product names
- Exact matching logic from notebook

Output Schema:
--------------
1. id_left - ID from left dataset
2. left_name - Product name from left dataset  
3. true_id_right - Ground truth match ID from right dataset
4. pred_id_right - Predicted match ID from right dataset
5. pred_right_name - Predicted product name from right dataset
6. similarity_score - Similarity score from algorithm
7. is_correct - 1 = correct match, 0 = wrong/no match
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import Tuple, Set

# String matching
import jellyfish

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings
from sentence_transformers import SentenceTransformer

# OpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

# Lazy loading
_sentence_model = None
_openai_client = None

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        print("Loading SentenceTransformer model...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_text_column(df, text_cols=['name', 'description']):
    """Combine text columns"""
    df = df.copy()
    text_parts = []
    for col in text_cols:
        if col in df.columns:
            text_parts.append(df[col].fillna('').astype(str))
    
    if text_parts:
        df['text'] = text_parts[0]
        for part in text_parts[1:]:
            df['text'] = df['text'] + ' ' + part
    else:
        df['text'] = ''
    
    return df

def create_ground_truth_dict(df_mapping):
    """Create dict mapping left_id -> right_id from ground truth"""
    gt_dict = {}
    left_col = df_mapping.columns[0]
    right_col = df_mapping.columns[1]
    
    for _, row in df_mapping.iterrows():
        left_id = str(row[left_col])
        right_id = str(row[right_col])
        gt_dict[left_id] = right_id
    
    return gt_dict

# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def jaro_winkler_similarity(str1, str2):
    return jellyfish.jaro_winkler_similarity(str1, str2)

def levenshtein_similarity(str1, str2):
    distance = jellyfish.levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)

def monge_elkan_similarity(str1, str2):
    tokens_a = str1.split()
    tokens_b = str2.split()
    
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 0.0
    
    sum_best = 0.0
    for token_a in tokens_a:
        best = max([jaro_winkler_similarity(token_a, token_b) for token_b in tokens_b])
        sum_best += best
    
    return sum_best / len(tokens_a)

def soft_tfidf_similarity(str1, str2, theta=0.9):
    tokens_a = str1.split()
    tokens_b = str2.split()
    
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 0.0
    
    score = 0.0
    for token_a in tokens_a:
        best_match = max([jaro_winkler_similarity(token_a, token_b) for token_b in tokens_b])
        if best_match >= theta:
            score += best_match
    
    return score / len(tokens_a) if len(tokens_a) > 0 else 0.0

# =============================================================================
# THRESHOLD OPTIMIZATION (FROM NOTEBOOK)
# =============================================================================

def find_best_threshold_with_details(df_left, df_right, similarity_func, gt_dict,
                                     method_name="Method"):
    """
    Find optimal threshold by trying 0.50 to 0.95 in steps of 0.05
    AND return detailed match results for best threshold
    """
    print(f"\n🔍 Optimizing threshold for {method_name}...")
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = 'name' if 'name' in df_left.columns else df_left.columns[1]
    name_col_right = 'name' if 'name' in df_right.columns else df_right.columns[1]
    
    # Compute all similarities ONCE (expensive!)
    print("Computing all similarities...")
    all_matches = []
    
    for idx_a, row_a in tqdm(df_left.iterrows(), total=len(df_left), desc="Matching"):
        id_a = str(row_a[id_col_left])
        text_a = row_a['text']
        name_a = row_a[name_col_left]
        true_id_b = gt_dict.get(id_a, None)
        
        # Find best match
        best_sim = 0.0
        best_id_b = None
        best_name_b = None
        
        for idx_b, row_b in df_right.iterrows():
            id_b = str(row_b[id_col_right])
            text_b = row_b['text']
            name_b = row_b[name_col_right]
            
            sim = similarity_func(text_a, text_b)
            
            if sim > best_sim:
                best_sim = sim
                best_id_b = id_b
                best_name_b = name_b
        
        all_matches.append({
            'id_left': id_a,
            'left_name': name_a,
            'true_id_right': true_id_b if true_id_b else '',
            'pred_id_right': best_id_b,
            'pred_right_name': best_name_b,
            'similarity_score': best_sim
        })
    
    # Now optimize threshold
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0
    
    print("\nTesting thresholds:")
    for threshold in thresholds:
        tp = fp = fn = 0
        
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = match['true_id_right'] != '' and match['pred_id_right'] == match['true_id_right']
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            elif not pred_match and match['true_id_right'] != '':
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add is_correct column using best threshold
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = match['true_id_right'] != '' and match['pred_id_right'] == match['true_id_right']
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches), best_threshold, best_f1

# =============================================================================
# STRING DISTANCE METHODS WITH THRESHOLD OPTIMIZATION
# =============================================================================

def string_distance_method(df_left, df_right, df_mapping, similarity_func, method_name):
    """Generic string distance with threshold optimization"""
    df_left = prepare_text_column(df_left.copy())
    df_right = prepare_text_column(df_right.copy())
    
    gt_dict = create_ground_truth_dict(df_mapping)
    
    results_df, best_threshold, best_f1 = find_best_threshold_with_details(
        df_left, df_right, similarity_func, gt_dict, method_name
    )
    
    return results_df

def jaro_winkler(df_left, df_right, df_mapping):
    return string_distance_method(df_left, df_right, df_mapping, 
                                  jaro_winkler_similarity, "Jaro-Winkler")

def levenshtein(df_left, df_right, df_mapping):
    return string_distance_method(df_left, df_right, df_mapping,
                                  levenshtein_similarity, "Levenshtein")

def monge_elkan(df_left, df_right, df_mapping):
    return string_distance_method(df_left, df_right, df_mapping,
                                  monge_elkan_similarity, "Monge-Elkan")

def soft_tfidf(df_left, df_right, df_mapping):
    return string_distance_method(df_left, df_right, df_mapping,
                                  soft_tfidf_similarity, "Soft-TF-IDF")

# =============================================================================
# TF-IDF WITH THRESHOLD OPTIMIZATION
# =============================================================================

def tfidf(df_left, df_right, df_mapping):
    """TF-IDF with threshold optimization"""
    print("\n🔍 Running TF-IDF with threshold optimization...")
    
    df_left = prepare_text_column(df_left.copy())
    df_right = prepare_text_column(df_right.copy())
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = 'name' if 'name' in df_left.columns else df_left.columns[1]
    name_col_right = 'name' if 'name' in df_right.columns else df_right.columns[1]
    
    gt_dict = create_ground_truth_dict(df_mapping)
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    all_texts = pd.concat([df_left['text'], df_right['text']])
    vectorizer.fit(all_texts)
    
    vectors_left = vectorizer.transform(df_left['text'])
    vectors_right = vectorizer.transform(df_right['text'])
    
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(vectors_left, vectors_right)
    
    # Build all matches
    all_matches = []
    for idx_a in tqdm(range(len(df_left)), desc="TF-IDF matching"):
        id_a = str(df_left.iloc[idx_a][id_col_left])
        name_a = df_left.iloc[idx_a][name_col_left]
        true_id_b = gt_dict.get(id_a, None)
        
        best_idx_b = np.argmax(similarity_matrix[idx_a])
        best_sim = similarity_matrix[idx_a, best_idx_b]
        best_id_b = str(df_right.iloc[best_idx_b][id_col_right])
        best_name_b = df_right.iloc[best_idx_b][name_col_right]
        
        all_matches.append({
            'id_left': id_a,
            'left_name': name_a,
            'true_id_right': true_id_b if true_id_b else '',
            'pred_id_right': best_id_b,
            'pred_right_name': best_name_b,
            'similarity_score': best_sim
        })
    
    # Optimize threshold
    print("\nOptimizing threshold...")
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = match['true_id_right'] != '' and match['pred_id_right'] == match['true_id_right']
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            elif not pred_match and match['true_id_right'] != '':
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = match['true_id_right'] != '' and match['pred_id_right'] == match['true_id_right']
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches)

# =============================================================================
# SENTENCE TRANSFORMER WITH THRESHOLD OPTIMIZATION
# =============================================================================

def sentence_transformer(df_left, df_right, df_mapping):
    """SentenceTransformer with threshold optimization"""
    print("\n🔍 Running SentenceTransformer with threshold optimization...")
    
    df_left = prepare_text_column(df_left.copy())
    df_right = prepare_text_column(df_right.copy())
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = 'name' if 'name' in df_left.columns else df_left.columns[1]
    name_col_right = 'name' if 'name' in df_right.columns else df_right.columns[1]
    
    gt_dict = create_ground_truth_dict(df_mapping)
    
    # Encode
    model = get_sentence_model()
    print("Encoding datasets...")
    embeddings_left = model.encode(df_left['text'].tolist(), show_progress_bar=True)
    embeddings_right = model.encode(df_right['text'].tolist(), show_progress_bar=True)
    
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings_left, embeddings_right)
    
    # Build matches
    all_matches = []
    for idx_a in tqdm(range(len(df_left)), desc="Matching"):
        id_a = str(df_left.iloc[idx_a][id_col_left])
        name_a = df_left.iloc[idx_a][name_col_left]
        true_id_b = gt_dict.get(id_a, None)
        
        best_idx_b = np.argmax(similarity_matrix[idx_a])
        best_sim = similarity_matrix[idx_a, best_idx_b]
        best_id_b = str(df_right.iloc[best_idx_b][id_col_right])
        best_name_b = df_right.iloc[best_idx_b][name_col_right]
        
        all_matches.append({
            'id_left': id_a,
            'left_name': name_a,
            'true_id_right': true_id_b if true_id_b else '',
            'pred_id_right': best_id_b,
            'pred_right_name': best_name_b,
            'similarity_score': best_sim
        })
    
    # Optimize threshold (same code as TF-IDF)
    print("\nOptimizing threshold...")
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = match['true_id_right'] != '' and match['pred_id_right'] == match['true_id_right']
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            elif not pred_match and match['true_id_right'] != '':
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = match['true_id_right'] != '' and match['pred_id_right'] == match['true_id_right']
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches)

# =============================================================================
# OPENAI EMBEDDINGS - Simplified (will add full version if needed)
# =============================================================================

def openai_embeddings(df_left, df_right, df_mapping):
    """OpenAI embeddings with threshold optimization"""
    print("\n🔍 Running OpenAI Embeddings...")
    print("⚠️  Note: This method requires OpenAI API calls and will incur costs")
    
    # Similar structure to sentence_transformer but using OpenAI
    raise NotImplementedError("OpenAI embeddings - add if needed")

# =============================================================================
# LLM - Simplified  (will add full version if needed)
# =============================================================================

def llm(df_left, df_right, df_mapping):
    """LLM matching with confidence optimization"""
    print("\n🔍 Running LLM matching...")
    print("⚠️  Note: This is expensive - requires API calls")
    
    raise NotImplementedError("LLM matching - add if needed")

# =============================================================================
# METHOD REGISTRY
# =============================================================================

METHODS = {
    'jaro_winkler': jaro_winkler,
    'levenshtein': levenshtein,
    'monge_elkan': monge_elkan,
    'tfidf': tfidf,
    'soft_tfidf': soft_tfidf,
    'sentence_transformer': sentence_transformer,
    # 'openai_embeddings': openai_embeddings,  # Uncomment when needed
    # 'llm': llm  # Uncomment when needed
}

if __name__ == "__main__":
    print("✅ Methods loaded with threshold optimization:")
    for i, name in enumerate(METHODS.keys(), 1):
        print(f"  {i}. {name}")
