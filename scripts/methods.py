"""
ENTITY MATCHING METHODS - FINAL DEFINITIVE VERSION
===================================================
One-to-many ground truth support (set-based)
Correct FN counting (uses left_has_truth)
Consistent threshold logic across all methods
Text preprocessing (.lower().strip())
predicted_match column for clear evaluation

Output Schema:
--------------
1. id_left - ID from left dataset
2. left_name - Product name from left dataset  
3. true_id_right - Ground truth match ID (first if multiple)
4. pred_id_right - Predicted match ID
5. pred_right_name - Predicted product name
6. similarity_score - Similarity score
7. predicted_match - 1 if above threshold, 0 otherwise
8. is_correct - 1 if predicted correctly, 0 otherwise
+ method, transformation, dataset, timestamp
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

# Anthropic
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Lazy loading
_sentence_model = None
_openai_client = None
_anthropic_client = None

# TITLE-ONLY MODE (set via environment variable)
TITLE_ONLY_MODE = os.environ.get('TITLE_ONLY', 'false').lower() == 'true'

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

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        _anthropic_client = Anthropic(api_key=api_key, timeout=60.0)
    return _anthropic_client

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_text_column(df, text_cols=None, title_only=False):
    """Combine text columns and normalize - auto-detect columns if not specified
    
    - Products (Abt-Buy, Amazon-Google): uses name/title + description + manufacturer
    - Publications (DBLP): uses title + authors + venue
    
    Parameters:
    - title_only: If True, only use title/name column (for fair comparison)
    """
    df = df.copy()
    
    # Auto-detect columns if not specified
    if text_cols is None:
        if title_only:
            # Use ONLY title/name column for fair comparison
            text_cols = []
            for col in df.columns[1:]:
                if col.lower() in ['title', 'name']:
                    text_cols = [col]
                    break
            print(f"  Using title/name only: {text_cols}")
        else:
            # Auto-detect all text columns (current behavior)
            text_cols = []
            # Skip first column (ID) and non-text columns
            for col in df.columns[1:]:
                if col in df.columns:
                    # Include string/object columns, exclude numeric/irrelevant ones
                    # (newer pandas reports text columns as 'str' rather than 'object')
                    if pd.api.types.is_string_dtype(df[col]) and col.lower() not in ['year', 'price', 'id']:
                        text_cols.append(col)
            
            print(f"  Auto-detected text columns: {text_cols}")
    
    text_parts = []
    for col in text_cols:
        if col in df.columns:
            text_parts.append(df[col].fillna('').astype(str))
    
    if text_parts:
        df['text'] = text_parts[0]
        for part in text_parts[1:]:
            df['text'] = df['text'] + ' ' + part
    else:
        # Fallback if no text columns found
        df['text'] = ''
        print(f"WARNING: No text columns found! Columns: {df.columns.tolist()}")
    
    # NORMALIZE: lowercase and strip (matches notebook)
    df['text'] = df['text'].str.lower().str.strip()
    
    # Validation
    avg_len = df['text'].str.len().mean()
    print(f"Text prepared: avg length = {avg_len:.0f} chars")
    
    return df

def get_display_column(df):
    """Get the best column name for display in results
   
    """
    # Prefer these columns in order
    preferred = ['name', 'title', 'authors', 'description']
    for col in preferred:
        if col in df.columns:
            return col
    
    # Fallback to second column (first is usually ID)
    if len(df.columns) > 1:
        return df.columns[1]
    return df.columns[0]

def create_ground_truth_dict(df_mapping):
    """Create dict mapping left_id -> right_id from ground truth
    WARNING: Overwrites if one left has multiple rights!
    Use create_ground_truth_set for one-to-many support."""
    gt_dict = {}
    left_col = df_mapping.columns[0]
    right_col = df_mapping.columns[1]
    
    for _, row in df_mapping.iterrows():
        left_id = str(row[left_col])
        right_id = str(row[right_col])
        gt_dict[left_id] = right_id
    
    return gt_dict

def create_ground_truth_set(df_mapping):
    """Convert ground truth to set of (left_id, right_id) pairs.
    Preserves ALL valid one-to-many relationships."""
    left_col = df_mapping.columns[0]
    right_col = df_mapping.columns[1]
    return set(zip(
        df_mapping[left_col].astype(str),
        df_mapping[right_col].astype(str)
    ))

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
    """DEPRECATED: fuzzy token overlap WITHOUT TF/IDF weighting.

    This is NOT Soft TF-IDF (Cohen et al. 2003): it has no term weighting,
    so it is really a thresholded Monge-Elkan. Kept only for reference.
    The experiments use soft_tfidf() below, which implements the real
    IDF-weighted Soft TF-IDF. Do not use this for the paper.
    """
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
# THRESHOLD OPTIMIZATION (STRING DISTANCE METHODS)
# =============================================================================

def find_best_threshold_with_details(df_left, df_right, similarity_func, true_matches_set,
                                     method_name="Method"):
    """
    Find optimal threshold with CORRECT FN counting.
    
    Args:
        true_matches_set: SET of (left_id, right_id) tuples for one-to-many support
    """
    print(f"\n🔍 Optimizing threshold for {method_name}...")
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)
    
    # BUILD left_has_truth MAP
    has_truth = {}
    for (la, rb) in true_matches_set:
        has_truth[la] = True
    
    # Compute all similarities ONCE
    print("Computing all similarities...")
    all_matches = []
    
    for idx_a, row_a in tqdm(df_left.iterrows(), total=len(df_left), desc="Matching"):
        id_a = str(row_a[id_col_left])
        text_a = row_a['text']
        name_a = row_a[name_col_left]
        
        # Find ALL true matches for this left (one-to-many)
        true_matches_for_a = [right for (left, right) in true_matches_set if left == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        
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
    
    # Optimize threshold
    thresholds = np.arange(0.05, 0.96, 0.05)  # full grid; cosine/edit matches score well below 0.5
    best_threshold = 0.5
    best_f1 = 0.0
    
    print("\nTesting thresholds:")
    for threshold in thresholds:
        tp = fp = fn = 0
        
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            id_a = match['id_left']
            id_b = match['pred_id_right']
            true_match = (id_a, id_b) in true_matches_set
            left_has_truth = has_truth.get(id_a, False)
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            
            # FIX: FN is ANY record with ground truth that's not TP
            if left_has_truth and not (pred_match and true_match):
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add predicted_match AND is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = (match['id_left'], match['pred_id_right']) in true_matches_set
        match['predicted_match'] = 1 if pred_match else 0  
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches), best_threshold, best_f1

# =============================================================================
# STRING DISTANCE METHODS
# =============================================================================

def string_distance_method(df_left, df_right, df_mapping, similarity_func, method_name):
    """Generic string distance with threshold optimization"""
    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)
    
    true_matches_set = create_ground_truth_set(df_mapping)
    
    results_df, best_threshold, best_f1 = find_best_threshold_with_details(
        df_left, df_right, similarity_func, true_matches_set, method_name
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

def _fuzzy_token_matrix(vocab, theta=0.9):
    """Sparse VxV Jaro-Winkler similarity matrix for token pairs with JW >= theta
    (diagonal excluded). Prefix-bucketed for speed (JW>=0.9 implies a shared prefix)."""
    from collections import defaultdict
    from scipy.sparse import csr_matrix
    buckets = defaultdict(list)
    for i, w in enumerate(vocab):
        buckets[w[0] if w else ""].append(i)
    rows, cols, vals = [], [], []
    for idxs in buckets.values():
        for a in range(len(idxs)):
            i = idxs[a]; wi = vocab[i]
            for b in range(a + 1, len(idxs)):
                j = idxs[b]; wj = vocab[j]
                if abs(len(wi) - len(wj)) > 3:
                    continue
                s = jaro_winkler_similarity(wi, wj)
                if s >= theta:
                    rows += [i, j]; cols += [j, i]; vals += [s, s]
    return csr_matrix((vals, (rows, cols)), shape=(len(vocab), len(vocab)))

def soft_tfidf(df_left, df_right, df_mapping):
    """Real Soft TF-IDF (Cohen et al. 2003): L2-normalized TF-IDF token weights with
    Jaro-Winkler fuzzy token matching (theta=0.9). Computed as the bilinear form
    score = L (I + M) R^T, where M holds fuzzy inter-token similarities. With theta=0.9
    a token has at most ~1 fuzzy partner, so this matches Cohen's max-based definition."""
    print("\n🔍 Running Soft TF-IDF (IDF-weighted, Jaro-Winkler fuzzy match)...")
    import numpy as np
    from scipy.sparse import identity

    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)

    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)

    true_matches = create_ground_truth_set(df_mapping)
    has_truth = {}
    for (la, rb) in true_matches:
        has_truth[la] = True

    vectorizer = TfidfVectorizer(norm="l2")
    vectorizer.fit(pd.concat([df_left['text'], df_right['text']]))
    vocab = vectorizer.get_feature_names_out()
    left_m = vectorizer.transform(df_left['text'])
    right_m = vectorizer.transform(df_right['text'])

    print(f"Building fuzzy token matrix over {len(vocab):,} terms...")
    M = _fuzzy_token_matrix(vocab)
    I = identity(len(vocab), format="csr")
    print("Computing Soft TF-IDF score matrix...")
    similarity_matrix = np.asarray((left_m @ (I + M) @ right_m.T).todense())

    all_matches = []
    for idx_a in tqdm(range(len(df_left)), desc="Soft TF-IDF matching"):
        id_a = str(df_left.iloc[idx_a][id_col_left])
        name_a = df_left.iloc[idx_a][name_col_left]
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        best_idx_b = int(np.argmax(similarity_matrix[idx_a]))
        best_sim = float(similarity_matrix[idx_a, best_idx_b])
        all_matches.append({
            'id_left': id_a,
            'left_name': name_a,
            'true_id_right': true_id_b if true_id_b else '',
            'pred_id_right': str(df_right.iloc[best_idx_b][id_col_right]),
            'pred_right_name': df_right.iloc[best_idx_b][name_col_right],
            'similarity_score': best_sim
        })

    print("\nOptimizing threshold...")
    thresholds = np.arange(0.05, 0.96, 0.05)  # full grid; cosine/edit matches score well below 0.5
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = (match['id_left'], match['pred_id_right']) in true_matches
            left_has_truth = has_truth.get(match['id_left'], False)
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            if left_has_truth and not (pred_match and true_match):
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = (match['id_left'], match['pred_id_right']) in true_matches
        match['predicted_match'] = 1 if pred_match else 0
        match['is_correct'] = int(pred_match and true_match)

    return pd.DataFrame(all_matches)

# =============================================================================
# TF-IDF
# =============================================================================

def tfidf(df_left, df_right, df_mapping):
    """TF-IDF with threshold optimization"""
    print("\n🔍 Running TF-IDF with threshold optimization...")
    
    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)
    
    true_matches = create_ground_truth_set(df_mapping)
    
    # BUILD left_has_truth MAP
    has_truth = {}
    for (la, rb) in true_matches:
        has_truth[la] = True
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    all_texts = pd.concat([df_left['text'], df_right['text']])
    vectorizer.fit(all_texts)
    
    vectors_left = vectorizer.transform(df_left['text'])
    vectors_right = vectorizer.transform(df_right['text'])
    
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(vectors_left, vectors_right)
    
    # Build matches
    all_matches = []
    for idx_a in tqdm(range(len(df_left)), desc="TF-IDF matching"):
        id_a = str(df_left.iloc[idx_a][id_col_left])
        name_a = df_left.iloc[idx_a][name_col_left]
        
        # Get first true match for display
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        
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
    thresholds = np.arange(0.05, 0.96, 0.05)  # full grid; cosine/edit matches score well below 0.5
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = (match['id_left'], match['pred_id_right']) in true_matches
            left_has_truth = has_truth.get(match['id_left'], False)  
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            # FN: any record with ground truth that is not TP
            if left_has_truth and not (pred_match and true_match):  
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add predicted_match AND is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = (match['id_left'], match['pred_id_right']) in true_matches
        match['predicted_match'] = 1 if pred_match else 0
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches)

# =============================================================================
# SENTENCE TRANSFORMER
# =============================================================================

def sentence_transformer(df_left, df_right, df_mapping):
    """SentenceTransformer with threshold optimization"""
    print("\nRunning SentenceTransformer with threshold optimization...")
    
    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)
    
    true_matches = create_ground_truth_set(df_mapping)
    
    # BUILD left_has_truth MAP
    has_truth = {}
    for (la, rb) in true_matches:
        has_truth[la] = True
    
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
        
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        
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
    
    # Optimize threshold (same as TF-IDF)
    print("\nOptimizing threshold...")
    thresholds = np.arange(0.05, 0.96, 0.05)  # full grid; cosine/edit matches score well below 0.5
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = (match['id_left'], match['pred_id_right']) in true_matches
            left_has_truth = has_truth.get(match['id_left'], False)
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            # FN: any record with ground truth that is not TP
            if left_has_truth and not (pred_match and true_match):
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add predicted_match AND is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = (match['id_left'], match['pred_id_right']) in true_matches
        match['predicted_match'] = 1 if pred_match else 0
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches)

# =============================================================================
# OPENAI EMBEDDINGS
# =============================================================================

def get_openai_embeddings_batch(texts, model="text-embedding-3-small"):
    """Get OpenAI embeddings in batches"""
    client = get_openai_client()
    embeddings = []
    batch_size = 100
    
    for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI API calls"):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        time.sleep(0.1)
    
    return np.array(embeddings)

def openai_embeddings(df_left, df_right, df_mapping):
    """OpenAI embeddings with threshold optimization"""
    print("\nRunning OpenAI Embeddings with threshold optimization...")
    print("This will make OpenAI API calls (~$0.02-0.05)")
    
    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)
    
    true_matches = create_ground_truth_set(df_mapping)
    
    # BUILD left_has_truth MAP
    has_truth = {}
    for (la, rb) in true_matches:
        has_truth[la] = True
    
    # Get embeddings
    print("Getting OpenAI embeddings...")
    embeddings_left = get_openai_embeddings_batch(df_left['text'].tolist())
    embeddings_right = get_openai_embeddings_batch(df_right['text'].tolist())
    
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings_left, embeddings_right)
    
    # Build matches
    all_matches = []
    for idx_a in tqdm(range(len(df_left)), desc="Matching"):
        id_a = str(df_left.iloc[idx_a][id_col_left])
        name_a = df_left.iloc[idx_a][name_col_left]
        
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        
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
    thresholds = np.arange(0.05, 0.96, 0.05)  # full grid; cosine/edit matches score well below 0.5
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = (match['id_left'], match['pred_id_right']) in true_matches
            left_has_truth = has_truth.get(match['id_left'], False)
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            # FN: any record with ground truth that is not TP
            if left_has_truth and not (pred_match and true_match):
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # FIXED: Use set-based checking with threshold
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        id_a = match['id_left']
        id_b = match['pred_id_right']
        true_match = (id_a, id_b) in true_matches
        match['predicted_match'] = 1 if pred_match else 0  
        match['is_correct'] = int(pred_match and true_match)  
    
    return pd.DataFrame(all_matches)

# =============================================================================
# LLM MATCHING
# =============================================================================

# =============================================================================
# LLM PROMPTS
# =============================================================================

PROMPT_STANDARD = """You are an expert at entity matching. Determine if any candidate matches the query.

QUERY:
{query_text}

CANDIDATES:
{candidates_text}

GUIDELINES:
- Entities are SAME if they refer to the exact same real-world item/entity
- Consider: all available attributes (names, identifiers, descriptions, metadata)
- Account for: abbreviations, different word orders, minor variations, formatting differences
- Be strict: only match if you're confident they're the same entity

RESPONSE FORMAT:
If match exists:
Match: [number]
Confidence: [0.0 to 1.0]
Reasoning: [brief]

If no match:
Match: 0
Confidence: [0.0 to 1.0]
Reasoning: [why not]

Your response:"""

PROMPT_TOKEN_FALLBACK = """You are an expert at entity matching. The text below has been transformed so that words are no longer readable English, but the transformation is consistent: the same original word always maps to the same transformed token in both the query and candidates. Your task is to find which candidate refers to the same entity as the query by comparing shared tokens, character patterns, numbers, and structural similarity — NOT by trying to read the words as meaningful language.

QUERY:
{query_text}

CANDIDATES:
{candidates_text}

GUIDELINES:
- Focus on overlapping tokens, shared substrings, matching numbers, and structural patterns
- The same original word always becomes the same transformed token across query and candidates
- Distinctive tokens (model numbers, codes, identifiers) are the strongest signal
- Do NOT try to interpret the words semantically — they are deliberately scrambled
- Be strict: only match if token overlap and structure clearly indicate the same entity

RESPONSE FORMAT:
If match exists:
Match: [number]
Confidence: [0.0 to 1.0]
Reasoning: [brief]

If no match:
Match: 0
Confidence: [0.0 to 1.0]
Reasoning: [why not]

Your response:"""

# =============================================================================
# LLM API CALL HELPERS
# =============================================================================

def _retry_with_backoff(fn, max_retries=5, base_delay=2.0):
    """Retry a function with exponential backoff on rate limit / transient errors."""
    import time as _time
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            is_retryable = any(k in err_str for k in ['429', 'rate_limit', 'overloaded', '529', '503', 'timeout', 'Timeout', 'timed out', 'Connection', 'connection', 'index out of range', 'RemoteDisconnected', 'BrokenPipe', 'ConnectionReset'])
            if not is_retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    Rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
            _time.sleep(delay)

def _call_openai_gpt4o(prompt, model="gpt-4o"):
    """Call OpenAI API with GPT-4o and return (answer_text, cost)."""
    def _do():
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        # GPT-4o pricing: $2.50/MTok input, $10/MTok output
        cost = (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.0
        return answer, cost
    return _retry_with_backoff(_do)

def _call_openai(prompt, model="gpt-4o-mini"):
    """Call OpenAI API and return (answer_text, cost)."""
    def _do():
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        # GPT-4o-mini pricing: $0.15/MTok input, $0.60/MTok output
        cost = (tokens / 1_000_000) * 0.15
        return answer, cost
    return _retry_with_backoff(_do)

def _call_anthropic(prompt, model="claude-sonnet-4-6"):
    """Call Anthropic API and return (answer_text, cost)."""
    def _do():
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Claude Sonnet 4.6 pricing: $3/MTok input, $15/MTok output
        cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0
        return answer, cost
    return _retry_with_backoff(_do)

def _call_anthropic_haiku(prompt, model="claude-haiku-4-5-20251001"):
    """Call Anthropic Haiku API and return (answer_text, cost)."""
    def _do():
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Claude Haiku 4.5 pricing: $0.80/MTok input, $4.00/MTok output
        cost = (input_tokens / 1_000_000) * 0.80 + (output_tokens / 1_000_000) * 4.0
        return answer, cost
    return _retry_with_backoff(_do)

def _parse_llm_response(answer, top_candidates):
    """Parse Match/Confidence from LLM response text. Returns (matched_id, matched_name, confidence)."""
    match_num = 0
    confidence = 0.0

    lines = answer.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Match:'):
            try:
                match_num = int(''.join(c for c in line.split('Match:')[1] if c.isdigit()))
            except:
                pass
        elif line.startswith('Confidence:'):
            try:
                conf_str = line.split('Confidence:')[1].strip()
                confidence = float(''.join(c for c in conf_str if c.isdigit() or c == '.'))
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except:
                pass

    if match_num > 0 and match_num <= len(top_candidates):
        matched_id = top_candidates[match_num - 1][2]
        matched_name = top_candidates[match_num - 1][3]
        return matched_id, matched_name, confidence

    return None, None, confidence

def llm_match_single(query_row, df_candidates, id_col_candidates, name_col_candidates,
                     client, top_k=20, blocking_threshold=0.1, max_text_length=500,
                     use_tfidf_blocking=False, vectorizer=None, right_tfidf_matrix=None):
    """Match a single query using LLM with blocking
    
    OPTIMIZED: Now accepts pre-computed TF-IDF for 10-20x speedup!
    """
    query_text = query_row['text']
    
    # Choose blocking method
    if use_tfidf_blocking:
        # TF-IDF blocking (better semantic similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Use pre-computed TF-IDF if available (MUCH faster!)
        if vectorizer is not None and right_tfidf_matrix is not None:
            query_tfidf = vectorizer.transform([query_text])
            similarities = cosine_similarity(query_tfidf, right_tfidf_matrix).flatten()
        else:
            # Fallback to old method (slow but works)
            from sklearn.feature_extraction.text import TfidfVectorizer
            all_texts = [query_text] + df_candidates['text'].tolist()
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        candidate_indices = similarities.argsort()[::-1][:top_k]
        
        candidates_with_scores = []
        for idx in candidate_indices:
            row = df_candidates.iloc[idx]
            cand_id = str(row[id_col_candidates])
            cand_name = row[name_col_candidates]
            candidates_with_scores.append((similarities[idx], idx, cand_id, cand_name, row['text']))
    else:
        # Jaro-Winkler blocking (original)
        candidates_with_scores = []
        for idx, row in df_candidates.iterrows():
            candidate_text = row['text']
            jw_sim = jaro_winkler_similarity(query_text, candidate_text)
            
            if jw_sim >= blocking_threshold:
                cand_id = str(row[id_col_candidates])
                cand_name = row[name_col_candidates]
                candidates_with_scores.append((jw_sim, idx, cand_id, cand_name, candidate_text))
        
        candidates_with_scores.sort(reverse=True, key=lambda x: x[0])
    
    top_candidates = candidates_with_scores[:top_k]
    
    if not top_candidates:
        return None, None, 0.0, 0.0
    
    # Truncate
    if max_text_length > 0 and len(query_text) > max_text_length:
        query_text = query_text[:max_text_length] + "..."
    
    candidates_text = ""
    for i, (sim, idx, cand_id, cand_name, cand_text) in enumerate(top_candidates, 1):
        if max_text_length > 0 and len(cand_text) > max_text_length:
            cand_text = cand_text[:max_text_length] + "..."
        candidates_text += f"{i}. {cand_text}\n\n"
    
    prompt = f"""You are an expert at entity matching. Determine if any candidate matches the query.

QUERY:
{query_text}

CANDIDATES:
{candidates_text}

GUIDELINES:
- Entities are SAME if they refer to the exact same real-world item/entity
- Consider: all available attributes (names, identifiers, descriptions, metadata)
- Account for: abbreviations, different word orders, minor variations, formatting differences
- Be strict: only match if you're confident they're the same entity

RESPONSE FORMAT:
If match exists:
Match: [number]
Confidence: [0.0 to 1.0]
Reasoning: [brief]

If no match:
Match: 0
Confidence: [0.0 to 1.0]
Reasoning: [why not]

Your response:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = (tokens / 1_000_000) * 0.15
        
        match_num = 0
        confidence = 0.0
        
        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Match:'):
                try:
                    match_num = int(''.join(c for c in line.split('Match:')[1] if c.isdigit()))
                except:
                    pass
            elif line.startswith('Confidence:'):
                try:
                    conf_str = line.split('Confidence:')[1].strip()
                    confidence = float(''.join(c for c in conf_str if c.isdigit() or c == '.'))
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                except:
                    pass
        
        if match_num > 0 and match_num <= len(top_candidates):
            matched_id = top_candidates[match_num - 1][2]
            matched_name = top_candidates[match_num - 1][3]
            return matched_id, matched_name, confidence, cost
        
        return None, None, confidence, cost
        
    except Exception as e:
        print(f"    Error: {str(e)}")
        return None, None, 0.0, 0.0

def llm(df_left, df_right, df_mapping):
    """LLM matching with confidence threshold optimization"""
    print("\nRunning LLM matching with confidence optimization...")
    print("This is EXPENSIVE (~$0.50-2.00 per transformation)")
    
    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)
    
    true_matches = create_ground_truth_set(df_mapping)
    client = get_openai_client()
    
    # BUILD left_has_truth MAP
    has_truth = {}
    for (la, rb) in true_matches:
        has_truth[la] = True
    
    # Parameters - ADAPTIVE based on dataset size
    total_candidates = len(df_right)
    if total_candidates < 2000:
        top_k = 50  # Small datasets (Abt-Buy, Amazon-Google)
    elif total_candidates < 10000:
        top_k = 100  # Medium datasets (DBLP-ACM)
    else:
        top_k = 200  # Large datasets (DBLP-Scholar)
    
    blocking_threshold = 0.3  # For JW blocking (not used with TF-IDF)
    max_text_length = 2000  # Increased for complex descriptions/papers
    use_tfidf_blocking = True  # BEST: TF-IDF finds better candidates
    
    print(f"Parameters: top_k={top_k} ({100*top_k/total_candidates:.2f}% of {total_candidates:,} candidates), blocking={'TF-IDF' if use_tfidf_blocking else f'JW≥{blocking_threshold}'}")
    
    # Match each query
    all_matches = []
    total_cost = 0.0
    
    # PERFORMANCE FIX: Pre-compute TF-IDF for blocking (10-20x speedup!)
    vectorizer = None
    right_tfidf_matrix = None
    if use_tfidf_blocking:
        print("Pre-computing TF-IDF vectors for all candidates...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer()
        # Fit on both left and right to ensure same vocabulary
        all_texts_for_vocab = pd.concat([df_left['text'], df_right['text']])
        vectorizer.fit(all_texts_for_vocab)
        
        # Transform right dataset once (instead of every query!)
        right_tfidf_matrix = vectorizer.transform(df_right['text'])
        print(f"TF-IDF cached: {right_tfidf_matrix.shape} - now 10-20x faster!")
    
    for idx_a, row_a in tqdm(df_left.iterrows(), total=len(df_left), desc="LLM matching"):
        id_a = str(row_a[id_col_left])
        name_a = row_a[name_col_left]
        
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        
        matched_id, matched_name, confidence, cost = llm_match_single(
            row_a, df_right, id_col_right, name_col_right, client,
            top_k, blocking_threshold, max_text_length, use_tfidf_blocking,
            vectorizer, right_tfidf_matrix 
        )
        
        total_cost += cost
        
        all_matches.append({
            'id_left': id_a,
            'left_name': name_a,
            'true_id_right': true_id_b if true_id_b else '',
            'pred_id_right': matched_id if matched_id else '',
            'pred_right_name': matched_name if matched_name else '',
            'similarity_score': confidence
        })
    
    print(f"\nTotal cost: ${total_cost:.2f}")
    
    # Optimize confidence threshold
    print("\nOptimizing confidence threshold...")
    thresholds = np.arange(0.05, 0.96, 0.05)  # full grid; cosine/edit matches score well below 0.5
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = (match['pred_id_right'] != '' and 
                         match['similarity_score'] >= threshold)
            id_a = match['id_left']
            id_b = match['pred_id_right']
            true_match = (id_a, id_b) in true_matches  
            left_has_truth = has_truth.get(id_a, False)  
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            # FN: any record with ground truth that is not TP
            if left_has_truth and not (pred_match and true_match):
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Add predicted_match AND use set-based checking
    for match in all_matches:
        pred_match = (match['pred_id_right'] != '' and 
                     match['similarity_score'] >= best_threshold)
        id_a = match['id_left']
        id_b = match['pred_id_right']
        true_match = (id_a, id_b) in true_matches 
        match['predicted_match'] = 1 if pred_match else 0  
        match['is_correct'] = int(pred_match and true_match)
    
    return pd.DataFrame(all_matches)

# =============================================================================
# GENERALIZED LLM MATCHING (supports multiple providers and prompts)
# =============================================================================

def _save_checkpoint(all_matches, total_cost, checkpoint_path):
    """Save current progress to a parquet checkpoint file."""
    from pathlib import Path
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_matches)
    df['total_cost_usd'] = total_cost
    df.to_parquet(checkpoint_path, index=False)

def _llm_match_single_generic(query_row, df_candidates, id_col_candidates, name_col_candidates,
                               top_k, max_text_length, vectorizer, right_tfidf_matrix,
                               api_caller, prompt_template):
    """Generic single-query LLM match. Works with any API caller and prompt."""
    query_text = query_row['text']

    # TF-IDF blocking
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    query_tfidf = vectorizer.transform([query_text])
    similarities = cos_sim(query_tfidf, right_tfidf_matrix).flatten()
    candidate_indices = similarities.argsort()[::-1][:top_k]

    candidates_with_scores = []
    for idx in candidate_indices:
        row = df_candidates.iloc[idx]
        cand_id = str(row[id_col_candidates])
        cand_name = row[get_display_column(df_candidates)]
        candidates_with_scores.append((similarities[idx], idx, cand_id, cand_name, row['text']))

    top_candidates = candidates_with_scores[:top_k]
    if not top_candidates:
        return None, None, 0.0, 0.0

    # Truncate text
    if max_text_length > 0 and len(query_text) > max_text_length:
        query_text = query_text[:max_text_length] + "..."

    candidates_text = ""
    for i, (sim, idx, cand_id, cand_name, cand_text) in enumerate(top_candidates, 1):
        if max_text_length > 0 and len(cand_text) > max_text_length:
            cand_text = cand_text[:max_text_length] + "..."
        candidates_text += f"{i}. {cand_text}\n\n"

    prompt = prompt_template.format(query_text=query_text, candidates_text=candidates_text)

    try:
        answer, cost = api_caller(prompt)
        matched_id, matched_name, confidence = _parse_llm_response(answer, top_candidates)
        return matched_id, matched_name, confidence, cost
    except Exception as e:
        print(f"    Error: {str(e)}")
        return None, None, 0.0, 0.0


def _run_llm_method(df_left, df_right, df_mapping, api_caller, method_label, prompt_template,
                    checkpoint_path=None):
    """Full LLM matching pipeline: blocking + LLM classification + threshold optimization.
    Shared by all LLM variants (GPT-4o-mini, Claude, token-fallback prompts).

    Saves a checkpoint after every record so progress survives crashes."""
    import time as _time
    import json

    print(f"\nRunning {method_label}...")
    print("This is EXPENSIVE - check cost estimates before running")

    df_left = prepare_text_column(df_left.copy(), title_only=TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=TITLE_ONLY_MODE)

    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = get_display_column(df_left)
    name_col_right = get_display_column(df_right)

    true_matches = create_ground_truth_set(df_mapping)
    has_truth = {la: True for (la, rb) in true_matches}

    # Adaptive top_k
    total_candidates = len(df_right)
    if total_candidates < 2000:
        top_k = 50
    elif total_candidates < 10000:
        top_k = 100
    else:
        top_k = 200
    max_text_length = 2000

    print(f"Parameters: top_k={top_k} ({100*top_k/total_candidates:.2f}% of {total_candidates:,} candidates)")

    # Pre-compute TF-IDF blocking vectors
    print("Pre-computing TF-IDF vectors for blocking...")
    from sklearn.feature_extraction.text import TfidfVectorizer as _TV
    vectorizer = _TV()
    vectorizer.fit(pd.concat([df_left['text'], df_right['text']]))
    right_tfidf_matrix = vectorizer.transform(df_right['text'])

    # Resume from checkpoint if it exists
    # Only keep records that have real results (non-empty prediction OR non-zero
    # similarity score).  Failed API calls produce pred_id_right='' AND
    # similarity_score=0; dropping them lets the retry loop pick them up again.
    all_matches = []
    total_cost = 0.0
    completed_ids = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_parquet(checkpoint_path)
        has_result = (checkpoint_df['pred_id_right'].astype(str).ne('') |
                      checkpoint_df['similarity_score'].fillna(0).gt(0))
        checkpoint_df = checkpoint_df[has_result]
        all_matches = checkpoint_df.to_dict('records')
        completed_ids = set(checkpoint_df['id_left'].astype(str))
        total_cost = float(checkpoint_df['total_cost_usd'].iloc[0]) if 'total_cost_usd' in checkpoint_df.columns and len(checkpoint_df) > 0 else 0.0
        print(f"Resumed from checkpoint: {len(completed_ids)} records with valid results (dropped failed rows)")

    wall_start = _time.time()
    records_this_run = 0

    for idx_a, row_a in tqdm(df_left.iterrows(), total=len(df_left), desc=f"{method_label}"):
        id_a = str(row_a[id_col_left])

        # Skip already-completed records
        if id_a in completed_ids:
            continue

        name_a = row_a[name_col_left]
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None

        matched_id, matched_name, confidence, cost = _llm_match_single_generic(
            row_a, df_right, id_col_right, name_col_right,
            top_k, max_text_length, vectorizer, right_tfidf_matrix,
            api_caller, prompt_template
        )
        total_cost += cost
        records_this_run += 1

        all_matches.append({
            'id_left': id_a,
            'left_name': name_a,
            'true_id_right': true_id_b if true_id_b else '',
            'pred_id_right': matched_id if matched_id else '',
            'pred_right_name': matched_name if matched_name else '',
            'similarity_score': confidence
        })

        # Save checkpoint every 10 records
        if checkpoint_path and records_this_run % 10 == 0:
            _save_checkpoint(all_matches, total_cost, checkpoint_path)

    # Final checkpoint save
    if checkpoint_path:
        _save_checkpoint(all_matches, total_cost, checkpoint_path)

    wall_elapsed = _time.time() - wall_start
    print(f"\nTotal cost: ${total_cost:.2f}")
    print(f"Wall time: {wall_elapsed:.0f}s ({wall_elapsed/60:.1f}min)")

    # Optimize confidence threshold
    print("\nOptimizing confidence threshold...")
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = (match['pred_id_right'] != '' and
                         match['similarity_score'] >= threshold)
            id_a = match['id_left']
            id_b = match['pred_id_right']
            true_match = (id_a, id_b) in true_matches
            left_has_truth = has_truth.get(id_a, False)

            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            if left_has_truth and not (pred_match and true_match):
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

    for match in all_matches:
        pred_match = (match['pred_id_right'] != '' and
                     match['similarity_score'] >= best_threshold)
        true_match = (match['id_left'], match['pred_id_right']) in true_matches
        match['predicted_match'] = 1 if pred_match else 0
        match['is_correct'] = int(pred_match and true_match)

    results_df = pd.DataFrame(all_matches)
    results_df['total_cost_usd'] = total_cost
    results_df['wall_time_seconds'] = wall_elapsed
    return results_df


# --- New LLM method wrappers ---

def llm_claude(df_left, df_right, df_mapping, output_path=None):
    """Claude Sonnet 4.6 with TF-IDF blocking (standard prompt)."""
    return _run_llm_method(
        df_left, df_right, df_mapping,
        api_caller=_call_anthropic,
        method_label="LLM (Claude Sonnet 4.6)",
        prompt_template=PROMPT_STANDARD,
        checkpoint_path=output_path
    )

def llm_gpt4o(df_left, df_right, df_mapping, output_path=None):
    """GPT-4o with TF-IDF blocking (standard prompt)."""
    return _run_llm_method(
        df_left, df_right, df_mapping,
        api_caller=_call_openai_gpt4o,
        method_label="LLM (GPT-4o)",
        prompt_template=PROMPT_STANDARD,
        checkpoint_path=output_path
    )

def llm_gpt4o_mini_token_fallback(df_left, df_right, df_mapping, output_path=None):
    """GPT-4o-mini with TF-IDF blocking (token-fallback prompt)."""
    return _run_llm_method(
        df_left, df_right, df_mapping,
        api_caller=_call_openai,
        method_label="LLM (GPT-4o-mini, token-fallback prompt)",
        prompt_template=PROMPT_TOKEN_FALLBACK,
        checkpoint_path=output_path
    )

def llm_claude_token_fallback(df_left, df_right, df_mapping, output_path=None):
    """Claude Sonnet 4.6 with TF-IDF blocking (token-fallback prompt)."""
    return _run_llm_method(
        df_left, df_right, df_mapping,
        api_caller=_call_anthropic,
        method_label="LLM (Claude Sonnet 4.6, token-fallback prompt)",
        prompt_template=PROMPT_TOKEN_FALLBACK,
        checkpoint_path=output_path
    )

def llm_haiku(df_left, df_right, df_mapping, output_path=None):
    """Claude Haiku 4.5 with TF-IDF blocking (standard prompt)."""
    return _run_llm_method(
        df_left, df_right, df_mapping,
        api_caller=_call_anthropic_haiku,
        method_label="LLM (Claude Haiku 4.5)",
        prompt_template=PROMPT_STANDARD,
        checkpoint_path=output_path
    )

def llm_haiku_token_fallback(df_left, df_right, df_mapping, output_path=None):
    """Claude Haiku 4.5 with TF-IDF blocking (token-fallback prompt)."""
    return _run_llm_method(
        df_left, df_right, df_mapping,
        api_caller=_call_anthropic_haiku,
        method_label="LLM (Claude Haiku 4.5, token-fallback prompt)",
        prompt_template=PROMPT_TOKEN_FALLBACK,
        checkpoint_path=output_path
    )


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
    'openai_embeddings': openai_embeddings,
    'llm': llm,
    'llm_gpt4o': llm_gpt4o,
    'llm_claude': llm_claude,
    'llm_gpt4o_mini_token_fallback': llm_gpt4o_mini_token_fallback,
    'llm_claude_token_fallback': llm_claude_token_fallback,
    'llm_haiku': llm_haiku,
    'llm_haiku_token_fallback': llm_haiku_token_fallback,
}

if __name__ == "__main__":
    print("All 8 methods loaded with FINAL fixes:")
    print("  - One-to-many ground truth support")
    print("  - Correct FN counting")
    print("  - Consistent threshold logic")
    print("  - Text normalization (.lower().strip())")
    print("  - predicted_match column for evaluation")
    print("\nMethods:")
    for i, name in enumerate(METHODS.keys(), 1):
        print(f"  {i}. {name}")
