"""
ENTITY MATCHING METHODS - FINAL DEFINITIVE VERSION
===================================================
✅ One-to-many ground truth support (set-based)
✅ Correct FN counting (uses left_has_truth)
✅ Consistent threshold logic across all methods
✅ Text preprocessing (.lower().strip())
✅ predicted_match column for clear evaluation

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
    """Combine text columns and normalize"""
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
    
    # ✅ NORMALIZE: lowercase and strip (matches notebook)
    df['text'] = df['text'].str.lower().str.strip()
    
    return df

def create_ground_truth_dict(df_mapping):
    """Create dict mapping left_id -> right_id from ground truth
    ⚠️ WARNING: Overwrites if one left has multiple rights!
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
    ✅ Preserves ALL valid one-to-many relationships."""
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
    name_col_left = 'name' if 'name' in df_left.columns else df_left.columns[1]
    name_col_right = 'name' if 'name' in df_right.columns else df_right.columns[1]
    
    # ✅ BUILD left_has_truth MAP
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
    thresholds = np.arange(0.50, 0.96, 0.05)
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
            
            # ✅ FIX: FN is ANY record with ground truth that's not TP
            if left_has_truth and not (pred_match and true_match):
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # ✅ Add predicted_match AND is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = (match['id_left'], match['pred_id_right']) in true_matches_set
        match['predicted_match'] = 1 if pred_match else 0  # ✅ NEW COLUMN
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches), best_threshold, best_f1

# =============================================================================
# STRING DISTANCE METHODS
# =============================================================================

def string_distance_method(df_left, df_right, df_mapping, similarity_func, method_name):
    """Generic string distance with threshold optimization"""
    df_left = prepare_text_column(df_left.copy())
    df_right = prepare_text_column(df_right.copy())
    
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

def soft_tfidf(df_left, df_right, df_mapping):
    return string_distance_method(df_left, df_right, df_mapping,
                                  soft_tfidf_similarity, "Soft-TF-IDF")

# =============================================================================
# TF-IDF
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
    
    true_matches = create_ground_truth_set(df_mapping)
    
    # ✅ BUILD left_has_truth MAP
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
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = match['similarity_score'] >= threshold
            true_match = (match['id_left'], match['pred_id_right']) in true_matches
            left_has_truth = has_truth.get(match['id_left'], False)  # ✅ FIXED
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            # FN: any record with ground truth that is not TP
            if left_has_truth and not (pred_match and true_match):  # ✅ CORRECT FN COUNTING
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # ✅ Add predicted_match AND is_correct
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        true_match = (match['id_left'], match['pred_id_right']) in true_matches
        match['predicted_match'] = 1 if pred_match else 0  # ✅ NEW COLUMN
        match['is_correct'] = 1 if (pred_match and true_match) else 0
    
    return pd.DataFrame(all_matches)

# =============================================================================
# SENTENCE TRANSFORMER
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
    
    true_matches = create_ground_truth_set(df_mapping)
    
    # ✅ BUILD left_has_truth MAP
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
    thresholds = np.arange(0.50, 0.96, 0.05)
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
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # ✅ Add predicted_match AND is_correct
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
    print("\n🔍 Running OpenAI Embeddings with threshold optimization...")
    print("⚠️  This will make OpenAI API calls (~$0.02-0.05)")
    
    df_left = prepare_text_column(df_left.copy())
    df_right = prepare_text_column(df_right.copy())
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = 'name' if 'name' in df_left.columns else df_left.columns[1]
    name_col_right = 'name' if 'name' in df_right.columns else df_right.columns[1]
    
    true_matches = create_ground_truth_set(df_mapping)
    
    # ✅ BUILD left_has_truth MAP
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
    thresholds = np.arange(0.50, 0.96, 0.05)
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
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # ✅ FIXED: Use set-based checking with threshold
    for match in all_matches:
        pred_match = match['similarity_score'] >= best_threshold
        id_a = match['id_left']
        id_b = match['pred_id_right']
        true_match = (id_a, id_b) in true_matches
        match['predicted_match'] = 1 if pred_match else 0  # ✅ NEW COLUMN
        match['is_correct'] = int(pred_match and true_match)  # ✅ FIXED
    
    return pd.DataFrame(all_matches)

# =============================================================================
# LLM MATCHING
# =============================================================================

def llm_match_single(query_row, df_candidates, id_col_candidates, name_col_candidates, 
                     client, top_k=20, blocking_threshold=0.1, max_text_length=500, 
                     use_tfidf_blocking=False):
    """Match a single query using LLM with blocking"""
    query_text = query_row['text']
    
    # Choose blocking method
    if use_tfidf_blocking:
        # TF-IDF blocking (better semantic similarity)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
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
- Products are SAME if they refer to same model/item
- Consider: brand, model number, key specifications
- Account for: abbreviations, different word orders

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
    print("\n🔍 Running LLM matching with confidence optimization...")
    print("⚠️  This is EXPENSIVE (~$0.50-2.00 per transformation)")
    
    df_left = prepare_text_column(df_left.copy())
    df_right = prepare_text_column(df_right.copy())
    
    id_col_left = df_left.columns[0]
    id_col_right = df_right.columns[0]
    name_col_left = 'name' if 'name' in df_left.columns else df_left.columns[1]
    name_col_right = 'name' if 'name' in df_right.columns else df_right.columns[1]
    
    true_matches = create_ground_truth_set(df_mapping)
    client = get_openai_client()
    
    # ✅ BUILD left_has_truth MAP
    has_truth = {}
    for (la, rb) in true_matches:
        has_truth[la] = True
    
    # Parameters - TRULY improved with OPTION for TF-IDF blocking
    top_k = 50  # ✅ More candidates
    blocking_threshold = 0.3  # ✅ MORE permissive (only for JW blocking)
    max_text_length = 1000  # ✅ Full descriptions
    use_tfidf_blocking = True  # ✅ BEST: TF-IDF finds better candidates (F1 ≥ 0.80!)
    
    print(f"Parameters: top_k={top_k}, blocking={'TF-IDF' if use_tfidf_blocking else f'JW≥{blocking_threshold}'}")
    
    # Match each query
    all_matches = []
    total_cost = 0.0
    
    for idx_a, row_a in tqdm(df_left.iterrows(), total=len(df_left), desc="LLM matching"):
        id_a = str(row_a[id_col_left])
        name_a = row_a[name_col_left]
        
        true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
        true_id_b = true_matches_for_a[0] if true_matches_for_a else None
        
        matched_id, matched_name, confidence, cost = llm_match_single(
            row_a, df_right, id_col_right, name_col_right, client,
            top_k, blocking_threshold, max_text_length, use_tfidf_blocking
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
    
    print(f"\n💰 Total cost: ${total_cost:.2f}")
    
    # Optimize confidence threshold
    print("\nOptimizing confidence threshold...")
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        tp = fp = fn = 0
        for match in all_matches:
            pred_match = (match['pred_id_right'] != '' and 
                         match['similarity_score'] >= threshold)
            id_a = match['id_left']
            id_b = match['pred_id_right']
            true_match = (id_a, id_b) in true_matches  # ✅ FIXED: set-based
            left_has_truth = has_truth.get(id_a, False)  # ✅ FIXED
            
            if pred_match and true_match:
                tp += 1
            elif pred_match and not true_match:
                fp += 1
            # FN: any record with ground truth that is not TP
            if left_has_truth and not (pred_match and true_match):  # ✅ CORRECT FN COUNTING
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # ✅ FIXED: Add predicted_match AND use set-based checking
    for match in all_matches:
        pred_match = (match['pred_id_right'] != '' and 
                     match['similarity_score'] >= best_threshold)
        id_a = match['id_left']
        id_b = match['pred_id_right']
        true_match = (id_a, id_b) in true_matches  # ✅ FIXED
        match['predicted_match'] = 1 if pred_match else 0  # ✅ NEW COLUMN
        match['is_correct'] = int(pred_match and true_match)  # ✅ FIXED
    
    return pd.DataFrame(all_matches)

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
    'llm': llm
}

if __name__ == "__main__":
    print("✅ All 8 methods loaded with FINAL fixes:")
    print("  - One-to-many ground truth support")
    print("  - Correct FN counting")
    print("  - Consistent threshold logic")
    print("  - Text normalization (.lower().strip())")
    print("  - predicted_match column for evaluation")
    print("\nMethods:")
    for i, name in enumerate(METHODS.keys(), 1):
        print(f"  {i}. {name}")
