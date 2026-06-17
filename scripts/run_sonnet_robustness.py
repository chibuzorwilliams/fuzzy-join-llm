"""
Build a consistent 100-record sample for the llm_robustness figure.

All three models (GPT-4o-mini, GPT-4o, Claude Sonnet) are evaluated on the
SAME 100 left-side entity IDs sampled from the amazon-google mapping file
(random_state=42).

  - llm / llm_gpt4o: results read from existing parquets (no API calls)
  - llm_claude original: read from existing original.parquet
  - llm_claude ablations: new Claude Sonnet API calls (3 × 100 calls)

Output: results/robustness_sample.csv  (cv_results.csv-compatible format)

Usage:
    .venv/bin/python scripts/run_sonnet_robustness.py
    .venv/bin/python scripts/run_sonnet_robustness.py --n 100
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(__file__).parent))
from methods import (
    _llm_match_single_generic,
    create_ground_truth_set,
    get_display_column,
    get_anthropic_client,
    prepare_text_column,
    PROMPT_STANDARD,
)


def _call_sonnet(prompt, model='claude-sonnet-4-6'):
    """Call Sonnet without retrying on empty-content responses.

    Anthropic's safety filters silently refuse ciphered/encoded text by returning
    an empty content list. The standard _call_anthropic treats this as a retryable
    error and waits 62 s per record before giving up. This wrapper returns
    (None, 0.0) immediately on empty content so the script stays fast.
    """
    client = get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        messages=[{'role': 'user', 'content': prompt}],
    )
    if not response.content:
        return None, 0.0
    answer = response.content[0].text.strip()
    cost = (
        (response.usage.input_tokens  / 1_000_000) * 3.0 +
        (response.usage.output_tokens / 1_000_000) * 15.0
    )
    return answer, cost

DATASET      = 'amazon-google'
MODELS       = ['llm', 'llm_gpt4o', 'llm_claude']
TRANSFORMATIONS = ['original', 'ciphered_letters', 'ciphered_words', 'scrambled']
DATA_DIR     = Path('data/amazon-google')
MAPPING_FILE = DATA_DIR / 'data_original/Amazon_GoogleProducts_perfectMapping.csv'
RESULTS_DIR  = Path('results/amazon-google')
OUT_FILE     = Path('results/robustness_sample.csv')


def compute_f1(records):
    total       = len(records)
    correct     = sum(r['is_correct'] for r in records)
    predictions = sum(1 for r in records if r.get('pred_id_right', ''))
    precision   = correct / predictions if predictions > 0 else 0.0
    recall      = correct / total       if total > 0      else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def load_parquet_sample(method, transformation, sample_ids):
    """Read existing parquet, filter to sample_ids, return records list."""
    pq_path = RESULTS_DIR / method / f'{transformation}.parquet'
    df = pd.read_parquet(pq_path)
    df = df[df['id_left'].astype(str).isin(sample_ids)].copy()
    if len(df) < len(sample_ids):
        print(f'  WARNING: only {len(df)}/{len(sample_ids)} sample IDs found in {pq_path}')
    return df[['id_left', 'true_id_right', 'pred_id_right', 'is_correct']].to_dict('records')


def run_sonnet_ablation(transformation, sample_ids, df_mapping, true_matches):
    """Run Claude Sonnet on sample_ids for a given transformation."""
    left_file  = DATA_DIR / f'data_test/amazon_{transformation}.csv'
    right_file = DATA_DIR / f'data_test/google_{transformation}.csv'

    map_r         = df_mapping.columns[1]
    matched_right = set(df_mapping[map_r].astype(str))

    for enc in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df_left  = pd.read_csv(left_file,  encoding=enc)
            df_right = pd.read_csv(right_file, encoding=enc)
            break
        except Exception:
            continue

    left_id_col  = df_left.columns[0]
    right_id_col = df_right.columns[0]

    df_right    = df_right[df_right[right_id_col].astype(str).isin(matched_right)].copy()
    df_left_sub = df_left[df_left[left_id_col].astype(str).isin(sample_ids)].copy()

    df_left_prep  = prepare_text_column(df_left_sub.copy())
    df_right_prep = prepare_text_column(df_right.copy())

    id_col_right   = df_right_prep.columns[0]
    name_col_right = get_display_column(df_right_prep)

    vectorizer  = TfidfVectorizer()
    vectorizer.fit(pd.concat([df_left_prep['text'], df_right_prep['text']]))
    right_tfidf = vectorizer.transform(df_right_prep['text'])
    # Keep prompts small (~3k tokens) to stay under Sonnet's tokens-per-minute limit.
    top_k          = 10
    max_text_length = 400

    records = []
    for _, row_a in df_left_prep.iterrows():
        id_a   = str(row_a[df_left_prep.columns[0]])
        name_a = row_a[get_display_column(df_left_prep)]

        matched_id, matched_name, confidence, _ = _llm_match_single_generic(
            row_a, df_right_prep, id_col_right, name_col_right,
            top_k, max_text_length, vectorizer, right_tfidf,
            _call_sonnet, PROMPT_STANDARD
        )
        time.sleep(2)

        is_correct = matched_id is not None and (id_a, matched_id) in true_matches
        status     = '✓' if is_correct else ('✗' if matched_id else '—')
        print(f'    [{status}] conf={confidence:.2f}  {name_a[:40]!r} -> {matched_name!r}')

        records.append({
            'id_left':       id_a,
            'true_id_right': next((r for l, r in true_matches if l == id_a), ''),
            'pred_id_right': matched_id or '',
            'is_correct':    int(is_correct),
        })

    return records


def run(n_cases: int):
    df_mapping   = pd.read_csv(MAPPING_FILE)
    map_l        = df_mapping.columns[0]
    all_left_ids = df_mapping[map_l].astype(str).unique().tolist()

    # Sample the same N entity IDs used across all models
    sample_series = pd.Series(all_left_ids).sample(
        n=min(n_cases, len(all_left_ids)), random_state=42
    )
    sample_ids = set(sample_series.values)
    print(f'Sampled {len(sample_ids)} entity IDs (random_state=42)\n')

    true_matches = create_ground_truth_set(df_mapping)
    rows_out     = []

    # Sonnet's safety filters block ciphered_letters (letter-substitution cipher
    # looks like encoded/obfuscated content). Skip it rather than paying for
    # 100 API calls that return nothing.
    # Sonnet's safety filters block all adversarial text transformations.
    SONNET_SKIP = {'ciphered_letters', 'ciphered_words', 'scrambled'}

    for method in MODELS:
        for transformation in TRANSFORMATIONS:
            print(f'── {method} / {transformation}')

            if method == 'llm_claude' and transformation in SONNET_SKIP:
                print(f'   SKIPPED — Sonnet safety filter blocks this transformation\n')
                rows_out.append({
                    'dataset':        DATASET,
                    'method':         method,
                    'transformation': transformation,
                    'f1_mean':        float('nan'),
                    'f1_std':         float('nan'),
                    'precision':      float('nan'),
                    'recall':         float('nan'),
                })
                continue

            if method == 'llm_claude' and transformation != 'original':
                # Run new Sonnet API calls
                records = run_sonnet_ablation(
                    transformation, sample_ids, df_mapping, true_matches
                )
            else:
                # Read from existing parquet
                records = load_parquet_sample(method, transformation, sample_ids)

            f1, prec, rec = compute_f1(records)
            correct = sum(r['is_correct'] for r in records)
            print(f'   {correct}/{len(records)} correct  '
                  f'F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}\n')

            rows_out.append({
                'dataset':        DATASET,
                'method':         method,
                'transformation': transformation,
                'f1_mean':        f1,
                'f1_std':         0.0,
                'precision':      prec,
                'recall':         rec,
            })

    df_out = pd.DataFrame(rows_out)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_FILE, index=False)
    print(f'Saved: {OUT_FILE}')
    print(df_out.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100,
                        help='Number of records to sample (default 100)')
    args = parser.parse_args()
    run(args.n)
