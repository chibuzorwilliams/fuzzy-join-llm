"""
Mini sanity-check for Claude Haiku results on Amazon-Google.

Runs llm_haiku (standard prompt) and llm_haiku_token_fallback (token-fallback
prompt) on a random sample of N_CASES matched records across all four
transformations. Results are saved to results/test_haiku/ — existing results
are NOT touched.

Usage:
    python scripts/test_haiku_mini.py          # 10 cases (default)
    python scripts/test_haiku_mini.py --n 20   # 20 cases
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from methods import (
    _call_anthropic_haiku,
    _llm_match_single_generic,
    _parse_llm_response,
    _run_llm_method,
    create_ground_truth_set,
    get_display_column,
    prepare_text_column,
    PROMPT_STANDARD,
    PROMPT_TOKEN_FALLBACK,
)
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET = "amazon-google"
TRANSFORMATIONS = ["original", "ciphered_letters", "ciphered_words", "scrambled"]
DATA_DIR = Path("data/amazon-google")
MAPPING_FILE = DATA_DIR / "data_original/Amazon_GoogleProducts_perfectMapping.csv"
BACKUP_DIR = Path("archive/haiku_backup")
OUT_DIR = Path("results/test_haiku")

METHODS = {
    "llm_haiku":               (PROMPT_STANDARD,      "Claude Haiku — Standard"),
    "llm_haiku_token_fallback": (PROMPT_TOKEN_FALLBACK, "Claude Haiku — Token Fallback"),
}


def load_transformation(transformation: str, n_cases: int):
    if transformation == "original":
        left_file  = DATA_DIR / "data_original/amazon.csv"
        right_file = DATA_DIR / "data_original/google.csv"
    else:
        left_file  = DATA_DIR / f"data_test/amazon_{transformation}.csv"
        right_file = DATA_DIR / f"data_test/google_{transformation}.csv"

    for enc in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            df_left  = pd.read_csv(left_file,  encoding=enc)
            df_right = pd.read_csv(right_file, encoding=enc)
            break
        except Exception:
            continue

    df_mapping = pd.read_csv(MAPPING_FILE)
    left_col   = df_left.columns[0]
    right_col  = df_right.columns[0]
    map_l      = df_mapping.columns[0]
    map_r      = df_mapping.columns[1]

    matched_left  = set(df_mapping[map_l].astype(str))
    matched_right = set(df_mapping[map_r].astype(str))

    df_left  = df_left[df_left[left_col].astype(str).isin(matched_left)].copy()
    df_right = df_right[df_right[right_col].astype(str).isin(matched_right)].copy()

    # Sample n_cases from the left side (reproducible)
    df_left_sample = df_left.sample(n=min(n_cases, len(df_left)), random_state=42)

    return df_left_sample, df_right, df_mapping


def run_mini(n_cases: int):
    # --- backup existing haiku parquets ---
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for method in ["llm_haiku", "llm_haiku_token_fallback"]:
        src = Path(f"results/{DATASET}/{method}")
        if src.exists():
            dst = BACKUP_DIR / method
            dst.mkdir(parents=True, exist_ok=True)
            for pq in src.glob("*.parquet"):
                shutil.copy2(pq, dst / pq.name)
            print(f"Backed up {src} → {dst}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for transformation in TRANSFORMATIONS:
        print(f"\n{'='*70}")
        print(f"Transformation: {transformation}")
        print(f"{'='*70}")

        df_left_sample, df_right, df_mapping = load_transformation(transformation, n_cases)

        # Prepare text columns
        df_left_prep  = prepare_text_column(df_left_sample.copy())
        df_right_prep = prepare_text_column(df_right.copy())

        id_col_left  = df_left_prep.columns[0]
        id_col_right = df_right_prep.columns[0]
        name_col_right = get_display_column(df_right_prep)

        true_matches = create_ground_truth_set(df_mapping)

        # Pre-compute TF-IDF blocking vectors once per transformation
        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([df_left_prep["text"], df_right_prep["text"]]))
        right_tfidf = vectorizer.transform(df_right_prep["text"])

        total_candidates = len(df_right_prep)
        top_k = 50 if total_candidates < 2000 else 100

        for method_key, (prompt_template, label) in METHODS.items():
            print(f"\n--- {label} ---")
            correct = total = 0

            rows = []
            for _, row_a in df_left_prep.iterrows():
                id_a   = str(row_a[id_col_left])
                name_a = row_a[get_display_column(df_left_prep)]

                true_matches_for_a = [r for (l, r) in true_matches if l == id_a]
                true_id_b = true_matches_for_a[0] if true_matches_for_a else None

                matched_id, matched_name, confidence, cost = _llm_match_single_generic(
                    row_a, df_right_prep, id_col_right, name_col_right,
                    top_k, 2000, vectorizer, right_tfidf,
                    _call_anthropic_haiku, prompt_template
                )

                is_match   = matched_id is not None
                is_correct = is_match and (id_a, matched_id) in true_matches
                total  += 1
                correct += int(is_correct)

                status = "✓" if is_correct else ("✗" if is_match else "—")
                print(f"  [{status}] conf={confidence:.2f}  left={name_a[:40]!r}  pred={matched_name!r}")

                rows.append({
                    "transformation": transformation,
                    "method": method_key,
                    "id_left": id_a,
                    "left_name": name_a,
                    "true_id_right": true_id_b or "",
                    "pred_id_right": matched_id or "",
                    "pred_right_name": matched_name or "",
                    "similarity_score": confidence,
                    "is_correct": int(is_correct),
                })

            acc = correct / total if total else 0
            print(f"\n  => {correct}/{total} correct ({acc:.0%})")

            out_file = OUT_DIR / f"{method_key}_{transformation}.parquet"
            pd.DataFrame(rows).to_parquet(out_file, index=False)
            print(f"  Saved: {out_file}")

            summary_rows.append({
                "transformation": transformation,
                "method": method_key,
                "label": label,
                "n": total,
                "correct": correct,
                "accuracy": acc,
            })

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))
    df_sum.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nFull summary saved to {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of cases to test")
    args = parser.parse_args()
    run_mini(args.n)
