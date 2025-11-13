"""
AUTOMATED ENTITY MATCHING EXPERIMENT RUNNER - CORRECTED
=======================================================
Runs methods with threshold optimization and proper output schema.

Usage:
    python scripts/run_experiments.py --dataset abt-buy
    python scripts/run_experiments.py --dataset abt-buy --methods jaro_winkler,tfidf
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from methods import METHODS

# Dataset configurations
DATASETS = {
    'abt-buy': {
        'dir': 'data/abt-buy',
        'left_name': 'abt',
        'right_name': 'buy',
        'mapping_file': 'data_original/abt_buy_perfect_mapping.csv'
    },
    'amazon-google': {
        'dir': 'data/amazon-google',
        'left_name': 'amazon',
        'right_name': 'googleproducts',
        'mapping_file': 'data_original/Amazon_GoogleProducts_perfectMapping.csv'
    },
    'dblp-acm': {
        'dir': 'data/dblp-acm',
        'left_name': 'dblp2',
        'right_name': 'acm',
        'mapping_file': 'data_original/DBLP-ACM_perfectMapping.csv'
    },
    'dblp-scholar': {
        'dir': 'data/dblp-scholar',
        'left_name': 'dblp1',
        'right_name': 'scholar',
        'mapping_file': 'data_original/DBLP-Scholar_perfectMapping.csv'
    }
}

TRANSFORMATIONS = ['original', 'ciphered_letters', 'ciphered_words', 'scrambled']

def load_dataset(dataset_name, transformation):
    """Load dataset files"""
    config = DATASETS[dataset_name]
    base_dir = Path(config['dir'])
    
    if transformation == 'original':
        left_file = base_dir / f"data_original/{config['left_name']}.csv"
        right_file = base_dir / f"data_original/{config['right_name']}.csv"
    else:
        left_file = base_dir / f"data_test/{config['left_name']}_{transformation}.csv"
        right_file = base_dir / f"data_test/{config['right_name']}_{transformation}.csv"
    
    # Load with encoding detection
    for enc in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df_left = pd.read_csv(left_file, encoding=enc)
            df_right = pd.read_csv(right_file, encoding=enc)
            break
        except:
            continue
    
    # Load ground truth
    mapping_file = base_dir / config['mapping_file']
    df_mapping = pd.read_csv(mapping_file)
    
    return df_left, df_right, df_mapping

def run_single_experiment(dataset_name, transformation, method_name, method_func):
    """Run single method on single transformation"""
    
    print(f"\n{'='*80}")
    print(f"Running: {method_name} on {dataset_name} ({transformation})")
    print(f"{'='*80}")
    
    try:
        # Load data
        print("Loading data...")
        df_left, df_right, df_mapping = load_dataset(dataset_name, transformation)
        print(f"  Left: {len(df_left)} records")
        print(f"  Right: {len(df_right)} records")
        print(f"  Ground truth: {len(df_mapping)} matches")
        
        # Run method (includes threshold optimization)
        print(f"\nRunning {method_name}...")
        results_df = method_func(df_left, df_right, df_mapping)
        
        # Add metadata
        results_df['method'] = method_name
        results_df['transformation'] = transformation
        results_df['dataset'] = dataset_name
        results_df['timestamp'] = datetime.now().isoformat()
        
        # Save to parquet
        output_dir = Path(f"results/{dataset_name}/{method_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{transformation}.parquet"
        
        results_df.to_parquet(output_file, index=False)
        print(f"\n✅ Saved: {output_file}")
        
        # Print summary
        correct = results_df['is_correct'].sum()
        total = len(results_df)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"  Total predictions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Run entity matching experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()) + ['all'],
                       help='Dataset to run')
    parser.add_argument('--methods', type=str, default='all',
                       help='Comma-separated methods or "all"')
    parser.add_argument('--transformations', type=str, default='all',
                       help='Comma-separated transformations or "all"')
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.dataset == 'all':
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = [args.dataset]
    
    if args.methods == 'all':
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    
    if args.transformations == 'all':
        transformations_to_run = TRANSFORMATIONS
    else:
        transformations_to_run = [t.strip() for t in args.transformations.split(',')]
    
    print("="*80)
    print("ENTITY MATCHING EXPERIMENT RUNNER")
    print("="*80)
    print(f"\nDatasets: {datasets_to_run}")
    print(f"Methods: {methods_to_run}")
    print(f"Transformations: {transformations_to_run}")
    print(f"\nTotal experiments: {len(datasets_to_run) * len(methods_to_run) * len(transformations_to_run)}")
    
    # Run experiments
    results_summary = []
    
    for dataset in datasets_to_run:
        for method_name in methods_to_run:
            if method_name not in METHODS:
                print(f"⚠️  Unknown method: {method_name}, skipping")
                continue
            
            method_func = METHODS[method_name]
            
            for transformation in transformations_to_run:
                success = run_single_experiment(dataset, transformation, method_name, method_func)
                results_summary.append({
                    'dataset': dataset,
                    'method': method_name,
                    'transformation': transformation,
                    'success': success
                })
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    df_summary = pd.DataFrame(results_summary)
    print(f"\nTotal experiments: {len(df_summary)}")
    print(f"Successful: {df_summary['success'].sum()}")
    print(f"Failed: {(~df_summary['success']).sum()}")
    
    if (~df_summary['success']).any():
        print("\nFailed experiments:")
        failed = df_summary[~df_summary['success']]
        for _, row in failed.iterrows():
            print(f"  - {row['dataset']} / {row['method']} / {row['transformation']}")

if __name__ == "__main__":
    main()
