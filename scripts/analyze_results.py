"""
RESULTS ANALYSIS SCRIPT - FINAL VERSION
========================================
✅ Uses predicted_match column for correct metrics
✅ Handles one-to-many ground truth properly
✅ Matches notebook evaluation logic

Usage:
    python scripts/analyze_results.py
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results(results_dir='results'):
    """Load all parquet files into single dataframe"""
    results = []
    results_path = Path(results_dir)
    
    for parquet_file in results_path.rglob('*.parquet'):
        df = pd.read_parquet(parquet_file)
        results.append(df)
    
    if not results:
        print(f"❌ No parquet files found in {results_dir}")
        return None
    
    return pd.concat(results, ignore_index=True)

def compute_metrics(df):
    """
    ✅ CORRECT metrics calculation using predicted_match column
    
    predicted_match = 1: We made a prediction (similarity >= threshold)
    predicted_match = 0: We didn't predict (similarity < threshold)
    is_correct = 1: Our prediction was correct
    is_correct = 0: Our prediction was wrong OR we didn't predict
    """
    metrics = []
    
    for (method, transformation, dataset), group in df.groupby(['method', 'transformation', 'dataset']):
        # Records with ground truth
        has_gt = (group['true_id_right'] != '')
        
        # ✅ Use predicted_match column (respects threshold)
        tp = ((group['predicted_match'] == 1) & (group['is_correct'] == 1)).sum()
        fp = ((group['predicted_match'] == 1) & (group['is_correct'] == 0)).sum()
        fn = (has_gt & (group['is_correct'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.append({
            'method': method,
            'transformation': transformation,
            'dataset': dataset,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_records': len(group),
            'true_matches': has_gt.sum(),
            'predicted_matches': (group['predicted_match'] == 1).sum()
        })
    
    return pd.DataFrame(metrics)

def print_summary(metrics_df):
    """Print summary table"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Pivot table: methods as rows, transformations as columns, F1 as values
    for dataset in metrics_df['dataset'].unique():
        print(f"\nDataset: {dataset}")
        print("-"*80)
        
        df_dataset = metrics_df[metrics_df['dataset'] == dataset]
        
        pivot = df_dataset.pivot_table(
            index='method',
            columns='transformation',
            values='f1',
            aggfunc='first'
        )
        
        print(pivot.round(3))
        print()
        
        # Show TP/FP/FN for one transformation
        if 'original' in df_dataset['transformation'].values:
            print("\nDetailed metrics (original transformation):")
            orig = df_dataset[df_dataset['transformation'] == 'original']
            print(orig[['method', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']].to_string(index=False))
            print()

def plot_results(metrics_df, output_dir='results/plots'):
    """Create visualization plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    for dataset in metrics_df['dataset'].unique():
        df_dataset = metrics_df[metrics_df['dataset'] == dataset]
        
        # Plot 1: F1 scores by method and transformation
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot = df_dataset.pivot_table(
            index='method',
            columns='transformation',
            values='f1',
            aggfunc='first'
        )
        
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(f'F1 Scores: {dataset}')
        ax.set_xlabel('Method')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        ax.legend(title='Transformation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_path / f'{dataset}_f1_scores.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / f'{dataset}_f1_scores.png'}")
        plt.close()
        
        # Plot 2: Precision vs Recall
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for transformation in df_dataset['transformation'].unique():
            df_trans = df_dataset[df_dataset['transformation'] == transformation]
            ax.scatter(df_trans['recall'], df_trans['precision'], 
                      label=transformation, s=100, alpha=0.6)
            
            # Add method labels
            for _, row in df_trans.iterrows():
                ax.annotate(row['method'], 
                          (row['recall'], row['precision']),
                          fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision vs Recall: {dataset}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path / f'{dataset}_precision_recall.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / f'{dataset}_precision_recall.png'}")
        plt.close()

def export_summary(metrics_df, output_file='results/summary.csv'):
    """Export summary to CSV"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_path, index=False)
    print(f"✅ Exported summary to: {output_path}")

def main():
    print("="*80)
    print("ANALYZING EXPERIMENT RESULTS")
    print("="*80)
    
    # Load all results
    print("\n1. Loading parquet files...")
    df_all = load_all_results()
    
    if df_all is None:
        return
    
    print(f"   Loaded {len(df_all)} total records")
    print(f"   Methods: {df_all['method'].unique().tolist()}")
    print(f"   Transformations: {df_all['transformation'].unique().tolist()}")
    print(f"   Datasets: {df_all['dataset'].unique().tolist()}")
    
    # Check for predicted_match column
    if 'predicted_match' not in df_all.columns:
        print("\n⚠️  WARNING: predicted_match column not found!")
        print("   Your parquet files may be from an old version.")
        print("   Please re-run experiments with the updated methods.py")
        return
    
    # Compute metrics
    print("\n2. Computing metrics...")
    metrics_df = compute_metrics(df_all)
    
    # Print summary
    print_summary(metrics_df)
    
    # Create plots
    print("\n3. Creating visualizations...")
    plot_results(metrics_df)
    
    # Export
    print("\n4. Exporting summary...")
    export_summary(metrics_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutputs:")
    print("  - results/summary.csv")
    print("  - results/plots/*.png")
    
    print("\n" + "="*80)
    print("VALIDATION CHECK")
    print("="*80)
    print("\nExpected F1 scores (from notebook):")
    print("  Jaro-Winkler: ~0.08")
    print("  TF-IDF: ~0.52")
    print("  SentenceTransformer: ~0.62")
    print("\nIf your F1 scores match these, the fix worked! ✅")

if __name__ == "__main__":
    main()
