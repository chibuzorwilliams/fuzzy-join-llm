"""
UNIFIED RESULTS ANALYSIS SCRIPT
================================
✅ Loads Abt-Buy (unfiltered, all records)
✅ Loads Amazon-Google (filtered to 1113, title-only)
✅ Generates ONE summary.csv with both datasets
✅ Creates plots for all experiments

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
    
    print("\nLoading results:")
    print("-" * 80)
    
    for parquet_file in results_path.rglob('*.parquet'):
        df = pd.read_parquet(parquet_file)
        
        # Log what we're loading
        dataset = df['dataset'].iloc[0] if 'dataset' in df.columns else 'unknown'
        method = df['method'].iloc[0] if 'method' in df.columns else 'unknown'
        trans = df['transformation'].iloc[0] if 'transformation' in df.columns else 'unknown'
        
        # Validate record counts
        if dataset == 'amazon-google' and len(df) == 1113:
            print(f"  ✅ {dataset:15s} {method:20s} {trans:20s} {len(df):5d} records (filtered, title-only)")
        elif dataset == 'abt-buy':
            print(f"  ✅ {dataset:15s} {method:20s} {trans:20s} {len(df):5d} records (unfiltered)")
        else:
            print(f"  ⚠️  {dataset:15s} {method:20s} {trans:20s} {len(df):5d} records (unexpected count!)")
        
        results.append(df)
    
    if not results:
        print(f"❌ No parquet files found in {results_dir}")
        return None
    
    return pd.concat(results, ignore_index=True)

def compute_metrics(df):
    """
    ✅ CORRECT metrics calculation using predicted_match column
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
    
    for dataset in sorted(metrics_df['dataset'].unique()):
        print(f"\nDataset: {dataset}")
        print("-"*80)
        
        df_dataset = metrics_df[metrics_df['dataset'] == dataset]
        
        # Show record count info
        sample_count = df_dataset['total_records'].iloc[0]
        if dataset == 'amazon-google':
            if sample_count == 1113:
                print("✅ Filtered to 1113 records (title-only)")
            else:
                print(f"⚠️  Unexpected count: {sample_count} records")
        elif dataset == 'abt-buy':
            print(f"✅ Unfiltered: {sample_count} records")
        
        # F1 scores pivot table
        pivot = df_dataset.pivot_table(
            index='method',
            columns='transformation',
            values='f1',
            aggfunc='first'
        )
        
        print("\nF1 Scores:")
        print(pivot.round(3).to_string())
        
        # Detailed metrics for original transformation
        if 'original' in df_dataset['transformation'].values:
            print("\nDetailed Metrics (Original Transformation):")
            orig = df_dataset[df_dataset['transformation'] == 'original'].copy()
            orig = orig.sort_values('f1', ascending=False)
            print(orig[['method', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn']].to_string(index=False))
        
        print()

def plot_results(metrics_df, output_dir='results/plots'):
    """Create visualization plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    for dataset in sorted(metrics_df['dataset'].unique()):
        df_dataset = metrics_df[metrics_df['dataset'] == dataset]
        
        # Determine if this is filtered or not for labeling
        sample_count = df_dataset['total_records'].iloc[0]
        if dataset == 'amazon-google' and sample_count == 1113:
            label_suffix = " (Title-Only)"
        else:
            label_suffix = ""
        
        # Plot 1: F1 scores by method and transformation
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot = df_dataset.pivot_table(
            index='method',
            columns='transformation',
            values='f1',
            aggfunc='first'
        )
        
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(f'F1 Scores: {dataset}{label_suffix}')
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
            
            for _, row in df_trans.iterrows():
                ax.annotate(row['method'], 
                          (row['recall'], row['precision']),
                          fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision vs Recall: {dataset}{label_suffix}')
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
    
    # Sort for readability
    metrics_df = metrics_df.sort_values(['dataset', 'method', 'transformation'])
    
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
    
    print(f"\n   Total records loaded: {len(df_all)}")
    print(f"   Methods: {sorted(df_all['method'].unique().tolist())}")
    print(f"   Transformations: {sorted(df_all['transformation'].unique().tolist())}")
    print(f"   Datasets: {sorted(df_all['dataset'].unique().tolist())}")
    
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
    print("DATA VALIDATION")
    print("="*80)
    
    # Verify Amazon-Google is filtered
    ag_metrics = metrics_df[metrics_df['dataset'] == 'amazon-google']
    if not ag_metrics.empty:
        ag_count = ag_metrics['total_records'].iloc[0]
        if ag_count == 1113:
            print("✅ Amazon-Google: Correctly filtered (1113 records, title-only)")
        else:
            print(f"⚠️  Amazon-Google: Wrong count ({ag_count} records, expected 1113)")
    
    # Verify Abt-Buy exists
    ab_metrics = metrics_df[metrics_df['dataset'] == 'abt-buy']
    if not ab_metrics.empty:
        ab_count = ab_metrics['total_records'].iloc[0]
        print(f"✅ Abt-Buy: Loaded ({ab_count} records, unfiltered)")

if __name__ == "__main__":
    main()
