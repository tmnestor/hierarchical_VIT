#!/usr/bin/env python
"""
Analyze and visualize Lottery Ticket Hypothesis pruning results.

This script analyzes the results of Lottery Ticket Hypothesis pruning 
and visualizes the relationship between pruning percentage, model size, 
and performance metrics.

It also helps identify the "winning tickets" - networks with the best 
performance at different levels of sparsity.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict


def load_pruning_results(results_dir):
    """
    Load pruning results from JSON files.
    
    Args:
        results_dir: Directory containing pruning results
        
    Returns:
        Dictionary of pruning results
    """
    results_path = Path(results_dir) / "pruning_results.json"
    
    if results_path.exists():
        # Load main results file
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        # Try to reconstruct from individual level results
        results = {}
        for level_dir in Path(results_dir).glob("level*"):
            level_name = level_dir.name
            level_results = []
            
            for pruned_dir in level_dir.glob("pruned_*pct"):
                result_path = pruned_dir / "results.json"
                if result_path.exists():
                    with open(result_path, 'r') as f:
                        level_results.append(json.load(f))
            
            if level_results:
                results[level_name] = level_results
                
        # Also check for multiclass
        multiclass_dir = Path(results_dir) / "multiclass"
        if multiclass_dir.exists():
            multiclass_results = []
            for pruned_dir in multiclass_dir.glob("pruned_*pct"):
                result_path = pruned_dir / "results.json"
                if result_path.exists():
                    with open(result_path, 'r') as f:
                        multiclass_results.append(json.load(f))
            
            if multiclass_results:
                results["multiclass"] = multiclass_results
        
        if not results:
            raise FileNotFoundError(f"No pruning results found in {results_dir}")
            
        return results


def extract_metrics(results):
    """
    Extract metrics from pruning results.
    
    Args:
        results: Dictionary of pruning results
        
    Returns:
        DataFrame with metrics for each pruning level
    """
    # Create a list to store rows for the DataFrame
    rows = []
    
    for level, level_results in results.items():
        for result in level_results:
            # Basic info
            row = {
                'Level': level,
                'Pruning_Percentage': result['pruning_percentage'],
                'Model_Path': result.get('model_path', ''),
                'Trained_Model_Path': result.get('trained_model_path', '')
            }
            
            # Extract metrics
            metrics = result.get('metrics', {})
            if not metrics:
                rows.append(row)
                continue
            
            # Handle hierarchical metrics differently
            if 'hierarchical' in metrics:
                hier_metrics = metrics['hierarchical']['overall']
                row.update({
                    'Accuracy': hier_metrics.get('accuracy', np.nan),
                    'Balanced_Accuracy': hier_metrics.get('balanced_accuracy', np.nan),
                    'F1_Macro': hier_metrics.get('f1_macro', np.nan)
                })
                
                # Add level-specific metrics if available
                for level_key, level_data in metrics['hierarchical'].items():
                    if level_key != 'overall' and isinstance(level_data, dict):
                        for metric_name, metric_value in level_data.items():
                            row[f"{level_key}_{metric_name}"] = metric_value
            else:
                # Regular metrics
                row.update({
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'Balanced_Accuracy': metrics.get('balanced_accuracy', np.nan),
                    'F1_Macro': metrics.get('f1_macro', np.nan)
                })
            
            rows.append(row)
    
    # Create DataFrame
    return pd.DataFrame(rows)


def find_winning_tickets(df, metric='F1_Macro'):
    """
    Find the "winning tickets" - best performing models at each pruning level.
    
    Args:
        df: DataFrame with metrics
        metric: Metric to optimize (default: F1_Macro)
        
    Returns:
        DataFrame with winning tickets
    """
    # Group by level and pruning percentage, find best model
    winning_tickets = df.loc[df.groupby(['Level', 'Pruning_Percentage'])[metric].idxmax()]
    
    # Determine if it's a "winning ticket" by comparing to unpruned performance
    # We get baseline from the lowest pruning percentage in each level
    baselines = {}
    
    for level in winning_tickets['Level'].unique():
        level_df = winning_tickets[winning_tickets['Level'] == level]
        if len(level_df) > 0:
            baseline = level_df.loc[level_df['Pruning_Percentage'].idxmin(), metric]
            baselines[level] = baseline
    
    # Mark as winning ticket if performance >= baseline
    winning_tickets['Is_Winning_Ticket'] = winning_tickets.apply(
        lambda row: row[metric] >= baselines.get(row['Level'], 0),
        axis=1
    )
    
    # Calculate improvement over baseline
    winning_tickets['Improvement'] = winning_tickets.apply(
        lambda row: (row[metric] - baselines.get(row['Level'], 0)) / baselines.get(row['Level'], 1) * 100,
        axis=1
    )
    
    # Sort by level and pruning percentage
    return winning_tickets.sort_values(['Level', 'Pruning_Percentage'])


def plot_metrics_vs_pruning(df, output_dir, metrics=['Accuracy', 'Balanced_Accuracy', 'F1_Macro']):
    """
    Plot metrics vs. pruning percentage for each level.
    
    Args:
        df: DataFrame with metrics
        output_dir: Directory to save plots
        metrics: List of metrics to plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot for each level
    for level in df['Level'].unique():
        level_df = df[df['Level'] == level].sort_values('Pruning_Percentage')
        
        if len(level_df) < 2:
            print(f"Not enough data points for {level}, skipping plot")
            continue
        
        # Plot all metrics on one figure
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            if metric not in level_df.columns:
                continue
                
            plt.subplot(len(metrics), 1, i+1)
            sns.lineplot(data=level_df, x='Pruning_Percentage', y=metric, marker='o')
            plt.title(f'{metric} vs. Pruning Percentage - {level.upper()}')
            plt.xlabel('Pruning Percentage')
            plt.ylabel(metric)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / f"{level}_metrics_vs_pruning.png")
        plt.close()
        
    # Also create a combined plot with all levels for each metric
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Pruning_Percentage', y=metric, hue='Level', marker='o')
        plt.title(f'{metric} vs. Pruning Percentage - All Levels')
        plt.xlabel('Pruning Percentage')
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(output_path / f"all_levels_{metric}_vs_pruning.png")
        plt.close()


def create_report(df, winning_tickets, output_dir):
    """
    Create an HTML report summarizing pruning results.
    
    Args:
        df: DataFrame with all pruning results
        winning_tickets: DataFrame with winning tickets
        output_dir: Directory to save report
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create HTML report
    with open(output_path / "pruning_report.html", 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lottery Ticket Pruning Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { text-align: left; padding: 8px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #4CAF50; color: white; }
                .winning { background-color: #dff0d8; }
                .header { background-color: #5bc0de; color: white; padding: 10px; margin-bottom: 10px; }
                .container { margin-bottom: 40px; }
                .chart-container { display: flex; flex-wrap: wrap; justify-content: center; }
                .chart { margin: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Lottery Ticket Hypothesis Pruning Report</h1>
                <p>Analysis of pruning results and identification of winning tickets</p>
            </div>
        """)
        
        # Add summary section
        f.write("""
            <div class="container">
                <h2>Pruning Summary</h2>
                <table>
                    <tr>
                        <th>Level</th>
                        <th>Pruning Percentage</th>
                        <th>Accuracy</th>
                        <th>Balanced Accuracy</th>
                        <th>F1 Macro</th>
                    </tr>
        """)
        
        # Add rows for each pruning level
        for level in df['Level'].unique():
            level_df = df[df['Level'] == level].sort_values('Pruning_Percentage')
            
            for _, row in level_df.iterrows():
                f.write(f"""
                    <tr>
                        <td>{row['Level']}</td>
                        <td>{row['Pruning_Percentage']:.1f}%</td>
                        <td>{row.get('Accuracy', 'N/A'):.4f}</td>
                        <td>{row.get('Balanced_Accuracy', 'N/A'):.4f}</td>
                        <td>{row.get('F1_Macro', 'N/A'):.4f}</td>
                    </tr>
                """)
        
        f.write("</table></div>")
        
        # Add winning tickets section
        f.write("""
            <div class="container">
                <h2>Winning Tickets</h2>
                <p>Models that maintain or improve performance after pruning</p>
                <table>
                    <tr>
                        <th>Level</th>
                        <th>Pruning Percentage</th>
                        <th>F1 Macro</th>
                        <th>Improvement</th>
                        <th>Model Path</th>
                    </tr>
        """)
        
        # Add rows for winning tickets
        for _, row in winning_tickets.iterrows():
            # Highlight winning tickets
            class_name = "winning" if row['Is_Winning_Ticket'] else ""
            
            f.write(f"""
                <tr class="{class_name}">
                    <td>{row['Level']}</td>
                    <td>{row['Pruning_Percentage']:.1f}%</td>
                    <td>{row.get('F1_Macro', 'N/A'):.4f}</td>
                    <td>{row['Improvement']:.2f}%</td>
                    <td>{os.path.basename(row['Trained_Model_Path'])}</td>
                </tr>
            """)
        
        f.write("</table></div>")
        
        # Add charts section
        f.write("""
            <div class="container">
                <h2>Performance Charts</h2>
                <div class="chart-container">
        """)
        
        # Add images for all charts
        for level in df['Level'].unique():
            chart_path = f"{level}_metrics_vs_pruning.png"
            if (output_path / chart_path).exists():
                f.write(f"""
                    <div class="chart">
                        <h3>{level.upper()} Model</h3>
                        <img src="{chart_path}" alt="{level} metrics" width="600">
                    </div>
                """)
        
        # Add combined charts
        for metric in ['Accuracy', 'Balanced_Accuracy', 'F1_Macro']:
            chart_path = f"all_levels_{metric}_vs_pruning.png"
            if (output_path / chart_path).exists():
                f.write(f"""
                    <div class="chart">
                        <h3>All Levels - {metric}</h3>
                        <img src="{chart_path}" alt="All levels {metric}" width="600">
                    </div>
                """)
        
        f.write("</div></div>")
        
        # Close HTML
        f.write("</body></html>")
    
    # Also save as CSV
    df.to_csv(output_path / "pruning_results.csv", index=False)
    winning_tickets.to_csv(output_path / "winning_tickets.csv", index=False)
    
    print(f"Pruning report generated at {output_path / 'pruning_report.html'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Lottery Ticket Hypothesis pruning results"
    )
    
    parser.add_argument(
        "--results_dir", "-r", required=True,
        help="Directory containing pruning results"
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Directory to save analysis results (default: same as results_dir)"
    )
    parser.add_argument(
        "--metric", "-m", default="F1_Macro",
        choices=["Accuracy", "Balanced_Accuracy", "F1_Macro"],
        help="Metric to optimize for finding winning tickets (default: F1_Macro)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    # Load results
    try:
        results = load_pruning_results(args.results_dir)
    except Exception as e:
        print(f"Error loading pruning results: {e}")
        return
    
    # Extract metrics
    df = extract_metrics(results)
    
    # Find winning tickets
    winning_tickets = find_winning_tickets(df, metric=args.metric)
    
    # Plot metrics
    plot_metrics_vs_pruning(df, args.output_dir)
    
    # Create report
    create_report(df, winning_tickets, args.output_dir)
    
    # Print summary to console
    print("\nWinning Tickets Summary:")
    for level in winning_tickets['Level'].unique():
        level_tickets = winning_tickets[(winning_tickets['Level'] == level) & 
                                        (winning_tickets['Is_Winning_Ticket'])]
        
        if len(level_tickets) > 0:
            best_ticket = level_tickets.loc[level_tickets[args.metric].idxmax()]
            print(f"\n{level.upper()} Model:")
            print(f"  Best pruned model: {best_ticket['Pruning_Percentage']:.1f}% pruned")
            print(f"  {args.metric}: {best_ticket[args.metric]:.4f} "
                  f"({best_ticket['Improvement']:.2f}% improvement)")
            print(f"  Model path: {os.path.basename(best_ticket['Trained_Model_Path'])}")
        else:
            print(f"\n{level.upper()} Model: No winning tickets found")


if __name__ == "__main__":
    main()