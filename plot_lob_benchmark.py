#!/usr/bin/env python3
"""
Plot LOB benchmark results using lob_bench plotting functions
"""
import sys
import os
import pickle
import gzip
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add lob_bench to path
sys.path.insert(0, '/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench')
import plotting
import scoring
import seaborn as sns
import pandas as pd

# Results directory
RESULTS_DIR = Path('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/brisk-violet-111/results')

# All 21 metrics
ALL_METRICS = [
    'vol_per_min', 'ofi_down', 'ofi_up', 'ask_cancellation_levels',
    'limit_ask_order_levels', 'limit_bid_order_levels', 'ask_cancellation_depth',
    'limit_ask_order_depth', 'bid_cancellation_levels', 'bid_cancellation_depth',
    'limit_bid_order_depth', 'spread', 'log_time_to_cancel', 'ofi_stay',
    'ofi', 'ask_volume', 'bid_volume', 'orderbook_imbalance',
    'ask_volume_touch', 'bid_volume_touch', 'log_inter_arrival_time'
]

def load_metric_scores(metric_name):
    """Load scores for a given metric from the nested directory structure"""
    metric_dir = RESULTS_DIR / metric_name / 'scores'

    if not metric_dir.exists():
        print(f"Warning: Directory not found for {metric_name}")
        return None, None

    # Find the pkl file (should be only one, or take the latest)
    pkl_files = sorted(list(metric_dir.glob('*.pkl')))
    if not pkl_files:
        print(f"Warning: No pkl file found for {metric_name}")
        return None, None

    # Use the most recent file (in case of log_time_to_cancel with 2 files)
    pkl_file = pkl_files[-1]
    print(f"Loading {metric_name} from {pkl_file.name}")

    try:
        with gzip.open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {metric_name}: {e}")
        return None, None

    # data is a tuple: (scores_dict, dataframe_dict)
    # The key in the dict might not match the metric_name due to directory structure
    # So we just take the first (and only) key
    actual_key = list(data[0].keys())[0]
    scores_dict = data[0][actual_key]  # {'l1': (...), 'wasserstein': (...)}
    score_df = data[1][actual_key]     # DataFrame with columns: score, group, type

    return scores_dict, score_df

def collect_all_data():
    """Collect all scores and dataframes from all metrics"""
    all_scores = {}
    all_dfs = {}

    for metric in ALL_METRICS:
        scores, df = load_metric_scores(metric)
        if scores is not None:
            all_scores[metric] = scores
            all_dfs[metric] = df

    return all_scores, all_dfs

def custom_summary_plot(summary_stats_stocks, save_path):
    """Custom summary plot with formatted x-axis labels"""
    import matplotlib.ticker as ticker

    n_stocks = len(summary_stats_stocks)
    stat_names = ['l1', 'wasserstein']
    n_stats = len(stat_names)

    fig, axs = plt.subplots(n_stats, 1, figsize=(8, 3.5))
    if n_stats == 1:
        axs = [axs]

    for i_stat, loss_metric in enumerate(stat_names):
        ax = axs[i_stat]

        for i_stock, (stock, summary_stats_models) in enumerate(summary_stats_stocks.items()):
            for i_model, (model, summary_stats) in enumerate(summary_stats_models.items()):
                if loss_metric in summary_stats:
                    scatter_vals = summary_stats[loss_metric]
                    scatter_x = np.array([val[0] for val in scatter_vals])
                    cis = np.array([val[1] for val in scatter_vals])

                    y_pos = np.arange(3)  # IQM, median, mean

                    # Plot points
                    ax.scatter(
                        scatter_x,
                        y_pos,
                        marker='x',
                        s=100,
                        color=f"C{i_model}",
                        label=model,
                        linewidths=2,
                    )

                    # Plot error bars
                    ax.errorbar(
                        x=cis.mean(axis=1),
                        y=y_pos,
                        xerr=np.diff(cis, axis=1).T[0] / 2,
                        fmt='none',
                        color=f'C{i_model}',
                        linewidth=2,
                        capsize=4,
                    )

        # Set y-axis
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['IQM', 'median', 'mean'], fontsize=12)
        ax.set_ylim(-0.5, 2.5)

        # Add stock label on the left
        ax.text(
            -0.15, 0.5, 'GOOG', fontsize=14, fontweight='bold',
            ha='right', va='center',
            transform=ax.transAxes
        )

        # Set title
        if loss_metric == 'l1':
            ax.set_title('L1', fontsize=16, fontweight='bold', loc='center', pad=10)
            # Format x-axis to 3 decimal places
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        else:
            ax.set_title('Wasserstein', fontsize=16, fontweight='bold', loc='center', pad=10)
            # Format x-axis to 2 decimal places
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(all_scores, metric_type, save_path):
    """Create bar chart for a given metric type"""
    # Extract data
    metrics = list(all_scores.keys())
    means = [all_scores[m][metric_type][0] for m in metrics]
    cis = np.array([all_scores[m][metric_type][1] for m in metrics])

    # Sort by mean
    sorted_idx = np.argsort(means)
    metrics_sorted = [metrics[i] for i in sorted_idx]
    means_sorted = [means[i] for i in sorted_idx]
    cis_sorted = cis[sorted_idx]

    # Compute error bars
    err_lower = np.array(means_sorted) - cis_sorted[:, 0]
    err_upper = cis_sorted[:, 1] - np.array(means_sorted)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(metrics_sorted))

    ax.barh(y_pos, means_sorted, xerr=[err_lower, err_upper],
            capsize=3, alpha=0.8, color='steelblue', edgecolor='navy')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics_sorted], fontsize=10)
    ax.set_xlabel(f'{metric_type.upper()} Distance', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_type.upper()} Distance for All LOB Metrics\n95% Confidence Intervals',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (val, label) in enumerate(zip(means_sorted, metrics_sorted)):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_and_comparison(all_scores, output_dir):
    """Create summary plots using lob_bench plotting functions"""

    # Organize data in format expected by plotting functions
    # Stock -> Model -> Metric -> Scores
    scores_by_stock = {
        'GOOG': {
            'lobs5': all_scores
        }
    }

    # Create summary stats
    print("\n[*] Computing summary statistics...")
    summary_stats = {
        'GOOG': {
            'lobs5': scoring.summary_stats(all_scores, bootstrap=True)
        }
    }

    print("\n[*] Creating summary plot (multi-model comparison format)...")
    # Use custom summary_plot function with formatted x-axis
    custom_summary_plot(
        summary_stats,
        save_path=str(output_dir / 'summary_stats.png')
    )
    print(f"✅ Saved: {output_dir / 'summary_stats.png'}")

    # Create bar plots for L1 and Wasserstein
    print("\n[*] Creating bar plots...")
    for metric_type in ['l1', 'wasserstein']:
        plot_bar_chart(
            scores_by_stock['GOOG']['lobs5'],
            metric_type,
            save_path=str(output_dir / f'bar_{metric_type}.png')
        )
        print(f"✅ Saved: {output_dir / f'bar_{metric_type}.png'}")

    # Create spider plots (skip if Chrome not available)
    print("\n[*] Creating spider plots...")
    try:
        for metric_type in ['l1', 'wasserstein']:
            fig = plotting.spider_plot(
                scores_by_stock['GOOG'],
                metric_type,
                title=f"{metric_type.upper()} Distance - GOOG (s5 model)",
                plot_cis=True,
                save_path=str(output_dir / f'spider_{metric_type}.png')
            )
            print(f"✅ Saved: {output_dir / f'spider_{metric_type}.png'}")
    except Exception as e:
        print(f"⚠️  Skipping spider plots (Chrome not available): {e}")

def plot_histograms(all_dfs, output_dir):
    """Plot histograms for all metrics"""
    print("\n[*] Creating histogram plots...")

    # Create plot functions for each metric
    plot_fns = {
        metric_name: plotting.get_plot_fn_uncond(score_df)
        for metric_name, score_df in all_dfs.items()
    }

    # Plot all histograms in a grid
    plotting.hist_subplots(
        plot_fns,
        figsize=(14, 28),
        suptile='LOB Metrics Histograms - GOOG (s5 model)',
        save_path=str(output_dir / 'histograms_all.png'),
        plot_legend=True,
    )
    print(f"✅ Saved: {output_dir / 'histograms_all.png'}")

def print_summary_table(all_scores):
    """Print a summary table of all metrics"""
    print("\n" + "="*100)
    print("📊 LOB BENCHMARK RESULTS SUMMARY")
    print("="*100)
    print(f"{'Metric':<30} {'L1 Distance':<25} {'Wasserstein Distance':<25}")
    print(f"{'':30} {'Mean [95% CI]':<25} {'Mean [95% CI]':<25}")
    print("-"*100)

    # Sort by L1 distance
    sorted_metrics = sorted(
        all_scores.items(),
        key=lambda x: x[1]['l1'][0]
    )

    for metric, scores in sorted_metrics:
        l1_mean, l1_ci, _ = scores['l1']
        wd_mean, wd_ci, _ = scores['wasserstein']

        l1_str = f"{l1_mean:.4f} [{l1_ci[0]:.4f}, {l1_ci[1]:.4f}]"
        wd_str = f"{wd_mean:.5f} [{wd_ci[0]:.5f}, {wd_ci[1]:.5f}]"

        print(f"{metric:<30} {l1_str:<25} {wd_str:<25}")

    print("="*100)

    # Print averages
    avg_l1 = np.mean([scores['l1'][0] for scores in all_scores.values()])
    avg_wd = np.mean([scores['wasserstein'][0] for scores in all_scores.values()])
    print(f"\n{'Average across all metrics:':<30} L1={avg_l1:.4f}  WD={avg_wd:.5f}")
    print()

def load_bf16_scores():
    """Load scores for bf16 model (dandy-aardvark-138) if available"""
    bf16_results_dir = Path('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/dandy-aardvark-138/results')

    if not bf16_results_dir.exists():
        return None

    all_scores_bf16 = {}
    for metric in ALL_METRICS:
        metric_dir = bf16_results_dir / metric / 'scores'
        if not metric_dir.exists():
            continue
        pkl_files = sorted(list(metric_dir.glob('*.pkl')))
        if not pkl_files:
            continue
        try:
            with gzip.open(pkl_files[-1], 'rb') as f:
                data = pickle.load(f)
            actual_key = list(data[0].keys())[0]
            all_scores_bf16[metric] = data[0][actual_key]
        except Exception as e:
            print(f"Warning: Could not load bf16 {metric}: {e}")
            continue

    if len(all_scores_bf16) < 10:  # Need at least 10 metrics for meaningful comparison
        return None
    return all_scores_bf16

def create_comparison_plot(all_scores, output_dir):
    """Create comparison plot with s5-2512, s5-2409, and optionally s5-2512-bf16 models"""

    # Current model (s5-2512)
    summary_stats_2512 = scoring.summary_stats(all_scores, bootstrap=True)

    # Manual data for s5-2409 (GenS5/LOBS5)
    # Format: {metric: [(mean, ci), (median, ci), (IQM, ci)]}
    summary_stats_2409 = {
        'l1': [
            (0.139, np.array([0.139, 0.140])),  # mean
            (0.128, np.array([0.126, 0.130])),  # median
            (0.137, np.array([0.136, 0.138])),  # IQM
        ],
        'wasserstein': [
            (0.115, np.array([0.114, 0.116])),  # mean
            (0.077, np.array([0.076, 0.079])),  # median
            (0.082, np.array([0.081, 0.084])),  # IQM
        ]
    }

    # Try to load bf16 model results
    all_scores_bf16 = load_bf16_scores()

    # Organize for plotting
    summary_stats_comp = {
        'GOOG': {
            's5-2512': summary_stats_2512,
            's5-2409': summary_stats_2409
        }
    }

    # Add bf16 if available
    if all_scores_bf16 is not None:
        summary_stats_bf16 = scoring.summary_stats(all_scores_bf16, bootstrap=True)
        summary_stats_comp['GOOG']['s5-2512-bf16'] = summary_stats_bf16
        print("\n[*] Creating 3-model comparison plot (s5-2512 vs s5-2409 vs s5-2512-bf16)...")
    else:
        print("\n[*] Creating 2-model comparison plot (s5-2512 vs s5-2409)...")
        print("    (bf16 model results not yet available)")

    custom_summary_plot(
        summary_stats_comp,
        save_path=str(output_dir / 'summary_stats_comp.png')
    )
    print(f"✅ Saved: {output_dir / 'summary_stats_comp.png'}")

def main():
    print("🔍 Loading LOB benchmark results...")
    all_scores, all_dfs = collect_all_data()

    print(f"\n✅ Successfully loaded {len(all_scores)}/{len(ALL_METRICS)} metrics")

    if not all_scores:
        print("❌ No results found!")
        return

    # Print summary table
    print_summary_table(all_scores)

    # Create output directory
    output_dir = Path('/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/lob_bench_plots')
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\n🎨 Creating visualizations...")
    plot_summary_and_comparison(all_scores, output_dir)

    # Create comparison plot with both models
    create_comparison_plot(all_scores, output_dir)

    plot_histograms(all_dfs, output_dir)

    print("\n" + "="*100)
    print("🎉 ALL DONE!")
    print(f"📁 Plots saved to: {output_dir}")
    print("="*100)

if __name__ == '__main__':
    main()
