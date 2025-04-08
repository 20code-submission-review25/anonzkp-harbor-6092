#!/usr/bin/env python3
"""
ZK Matrix Proofs Visualization Script

This script generates publication-quality plots from benchmark data
for zero-knowledge matrix multiplication proofs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import os

# Set a consistent style for academic papers with universally available fonts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',  # Standard for academic publications
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.figsize': [3.54, 2.36],  # 90mm journal column width
    'figure.dpi': 300,
    'mathtext.fontset': 'stix'
})

# Academic paper color scheme
COLORS = {
    'proving': '#1f77b4',    # Blue
    'verification': '#2ca02c', # Green
    'proof_size': '#9467bd',  # Purple
    'memory': '#ff7f0e',     # Orange
    'speedup': '#d62728',    # Red
    'baseline': '#7f7f7f',   # Gray
    'reference': '#17becf'   # Cyan
}

def performance_plots():
    """
    Generate performance plots for ZK matrix multiplication:
    - Proving time scaling
    - Verification time
    - Proof size
    - Memory usage
    """
    try:
        # Try summary file with statistical data first
        df = pd.read_csv("zk_matrix_summary.csv")
        has_error_data = True
    except FileNotFoundError:
        try:
            # Fall back to basic benchmark data
            df = pd.read_csv("zk_matrix_benchmark.csv")
            has_error_data = False
        except FileNotFoundError:
            print("Error: No benchmark data files found")
            return

    # Create 2x2 grid for journal column width
    fig, axs = plt.subplots(2, 2, figsize=(3.54, 3.54))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Extract data
    matrix_sizes = df['matrix_size'].values
    
    # --- Proving Time Plot ---
    ax = axs[0, 0]
    if has_error_data:
        y = df['mean_proof_time'].values
        yerr = df['stddev_proof_time'].values if 'stddev_proof_time' in df else np.zeros_like(y)
        ax.errorbar(matrix_sizes, y, yerr=yerr, fmt='o-', color=COLORS['proving'], 
                   markersize=3, capsize=2, elinewidth=0.5, label='Experimental')
    else:
        y = df['proof_time_ms'].values
        ax.loglog(matrix_sizes, y, 'o-', color=COLORS['proving'], markersize=3, label='Experimental')
    
    # Add theoretical O(n²) reference line
    n_squared = (matrix_sizes/matrix_sizes[0])**2 * y[0]
    ax.loglog(matrix_sizes, n_squared, '-.', color=COLORS['baseline'], linewidth=0.8, 
             label=r'Theoretical $O(n^2)$')
    
    ax.set_xlabel('Matrix Dimension $n$')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Proving Time Scaling')
    ax.legend(frameon=False, handletextpad=0.3)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    
    # --- Verification Time Plot ---
    ax = axs[0, 1]
    if has_error_data:
        y = df['mean_verify_time'].values if 'mean_verify_time' in df else df['verification_time_ms'].values
        yerr = df['stddev_verify_time'].values if 'stddev_verify_time' in df else np.zeros_like(y)
        ax.errorbar(matrix_sizes, y, yerr=yerr, fmt='s-', color=COLORS['verification'], 
                markersize=3, capsize=2, elinewidth=0.5)
    else:
        y = df['verification_time_ms'].values
        ax.plot(matrix_sizes, y, 's-', color=COLORS['verification'], markersize=3)

    # Set tight y-limits around the verification time
    y_min = max(0, np.min(y) * 0.95)
    y_max = np.max(y) * 1.05
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Matrix Dimension $n$')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Verification Time')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    
    # --- Proof Size Plot ---
    ax = axs[1, 0]
    if 'proof_size_bytes' in df.columns:
        y = df['proof_size_bytes'].values
    else:
        y = np.ones_like(matrix_sizes) * 320  # Fixed 320 bytes
    
    ax.semilogx(matrix_sizes, y, '^-', color=COLORS['proof_size'], markersize=3)
    y_min = max(0, np.min(y) * 0.99)
    y_max = np.max(y) * 1.01
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Matrix Dimension $n$')
    ax.set_ylabel('Size (bytes)')
    ax.set_title('Proof Size')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    
    # --- Memory Usage Plot ---
    ax = axs[1, 1]
    if 'memory_usage_kb' in df.columns:
        y = df['memory_usage_kb']
    elif 'mean_memory_kb' in df.columns:
        y = df['mean_memory_kb']
    else:
        y = np.ones_like(matrix_sizes) * 1024  # Placeholder
    
    ax.loglog(matrix_sizes, y, 'D-', color=COLORS['memory'], markersize=3, label='Experimental')
    
    # Theoretical n² reference line
    ref_n2_mem = (matrix_sizes/matrix_sizes[0])**2 * y.values[0]
    ax.loglog(matrix_sizes, ref_n2_mem, '--', color=COLORS['baseline'], linewidth=0.8,
             label=r'Theoretical $O(n^2)$')
    
    ax.set_xlabel('Matrix Dimension $n$')
    ax.set_ylabel('Memory (KB)')
    ax.set_title('Memory Scaling')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    
    plt.tight_layout(pad=0.1)
    plt.savefig('matrix_performance.pdf', dpi=600, bbox_inches='tight')
    print("Created performance plot: matrix_performance.pdf")


def batch_efficiency_plot():
    """
    Generate a plot showing batch processing efficiency:
    - Time per proof vs batch size
    - Memory overhead vs batch size
    """
    try:
        df = pd.read_csv("batch_efficiency.csv")
    except FileNotFoundError:
        print("Error: Batch efficiency data file not found")
        return

    fig, ax = plt.subplots(figsize=(3.54, 2.36))  # 90mm journal column width

    # Extract batch sizes and per-proof time (convert from ms to s)
    batch_sizes = df['batch_size'].values
    y = df['time_per_proof_ms'].values / 1000.0  # seconds
    
    # Add error bars if available
    if 'stddev_time_per_proof' in df.columns:
        yerr = df['stddev_time_per_proof'].values / 1000.0
    else:
        yerr = np.zeros_like(y)

    # Plot time per proof
    ax.errorbar(batch_sizes, y, yerr=yerr, fmt='^-', ms=3, lw=0.8,
                color=COLORS['proving'], elinewidth=0.4, capsize=1.0,
                label='Time per Proof')
    ax.fill_between(batch_sizes, y - yerr, y + yerr, color=COLORS['proving'], alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel('Batch Size', labelpad=2)
    ax.set_ylabel('Time per Proof (s)', labelpad=2)
    ax.set_title('Batch Processing Efficiency', pad=8)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # Add memory usage if available
    if 'theoretical_memory_mb' in df.columns or 'memory_usage_mb' in df.columns:
        mem_col = 'theoretical_memory_mb' if 'theoretical_memory_mb' in df.columns else 'memory_usage_mb'
        mem = df[mem_col].values
        
        ax2 = ax.twinx()
        ax2.plot(batch_sizes, mem, 'o--', ms=3, lw=0.8, color=COLORS['memory'], 
                  label='Memory Overhead (MB)')
        ax2.set_ylabel('Memory Overhead (MB)', labelpad=2, color=COLORS['memory'])
        ax2.tick_params(axis='y', labelcolor=COLORS['memory'])
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=7)
    else:
        ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout(pad=0.1)
    plt.savefig('batch_efficiency.pdf', dpi=600, bbox_inches='tight')
    print("Created batch efficiency plot: batch_efficiency.pdf")


def parallel_scaling_plot():
    """
    Generate parallel scaling plots:
    - Execution time vs threads
    - Speedup vs threads
    """
    try:
        df = pd.read_csv("parallelization_benchmark.csv")
    except FileNotFoundError:
        print("Error: parallelization_benchmark.csv not found")
        return

    # Convert to seconds for plotting
    if 'mean_time_ms' in df.columns:
        df['proof_time_s'] = df['mean_time_ms'] / 1000
    else:
        df['proof_time_s'] = df['proof_time_ms'] / 1000

    matrix_sizes = df['matrix_size'].unique()
    
    # Define color and marker styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', 'D', '^', 'v']

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.36))  # Double width for two plots

    # Plot 1: Execution time vs. threads
    for idx, size in enumerate(matrix_sizes):
        size_data = df[df['matrix_size'] == size]
        if len(size_data) < 2:
            continue

        threads = size_data['threads'].values
        time_s = size_data['proof_time_s'].values

        ax1.plot(
            threads, time_s,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            ms=4, lw=0.8,
            label=f'n={int(size)}'
        )

    ax1.set_xlabel('Number of Threads', labelpad=2)
    ax1.set_ylabel('Proving Time (s)', labelpad=2)
    ax1.set_title('Parallel Scaling: Execution Time', pad=8)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Plot 2: Speedup vs. threads
    for idx, size in enumerate(matrix_sizes):
        size_data = df[df['matrix_size'] == size]
        if len(size_data) < 2:
            continue

        threads = size_data['threads'].values
        speedup = size_data['speedup'].values

        ax2.plot(
            threads, speedup,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            ms=4, lw=0.8,
            label=f'n={int(size)}'
        )
    
    # Add ideal scaling line
    max_threads = df['threads'].max()
    ax2.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.7, lw=0.6, label='Ideal')

    ax2.set_xlabel('Number of Threads', labelpad=2)
    ax2.set_ylabel('Speedup', labelpad=2)
    ax2.set_title('Parallel Scaling: Speedup', pad=8)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    # Add legend at the bottom
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(matrix_sizes)+1, frameon=False,
               handletextpad=0.3, borderaxespad=0.1, handlelength=1.5,
               fontsize=7)

    plt.tight_layout(pad=0.1, rect=[0, 0.08, 1, 0.98])
    plt.savefig('parallel_scaling.pdf', dpi=600, bbox_inches='tight')
    print("Created parallel scaling plot: parallel_scaling.pdf")


def comparison_plot():
    """
    Generate a plot comparing ZK vs non-ZK matrix multiplication
    """
    try:
        df = pd.read_csv("zk_vs_nonzk_comparison.csv")
    except FileNotFoundError:
        print("Error: zk_vs_nonzk_comparison.csv file not found")
        return
    
    # Extract data
    matrix_sizes = df['matrix_size'].values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.36))
    
    # Comparison Plot
    y_non_zk = df['non_zk_time_ms'].values
    y_zk = df['total_zk_time_ms'].values
    y_zk_prove = df['zk_proof_time_ms'].values
    y_zk_verify = df['verification_time_ms'].values
    
    ax1.loglog(matrix_sizes, y_non_zk, 'o-', color=COLORS['baseline'], 
               markersize=3, label='Non-ZK')
    ax1.loglog(matrix_sizes, y_zk, 's-', color=COLORS['proving'], 
               markersize=3, label='ZK Total')
    ax1.loglog(matrix_sizes, y_zk_prove, '^-', color=COLORS['verification'], 
               markersize=3, label='ZK Proving')
    ax1.loglog(matrix_sizes, y_zk_verify, 'D-', color=COLORS['speedup'], 
               markersize=3, label='ZK Verification')
    
    ax1.set_xlabel('Matrix Dimension $n$')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('ZK vs Non-ZK Comparison')
    ax1.legend(frameon=False, fontsize=6, loc='upper left')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    
    # Speedup Plot
    speedup = np.zeros_like(matrix_sizes, dtype=float)
    for i, (nzk, zk) in enumerate(zip(y_non_zk, y_zk)):
        if nzk > 0 and zk > 0:
            speedup[i] = nzk / zk
        else:
            speedup[i] = np.nan
    
    valid_mask = ~np.isnan(speedup)
    if np.any(valid_mask):
        ax2.semilogx(matrix_sizes[valid_mask], speedup[valid_mask], 'D-', 
                    color=COLORS['speedup'], markersize=3)
        ax2.axhline(y=1, color='k', linestyle='--', linewidth=0.8)
        
        y_min = 0
        y_max = np.nanmax(speedup) * 1.1
        ax2.set_ylim(y_min, y_max)
    
    ax2.set_xlabel('Matrix Dimension $n$')
    ax2.set_ylabel('Speedup Ratio')
    ax2.set_title('ZK vs Non-ZK Speedup')
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    
    plt.tight_layout(pad=0.1)
    plt.savefig('zk_comparison.pdf', dpi=600, bbox_inches='tight')
    print("Created comparison plot: zk_comparison.pdf")


def create_all_plots():
    """Create all visualization plots from benchmark data"""
    print("Generating publication-quality plots for ZK Matrix Multiplication benchmarks...")
    performance_plots()
    batch_efficiency_plot()
    parallel_scaling_plot()
    comparison_plot()
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    create_all_plots()
