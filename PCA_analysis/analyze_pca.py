#!/usr/bin/env python3
"""
PCA analysis on cache differences.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from glob import glob
from tqdm import tqdm
from datetime import datetime


def load_cache_diffs(data_dir):
    """Load all cache difference tensors from a directory."""
    pt_files = sorted(glob(os.path.join(data_dir, "diff_*.pt")))
    
    if len(pt_files) == 0:
        raise ValueError(f"No .pt files found in {data_dir}")
    
    print(f"Found {len(pt_files)} cache difference files")
    
    cache_diffs = []
    metadata = []
    
    for pt_file in tqdm(pt_files, desc="Loading data"):
        data = torch.load(pt_file, map_location='cpu')
        cache_diff = data['cache_diff']  # Shape: [num_layers, d_inner, d_state]
        
        cache_diffs.append(cache_diff)
        metadata.append({
            'file': os.path.basename(pt_file),
            'item_id': data.get('item_id', 'unknown'),
            'question': data.get('question', ''),
            'answer': data.get('answer', ''),
        })
    
    # Stack all cache diffs: [n_samples, num_layers, d_inner, d_state]
    cache_diffs = torch.stack(cache_diffs, dim=0)
    
    print(f"Loaded cache diffs shape: {cache_diffs.shape}")
    print(f"  n_samples: {cache_diffs.shape[0]}")
    print(f"  num_layers: {cache_diffs.shape[1]}")
    print(f"  d_inner: {cache_diffs.shape[2]}")
    print(f"  d_state: {cache_diffs.shape[3]}")
    
    return cache_diffs, metadata


def prepare_data_for_pca(cache_diffs, flatten_mode='all'):
    """
    Prepare data for PCA analysis.
    
    Args:
        cache_diffs: torch.Tensor of shape [n_samples, num_layers, d_inner, d_state]
        flatten_mode: How to flatten the data
            - 'all': Flatten everything to [n_samples, num_layers*d_inner*d_state]
            - 'layer_mean': Average over layers to [n_samples, d_inner*d_state]
            - 'state_mean': Average over d_state to [n_samples, num_layers*d_inner]
    
    Returns:
        numpy array of shape [n_samples, n_features]
    """
    cache_diffs = cache_diffs.numpy()
    n_samples = cache_diffs.shape[0]
    
    if flatten_mode == 'all':
        # Flatten to [n_samples, num_layers*d_inner*d_state]
        X = cache_diffs.reshape(n_samples, -1)
        print(f"Flatten mode 'all': [{n_samples}, {X.shape[1]}]")
        
    elif flatten_mode == 'layer_mean':
        # Average over layers: [n_samples, num_layers, d_inner, d_state] -> [n_samples, d_inner, d_state]
        X = cache_diffs.mean(axis=1)
        X = X.reshape(n_samples, -1)
        print(f"Flatten mode 'layer_mean': [{n_samples}, {X.shape[1]}]")
        
    elif flatten_mode == 'state_mean':
        # Average over d_state: [n_samples, num_layers, d_inner, d_state] -> [n_samples, num_layers, d_inner]
        X = cache_diffs.mean(axis=3)
        X = X.reshape(n_samples, -1)
        print(f"Flatten mode 'state_mean': [{n_samples}, {X.shape[1]}]")
        
    else:
        raise ValueError(f"Unknown flatten_mode: {flatten_mode}")
    
    return X


def perform_pca(X, target_variance=0.99, standardize=True):
    """
    Perform PCA analysis until target variance is reached.
    
    Args:
        X: numpy array of shape [n_samples, n_features]
        target_variance: Target cumulative explained variance (default: 0.99)
        standardize: Whether to standardize features before PCA
    
    Returns:
        pca: Fitted PCA object
        X_transformed: Transformed data
        scaler: StandardScaler object (or None)
        n_components_used: Number of components used to reach target variance
    """
    print(f"\nPerforming PCA on data with shape: {X.shape}")
    print(f"Target: {target_variance*100:.1f}% cumulative variance")
    
    # Standardize features
    scaler = None
    if standardize:
        print("Standardizing features...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Calculate all possible components first (with a hard limit of 2000)
    max_components = min(X.shape[0], X.shape[1], 2000)
    print(f"\nCalculating up to {max_components} components...")
    print("Using randomized SVD for efficiency...")
    print("This may take a while...\n")
    
    # Fit PCA with randomized SVD (avoids integer overflow for large matrices)
    pca = PCA(n_components=max_components, svd_solver='randomized', random_state=42)
    X_transformed = pca.fit_transform(X)
    
    # Monitor cumulative variance and print every 10 components
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print("Progress (every 10 components):")
    print("-" * 60)
    print(f"{'Components':<15} {'Cumulative Variance':<25} {'Status'}")
    print("-" * 60)
    
    n_components_used = None
    for i in range(max_components):
        # Print every 10 components
        if (i + 1) % 10 == 0:
            print(f"PC 1-{i+1:<10} {cumsum[i]:.6f} ({cumsum[i]*100:.2f}%)", end="")
            if n_components_used is None and cumsum[i] >= target_variance:
                print(f"  ✓ Reached {target_variance*100:.0f}%!")
                n_components_used = i + 1
            else:
                print()
        
        # Check if we reached target
        if n_components_used is None and cumsum[i] >= target_variance:
            if (i + 1) % 10 != 0:  # Print if not already printed
                print(f"PC 1-{i+1:<10} {cumsum[i]:.6f} ({cumsum[i]*100:.2f}%)  ✓ Reached {target_variance*100:.0f}%!")
            n_components_used = i + 1
    
    print("-" * 60)
    
    if n_components_used is None:
        n_components_used = max_components
        print(f"\nWarning: Only reached {cumsum[-1]*100:.2f}% variance with all {max_components} components")
    else:
        print(f"\n✓ Successfully reached {target_variance*100:.0f}% variance with {n_components_used} components")
        print(f"  Actual variance covered: {cumsum[n_components_used-1]*100:.4f}%")
    
    # Trim to only keep components up to target
    X_transformed = X_transformed[:, :n_components_used]
    
    return pca, X_transformed, scaler, n_components_used


def plot_scree_plot(pca, output_dir):
    """Plot scree plot showing explained variance for ALL components."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Show ALL components
    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)
    
    # Plot 1: Explained variance ratio
    ax1.bar(components, pca.explained_variance_ratio_, alpha=0.7, color='steelblue', width=1.0)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title(f'Scree Plot - All {n_components} Components', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    if n_components > 100:
        ax1.set_xlim([0, n_components + 1])
    
    # Plot 2: Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(components, cumsum, linestyle='-', linewidth=2, color='darkred')
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.5, label='80% variance')
    ax2.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, label='90% variance')
    ax2.axhline(y=0.95, color='lightgray', linestyle='-.', linewidth=1.5, label='95% variance')
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title(f'Cumulative Variance - All {n_components} Components', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, n_components + 1])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'scree_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved scree plot to {output_file}")
    plt.close()


def plot_pc_scatter(X_transformed, output_dir, pc_pairs=[(0, 1), (0, 2), (1, 2)]):
    """Plot scatter plots of principal components."""
    n_pairs = len(pc_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    
    if n_pairs == 1:
        axes = [axes]
    
    for ax, (pc1, pc2) in zip(axes, pc_pairs):
        ax.scatter(X_transformed[:, pc1], X_transformed[:, pc2], 
                   alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'PC{pc1 + 1}', fontsize=12)
        ax.set_ylabel(f'PC{pc2 + 1}', fontsize=12)
        ax.set_title(f'PC{pc1 + 1} vs PC{pc2 + 1}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'pc_scatter.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PC scatter plots to {output_file}")
    plt.close()


def save_pca_results(pca, X_transformed, metadata, output_dir, n_components_used, scaler=None):
    """Save PCA results to files."""
    # Save explained variance
    variance_file = os.path.join(output_dir, 'explained_variance.txt')
    with open(variance_file, 'w') as f:
        f.write("Principal Component Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        
        # Only save up to n_components_used
        f.write(f"Components Used: {n_components_used}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'PC':<6} {'Variance':<15} {'Cumulative':<15}\n")
        f.write("-" * 70 + "\n")
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        for i in range(n_components_used):
            f.write(f"PC{i+1:<4} {pca.explained_variance_ratio_[i]:<15.6f} {cumsum[i]:<15.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total components used: {n_components_used}\n")
        f.write(f"Variance explained by used components: {cumsum[n_components_used-1]:.6f}\n\n")
        
        # Find components needed for various thresholds
        for threshold in [0.8, 0.9, 0.95, 0.99]:
            idx = np.argmax(cumsum >= threshold)
            if idx < len(cumsum) and cumsum[idx] >= threshold:
                n_comp = idx + 1
                f.write(f"Components needed for {threshold*100:.0f}% variance: {n_comp} (actual: {cumsum[idx]:.4f})\n")
            else:
                f.write(f"Components needed for {threshold*100:.0f}% variance: >{n_components_used} (max achieved: {cumsum[n_components_used-1]:.4f})\n")
    
    print(f"Saved explained variance to {variance_file}")
    
    # Save transformed data
    transformed_file = os.path.join(output_dir, 'pca_transformed.npy')
    np.save(transformed_file, X_transformed)
    print(f"Saved transformed data to {transformed_file}")
    
    # Save PCA components
    components_file = os.path.join(output_dir, 'pca_components.npy')
    np.save(components_file, pca.components_)
    print(f"Saved PCA components to {components_file}")
    
    # Save PCA mean (for centering new data)
    mean_file = os.path.join(output_dir, 'pca_mean.npy')
    np.save(mean_file, pca.mean_)
    print(f"Saved PCA mean to {mean_file}")
    
    # Save StandardScaler parameters if used
    if scaler is not None:
        scaler_mean_file = os.path.join(output_dir, 'scaler_mean.npy')
        scaler_scale_file = os.path.join(output_dir, 'scaler_scale.npy')
        np.save(scaler_mean_file, scaler.mean_)
        np.save(scaler_scale_file, scaler.scale_)
        print(f"Saved scaler parameters to {scaler_mean_file} and {scaler_scale_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA analysis on cache differences")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing cache difference .pt files')
    parser.add_argument('--output_base_dir', type=str, default='./PCA_analysis/experiments',
                        help='Base directory for experiments (default: ./PCA_analysis/experiments)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Specific output directory (default: output_base_dir/timestamp)')
    parser.add_argument('--flatten_mode', type=str, default='all',
                        choices=['all', 'layer_mean', 'state_mean'],
                        help='How to flatten data for PCA')
    parser.add_argument('--target_variance', type=float, default=0.99,
                        help='Target cumulative variance to reach (default: 0.99)')
    parser.add_argument('--no_standardize', action='store_true',
                        help='Do not standardize features before PCA')
    args = parser.parse_args()
    
    # Setup output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%d%H%M%S")
        args.output_dir = os.path.join(args.output_base_dir, f"pca_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PCA ANALYSIS ON CACHE DIFFERENCES")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Flatten mode: {args.flatten_mode}")
    print(f"Target variance: {args.target_variance*100:.1f}%")
    print(f"Standardize: {not args.no_standardize}")
    print()
    
    # Load data
    print("Step 1: Loading cache differences...")
    cache_diffs, metadata = load_cache_diffs(args.data_dir)
    
    # Prepare data
    print("\nStep 2: Preparing data for PCA...")
    X = prepare_data_for_pca(cache_diffs, flatten_mode=args.flatten_mode)
    
    # Perform PCA
    print("\nStep 3: Performing PCA...")
    pca, X_transformed, scaler, n_components_used = perform_pca(
        X, 
        target_variance=args.target_variance,
        standardize=not args.no_standardize
    )
    
    print(f"\n→ Using {n_components_used} components for analysis and visualization")
    
    # Plot results
    print("\nStep 4: Generating plots...")
    plot_scree_plot(pca, args.output_dir)  # Show ALL components
    plot_pc_scatter(X_transformed, args.output_dir, pc_pairs=[(0, 1), (0, 2), (1, 2)])
    
    # Save results
    print("\nStep 5: Saving results...")
    save_pca_results(pca, X_transformed, metadata, args.output_dir, n_components_used, scaler)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Components used: {n_components_used}")
    print(f"Target variance: {args.target_variance*100:.1f}%")
    print(f"Achieved variance: {np.cumsum(pca.explained_variance_ratio_)[n_components_used-1]*100:.4f}%")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - scree_plot.png: Variance explained visualization")
    print(f"  - pc_scatter.png: Principal component scatter plots")
    print(f"  - explained_variance.txt: Detailed variance statistics")
    print(f"  - pca_transformed.npy: Transformed data ({n_components_used} components)")
    print(f"  - pca_components.npy: PCA components")
    print()

