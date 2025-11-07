#!/usr/bin/env python3
"""
Transform new cache differences using pre-computed PCA.
"""

import argparse
import numpy as np
import torch
import os


def load_pca_model(pca_dir):
    """Load PCA transformation parameters."""
    components = np.load(os.path.join(pca_dir, 'pca_components.npy'))
    pca_mean = np.load(os.path.join(pca_dir, 'pca_mean.npy'))
    
    # Load scaler parameters if they exist
    scaler_mean_file = os.path.join(pca_dir, 'scaler_mean.npy')
    scaler_scale_file = os.path.join(pca_dir, 'scaler_scale.npy')
    
    if os.path.exists(scaler_mean_file) and os.path.exists(scaler_scale_file):
        scaler_mean = np.load(scaler_mean_file)
        scaler_scale = np.load(scaler_scale_file)
    else:
        scaler_mean = None
        scaler_scale = None
    
    return {
        'components': components,
        'pca_mean': pca_mean,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'n_components': components.shape[0],
        'n_features': components.shape[1]
    }


def flatten_cache_diff(cache_diff, flatten_mode='all'):
    """
    Flatten cache difference tensor.
    
    Args:
        cache_diff: numpy array or torch.Tensor of shape [num_layers, d_inner, d_state]
        flatten_mode: 'all', 'layer_mean', or 'state_mean'
    
    Returns:
        Flattened 1D numpy array
    """
    if isinstance(cache_diff, torch.Tensor):
        cache_diff = cache_diff.numpy()
    
    if flatten_mode == 'all':
        return cache_diff.flatten()
    elif flatten_mode == 'layer_mean':
        return cache_diff.mean(axis=0).flatten()
    elif flatten_mode == 'state_mean':
        return cache_diff.mean(axis=2).flatten()
    else:
        raise ValueError(f"Unknown flatten_mode: {flatten_mode}")


def transform_cache_diff(cache_diff, pca_model, flatten_mode='all'):
    """
    Transform a cache difference using PCA.
    
    Args:
        cache_diff: numpy array or torch.Tensor of shape [num_layers, d_inner, d_state]
        pca_model: Dictionary containing PCA parameters (from load_pca_model)
        flatten_mode: 'all', 'layer_mean', or 'state_mean'
    
    Returns:
        Transformed array of shape [n_components]
    """
    # Step 1: Flatten
    x = flatten_cache_diff(cache_diff, flatten_mode)
    
    # Check dimensions
    if len(x) != pca_model['n_features']:
        raise ValueError(
            f"Feature dimension mismatch! Expected {pca_model['n_features']}, got {len(x)}. "
            f"Make sure you're using the same flatten_mode as training."
        )
    
    # Step 2: Standardize (if scaler was used)
    if pca_model['scaler_mean'] is not None:
        x = (x - pca_model['scaler_mean']) / pca_model['scaler_scale']
    
    # Step 3: Center (subtract PCA mean)
    x = x - pca_model['pca_mean']
    
    # Step 4: Project onto principal components
    x_pca = x @ pca_model['components'].T
    
    return x_pca


def transform_batch(cache_diffs, pca_model, flatten_mode='all'):
    """
    Transform a batch of cache differences.
    
    Args:
        cache_diffs: numpy array or torch.Tensor of shape [n_samples, num_layers, d_inner, d_state]
        pca_model: Dictionary containing PCA parameters
        flatten_mode: 'all', 'layer_mean', or 'state_mean'
    
    Returns:
        Transformed array of shape [n_samples, n_components]
    """
    if isinstance(cache_diffs, torch.Tensor):
        cache_diffs = cache_diffs.numpy()
    
    n_samples = cache_diffs.shape[0]
    results = []
    
    for i in range(n_samples):
        x_pca = transform_cache_diff(cache_diffs[i], pca_model, flatten_mode)
        results.append(x_pca)
    
    return np.array(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform cache differences using pre-computed PCA")
    parser.add_argument('--pca_dir', type=str, required=True,
                        help='Directory containing PCA results (e.g., PCA_analysis/experiments/pca_XXXXXX)')
    parser.add_argument('--cache_diff_file', type=str, required=True,
                        help='Path to cache difference .pt file')
    parser.add_argument('--flatten_mode', type=str, default='all',
                        choices=['all', 'layer_mean', 'state_mean'],
                        help='Must match the mode used during PCA training')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for transformed data (default: same dir as input)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PCA TRANSFORMATION")
    print("=" * 70)
    print(f"PCA directory: {args.pca_dir}")
    print(f"Input file: {args.cache_diff_file}")
    print(f"Flatten mode: {args.flatten_mode}")
    print()
    
    # Load PCA model
    print("Loading PCA model...")
    pca_model = load_pca_model(args.pca_dir)
    print(f"  Components: {pca_model['n_components']}")
    print(f"  Features: {pca_model['n_features']}")
    print(f"  Standardization: {'Yes' if pca_model['scaler_mean'] is not None else 'No'}")
    print()
    
    # Load cache difference
    print("Loading cache difference...")
    data = torch.load(args.cache_diff_file, map_location='cpu')
    cache_diff = data['cache_diff']
    print(f"  Shape: {cache_diff.shape}")
    print()
    
    # Transform
    print("Transforming...")
    x_pca = transform_cache_diff(cache_diff, pca_model, args.flatten_mode)
    print(f"  Transformed shape: {x_pca.shape}")
    print()
    
    # Save
    if args.output_file is None:
        base_name = os.path.basename(args.cache_diff_file).replace('.pt', '_pca.npy')
        args.output_file = os.path.join(os.path.dirname(args.cache_diff_file), base_name)
    
    np.save(args.output_file, x_pca)
    print(f"Saved to: {args.output_file}")
    print()
    
    # Show statistics
    print("Statistics:")
    print(f"  Min: {x_pca.min():.6f}")
    print(f"  Max: {x_pca.max():.6f}")
    print(f"  Mean: {x_pca.mean():.6f}")
    print(f"  Std: {x_pca.std():.6f}")
    print()

