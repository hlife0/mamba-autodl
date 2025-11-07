import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize Alpha statistics")
    parser.add_argument('--data_path', type=str, 
                       default='/home/hlife/Mamba-experiment/attention_difference/experiments/alpha_beta_stats_06094409/5a7bb1315542997c3ec97253.pt',
                       help='Path to the .pt file')
    parser.add_argument('--n_channels', type=int, default=6, help='Number of random channels to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for channel selection')
    parser.add_argument('--output_dir', type=str, default='./attention_difference/visualizations', help='Output directory for plots')
    args = parser.parse_args()
    
    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = torch.load(args.data_path)
    print(f"File: {args.data_path}")
    print(f"Size: {os.path.getsize(args.data_path) / (1024**3):.2f} GB")
    print(f"\nKeys: {list(data.keys())}")
    
    # Print detailed info
    print("\n" + "=" * 70)
    print("DETAILED INFORMATION")
    print("=" * 70)
    for key, value in data.items():
        print(f"\n[{key}]")
        if isinstance(value, torch.Tensor):
            print(f"  Shape: {list(value.shape)} | Dtype: {value.dtype}")
            print(f"  Memory: {value.element_size() * value.nelement() / (1024**2):.2f} MB")
            print(f"  Range: [{value.min().item():.6f}, {value.max().item():.6f}]")
            print(f"  Mean±Std: {value.mean().item():.6f} ± {value.std().item():.6f}")
        else:
            print(f"  Value: {value}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get doc2_id for filename
    doc2_id = data['doc2_id']
    
    # Get alpha data
    alpha_mean = data['alpha_mean']  # [64, seqlen, 5120]
    alpha_var = data['alpha_var']
    
    # Randomly select channels (seed ensures reproducibility)
    np.random.seed(args.seed)
    selected_channels = np.random.choice(5120, args.n_channels, replace=False)
    selected_layers = [0, 15, 31, 47, 63]
    
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print(f"Doc2 ID: {doc2_id}")
    print(f"Random seed: {args.seed} (ensures same channels are selected each time)")
    print(f"Selected {args.n_channels} random channels: {selected_channels}")
    print(f"Selected layers for line plots: {selected_layers}")
    
    # ==================== ALPHA MEAN HEATMAPS ====================
    print("\n[1/4] Creating Alpha Mean heatmaps...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, ch in enumerate(selected_channels):
        alpha_ch = alpha_mean[:, :, ch].cpu().numpy()
        im = axes[idx].imshow(alpha_ch, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[idx].set_xlabel('Sequence Position')
        axes[idx].set_ylabel('Layer')
        axes[idx].set_title(f'Alpha Mean - Channel {ch}')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    output_file = os.path.join(args.output_dir, f'{doc2_id}_alpha_mean_heatmaps.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # ==================== ALPHA MEAN LINE PLOTS ====================
    print("[2/4] Creating Alpha Mean line plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, ch in enumerate(selected_channels):
        for layer in selected_layers:
            alpha_ch_layer = alpha_mean[layer, :, ch].cpu().numpy()
            axes[idx].plot(alpha_ch_layer, label=f'Layer {layer}', alpha=0.7)
        
        axes[idx].set_xlabel('Sequence Position')
        axes[idx].set_ylabel('Alpha Mean')
        axes[idx].set_title(f'Channel {ch} - Selected Layers')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(args.output_dir, f'{doc2_id}_alpha_mean_lines.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # ==================== ALPHA VARIANCE HEATMAPS ====================
    print("[3/4] Creating Alpha Variance heatmaps...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, ch in enumerate(selected_channels):
        alpha_var_ch = alpha_var[:, :, ch].cpu().numpy()
        im = axes[idx].imshow(alpha_var_ch, aspect='auto', cmap='hot', interpolation='nearest')
        axes[idx].set_xlabel('Sequence Position')
        axes[idx].set_ylabel('Layer')
        axes[idx].set_title(f'Alpha Variance - Channel {ch}')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    output_file = os.path.join(args.output_dir, f'{doc2_id}_alpha_var_heatmaps.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # ==================== ALPHA VARIANCE LINE PLOTS ====================
    print("[4/4] Creating Alpha Variance line plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, ch in enumerate(selected_channels):
        for layer in selected_layers:
            alpha_var_ch_layer = alpha_var[layer, :, ch].cpu().numpy()
            axes[idx].plot(alpha_var_ch_layer, label=f'Layer {layer}', alpha=0.7)
        
        axes[idx].set_xlabel('Sequence Position')
        axes[idx].set_ylabel('Alpha Variance')
        axes[idx].set_title(f'Channel {ch} Variance - Selected Layers')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(args.output_dir, f'{doc2_id}_alpha_var_lines.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # ==================== BETA VISUALIZATION ====================
    print("[5/5] Creating Beta visualization...")
    beta_mean = data['beta_mean']  # [64, 4, 5120, 16]
    beta_var = data['beta_var']
    
    beta_mean_avg = beta_mean.mean(dim=(2, 3))
    beta_var_avg = beta_var.mean(dim=(2, 3))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x_pos = [-4, -3, -2, -1]
    
    # Beta Mean
    im1 = axes[0, 0].imshow(beta_mean_avg.cpu().numpy(), aspect='auto', cmap='plasma', interpolation='nearest')
    axes[0, 0].set_xlabel('Last 4 Tokens of First Half')
    axes[0, 0].set_ylabel('Layer')
    axes[0, 0].set_title('Beta Mean (averaged over d_inner, d_state)')
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].set_xticklabels(['-4', '-3', '-2', '-1'])
    plt.colorbar(im1, ax=axes[0, 0])
    
    for layer in selected_layers:
        axes[0, 1].plot(x_pos, beta_mean_avg[layer].cpu().numpy(), marker='o', label=f'Layer {layer}', alpha=0.7)
    axes[0, 1].set_xlabel('Token Position')
    axes[0, 1].set_ylabel('Beta Mean')
    axes[0, 1].set_title('Beta Mean (selected layers)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Beta Variance
    im2 = axes[1, 0].imshow(beta_var_avg.cpu().numpy(), aspect='auto', cmap='hot', interpolation='nearest')
    axes[1, 0].set_xlabel('Last 4 Tokens of First Half')
    axes[1, 0].set_ylabel('Layer')
    axes[1, 0].set_title('Beta Variance (averaged over d_inner, d_state)')
    axes[1, 0].set_xticks([0, 1, 2, 3])
    axes[1, 0].set_xticklabels(['-4', '-3', '-2', '-1'])
    plt.colorbar(im2, ax=axes[1, 0])
    
    for layer in selected_layers:
        axes[1, 1].plot(x_pos, beta_var_avg[layer].cpu().numpy(), marker='o', label=f'Layer {layer}', alpha=0.7)
    axes[1, 1].set_xlabel('Token Position')
    axes[1, 1].set_ylabel('Beta Variance')
    axes[1, 1].set_title('Beta Variance (selected layers)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(args.output_dir, f'{doc2_id}_beta_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Doc2 ID: {doc2_id}")
    print(f"Aggregated over: {data['num_doc1_samples']} doc1 samples")
    print(f"Max sequence length: {data['max_seqlen']} tokens")
    print()
    print("Generated visualizations:")
    print(f"  ✓ {doc2_id}_alpha_mean_heatmaps.png")
    print(f"  ✓ {doc2_id}_alpha_mean_lines.png")
    print(f"  ✓ {doc2_id}_alpha_var_heatmaps.png")
    print(f"  ✓ {doc2_id}_alpha_var_lines.png")
    print(f"  ✓ {doc2_id}_beta_visualization.png")
    print()
    print(f"All saved to: {args.output_dir}")
    print(f"\nSeed={args.seed} ensures reproducible channel selection")
    print("  (same seed → same random channels each time)")
    print("=" * 70)

if __name__ == "__main__":
    main()

