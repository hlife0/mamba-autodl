import sys
import os

# Add parent directory to path to import dataset module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from dataset.hotpot import HotpotQAIterator
from skip_layer_pre.utils import inference_logic
from tqdm import tqdm


def layervalue_hybrid_logic(ssm_fast_stack, ssm_slow_stack, top_k_percent=10.0):
    """
    Layer-wise value-level strategy: for each layer, select top-k% values with largest differences.
    
    Args:
        ssm_fast_stack: Fast cache SSM states [num_layers, batch, d_inner, d_state]
        ssm_slow_stack: Slow cache SSM states [num_layers, batch, d_inner, d_state]
        top_k_percent: Percentage of values per layer with largest differences to use slow cache
        
    Returns:
        ssm_hybrid_stack: Mixed SSM states [num_layers, batch, d_inner, d_state]
    """
    # Compute absolute differences for each individual value
    diff_values = torch.abs(ssm_fast_stack - ssm_slow_stack)
    
    # For each layer, flatten all values and select top-k%
    num_layers_actual = diff_values.shape[0]
    batch_size = diff_values.shape[1]
    d_inner = diff_values.shape[2]
    d_state = diff_values.shape[3]
    
    # Reshape to [num_layers, batch * d_inner * d_state]
    diff_flat = diff_values.view(num_layers_actual, -1)
    
    # Compute k values per layer
    values_per_layer = diff_flat.shape[1]
    k_values = max(1, int(values_per_layer * top_k_percent / 100.0))
    
    # Get top-k value indices for each layer
    _, top_value_indices = torch.topk(diff_flat, k_values, dim=1)
    
    # Create masks for values
    value_masks = torch.zeros_like(diff_flat, dtype=torch.bool)
    value_masks.scatter_(1, top_value_indices, True)
    
    # Reshape back to [num_layers, batch, d_inner, d_state]
    full_masks = value_masks.view(num_layers_actual, batch_size, d_inner, d_state)
    
    # Apply masks: use slow cache for top-k values, fast cache for others
    ssm_hybrid_stack = torch.where(full_masks, ssm_slow_stack, ssm_fast_stack)
    
    return ssm_hybrid_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise value-level hybrid cache experiment: per-layer top-k% largest value differences")
    parser.add_argument('--top_k_percent', type=float, default=10.0, help='Percentage of values per layer with largest differences to use slow cache')
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-2.8b', help='Model name or path')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_new_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--output_dir', type=str, default='./skip_layer_pre/experiments', help='Output directory')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print(f"Loading model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Top-k percent: {args.top_k_percent}% (per-layer top-k% with largest value differences use slow cache)\n")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    print("=" * 70)
    print("LAYER-WISE VALUE-LEVEL HYBRID CACHE")
    print("=" * 70)
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Top-k percent: {args.top_k_percent}% (per-layer top-k%)")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Seed: {args.seed}")
    print()
    
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    k_str = str(int(args.top_k_percent)) if args.top_k_percent == int(args.top_k_percent) else str(args.top_k_percent).replace('.', 'p')
    output_file = os.path.join(args.output_dir, f"layervalue{k_str}_{timestamp}.csv")
    
    fieldnames = ['id', 'decoded', 'answer', 'question', 'doc1_title', 'doc2_title', 'doc1_content', 'doc2_content']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        
        count = 0
        for item in tqdm(sample_dataset, desc="Processing"):
            docs = item.get_useful()
            if len(docs) < 2:
                continue
            
            doc1_with_title = docs[0]['title'] + ": " + docs[0]['content']
            doc2_with_title = docs[1]['title'] + ": " + docs[1]['content']
            
            try:
                # Create hybrid logic with top_k_percent parameter
                def hybrid_logic_func(fast, slow):
                    return layervalue_hybrid_logic(fast, slow, top_k_percent=args.top_k_percent)
                
                decoded = inference_logic(
                    model=model,
                    tokenizer=tokenizer,
                    hybrid_logic=hybrid_logic_func,
                    question=item.question,
                    doc1=doc1_with_title,
                    doc2=doc2_with_title,
                    max_new_tokens=args.max_new_tokens,
                    device=device
                )
            except Exception as e:
                print(f"\nError processing item {item.id}: {e}")
                decoded = ""
            
            writer.writerow({
                'id': item.id,
                'decoded': decoded,
                'answer': item.answer,
                'question': item.question,
                'doc1_title': docs[0]['title'],
                'doc2_title': docs[1]['title'],
                'doc1_content': docs[0]['content'],
                'doc2_content': docs[1]['content']
            })
            f.flush()
            count += 1
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"Processed: {count} questions")
    print()
