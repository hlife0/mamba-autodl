import sys
import os
# Add parent directory to path to import dataset module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from dataset.hotpot import HotpotQAIterator
from skip_layer_pre_130M.utils import inference_logic
from tqdm import tqdm


def allangle_hybrid_logic(ssm_fast_stack, ssm_slow_stack, top_k_percent=10.0):
    """
    Global angle strategy: use fast cache for layers [0, skip_layers), slow cache for layers [skip_layers, end].
    
    Args:
        ssm_fast_stack: Fast cache SSM states [num_layers, batch, d_inner, d_state]
        ssm_slow_stack: Slow cache SSM states [num_layers, batch, d_inner, d_state]
        top_k_percent: Percentage of dimensions with lowest cosine similarity to use slow cache
        
    Returns:
        ssm_hybrid_stack: Mixed SSM states [num_layers, batch, d_inner, d_state]
    """
    # Normalize vectors along d_state dimension
    ssm_fast_norm = F.normalize(ssm_fast_stack, p=2, dim=-1)
    ssm_slow_norm = F.normalize(ssm_slow_stack, p=2, dim=-1)
    
    # Compute cosine similarity (dot product of normalized vectors)
    # Shape: [num_layers, batch, d_inner]
    cosine_sim = (ssm_fast_norm * ssm_slow_norm).sum(dim=-1)
    
    # Flatten ALL dimensions across ALL layers for global bottom-k selection
    num_layers_actual = cosine_sim.shape[0]
    batch_size = cosine_sim.shape[1]
    d_inner = cosine_sim.shape[2]
    
    cosine_global_flat = cosine_sim.view(-1)
    
    # Compute k dimensions globally
    total_dims = cosine_global_flat.shape[0]
    k_dims_global = max(1, int(total_dims * top_k_percent / 100.0))
    
    # Get global bottom-k dimension indices (lowest cosine similarity)
    _, bottom_global_indices = torch.topk(cosine_global_flat, k_dims_global, dim=0, largest=False)
    
    # Create global mask
    global_mask = torch.zeros_like(cosine_global_flat, dtype=torch.bool)
    global_mask.scatter_(0, bottom_global_indices, True)
    
    # Reshape back to [num_layers, batch, d_inner]
    dim_masks = global_mask.view(num_layers_actual, batch_size, d_inner)
    
    # Expand mask to include d_state dimension
    d_state = ssm_fast_stack.shape[-1]
    full_masks = dim_masks.unsqueeze(-1).expand(-1, -1, -1, d_state)
    
    # Apply masks: use slow cache for bottom-k dimensions (lowest similarity), fast cache for others
    ssm_hybrid_stack = torch.where(full_masks, ssm_slow_stack, ssm_fast_stack)
    
    return ssm_hybrid_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global angle hybrid cache experiment: global bottom-k% cosine similarity")
    parser.add_argument('--top_k_percent', type=float, default=10.0, help='Percentage of dimensions with lowest cosine similarity to use slow cache')
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-130m', help='Model name or path')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_new_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--output_dir', type=str, default='./skip_layer_pre_130M/experiments', help='Output directory')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print(f"Loading model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Top-k percent: {args.top_k_percent}% (global bottom-k% with lowest cosine similarity use slow cache)\n")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    print("=" * 70)
    print("GLOBAL ANGLE HYBRID CACHE")
    print("=" * 70)
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Top-k percent: {args.top_k_percent}% (global bottom-k%)")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Seed: {args.seed}")
    print()
    
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    k_str = str(int(args.top_k_percent)) if args.top_k_percent == int(args.top_k_percent) else str(args.top_k_percent).replace('.', 'p')
    output_file = os.path.join(args.output_dir, f"allangle{k_str}_{timestamp}.csv")
    
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
                    return allangle_hybrid_logic(fast, slow, top_k_percent=args.top_k_percent)
                
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
