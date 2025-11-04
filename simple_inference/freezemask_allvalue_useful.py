#!/usr/bin/env python3
"""
Freeze mask generation for global value strategy.
Records which cache positions are selected across all samples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from dataset.hotpot import HotpotQAIterator
from tqdm import tqdm


def prefill_from_scratch(model, input_ids, device):
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    inference_params = InferenceParams(max_seqlen=1024, max_batch_size=batch_size)
    with torch.no_grad():
        logits = model(input_ids, inference_params=inference_params, num_last_tokens=1).logits
    inference_params.seqlen_offset += prompt_len
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return inference_params, next_token_id


def get_mask_allvalue(cache_fast, cache_slow, top_k_percent):
    """
    Get mask for global value strategy.
    Returns: mask [num_layers, batch, d_inner, d_state] (True = use slow cache)
    """
    num_layers = len(cache_fast.key_value_memory_dict)
    
    # Stack SSM states
    ssm_fast_stack = torch.stack([cache_fast.key_value_memory_dict[i][1] for i in range(num_layers)])
    ssm_slow_stack = torch.stack([cache_slow.key_value_memory_dict[i][1] for i in range(num_layers)])
    
    # Compute absolute differences
    diff_values = torch.abs(ssm_fast_stack - ssm_slow_stack)  # [num_layers, batch, d_inner, d_state]
    
    # Global value-level top-k selection
    diff_global_flat = diff_values.view(-1)
    total_values = diff_global_flat.shape[0]
    k_values_global = max(1, int(total_values * top_k_percent / 100.0))
    
    _, top_global_indices = torch.topk(diff_global_flat, k_values_global, dim=0)
    
    global_mask = torch.zeros_like(diff_global_flat, dtype=torch.bool)
    global_mask.scatter_(0, top_global_indices, True)
    
    num_layers_actual = ssm_fast_stack.shape[0]
    batch_size = ssm_fast_stack.shape[1]
    d_inner = ssm_fast_stack.shape[2]
    d_state = ssm_fast_stack.shape[3]
    
    full_masks = global_mask.view(num_layers_actual, batch_size, d_inner, d_state)
    
    return full_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate freeze mask for global value strategy")
    parser.add_argument('top_k_percent', type=float, help='Percentage of values to use slow cache')
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-2.8b', help='Model name or path')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./simple_inference/masks', help='Output directory')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print(f"Loading model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Top-k percent: {args.top_k_percent}%\n")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    # Few-shot prompt
    part_A = (
        "Q: Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    # Initialize accumulator (will determine shape from first sample)
    accumulator = None
    count = 0
    
    print("=" * 70)
    print("GENERATING FREEZE MASK (GLOBAL VALUE)")
    print("=" * 70)
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Top-k percent: {args.top_k_percent}%")
    print()
    
    for item in tqdm(sample_dataset, desc="Processing"):
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        
        doc1_with_title = docs[0]['title'] + ": " + docs[0]['content']
        doc2_with_title = docs[1]['title'] + ": " + docs[1]['content']
        
        part_B = f"Document 1: {doc1_with_title}\n\n"
        part_C = f"Document 2: {doc2_with_title}\n\n"
        
        try:
            # Fast path: fewshot + doc2
            prompt_fast = part_A + part_C
            tokens_fast = tokenizer(prompt_fast, return_tensors="pt")
            input_ids_fast = tokens_fast.input_ids.to(device)
            cache_fast, _ = prefill_from_scratch(model, input_ids_fast, device)
            
            # Slow path: fewshot + doc1 + doc2
            prompt_slow = part_A + part_B + part_C
            tokens_slow = tokenizer(prompt_slow, return_tensors="pt")
            input_ids_slow = tokens_slow.input_ids.to(device)
            cache_slow, _ = prefill_from_scratch(model, input_ids_slow, device)
            
            # Get mask
            mask = get_mask_allvalue(cache_fast, cache_slow, args.top_k_percent)
            
            # Initialize accumulator on first sample
            if accumulator is None:
                accumulator = torch.zeros_like(mask, dtype=torch.int32)
            
            # Accumulate
            accumulator += mask.int()
            count += 1
            
        except Exception as e:
            print(f"\nError processing item {item.id}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print("SAVING MASK")
    print("=" * 70)
    
    # Convert to numpy and save
    accumulator_np = accumulator.cpu().numpy()
    
    os.makedirs(args.output_dir, exist_ok=True)
    k_str = str(int(args.top_k_percent)) if args.top_k_percent == int(args.top_k_percent) else str(args.top_k_percent).replace('.', 'p')
    output_file = os.path.join(args.output_dir, f"mask_allvalue{k_str}_useful.npy")
    
    np.save(output_file, accumulator_np)
    
    print(f"Output file: {output_file}")
    print(f"Shape: {accumulator_np.shape}")
    print(f"Processed: {count} samples")
    print(f"Min value: {accumulator_np.min()}")
    print(f"Max value: {accumulator_np.max()}")
    print(f"Mean value: {accumulator_np.mean():.2f}")
    print()
