#!/usr/bin/env python3
"""
All-dimension angle analysis for Mamba cache mixing strategy.

This script implements a hybrid caching strategy that considers cosine similarity
across ALL dimensions in ALL layers, selecting global bottom-k% for mixing.

The key idea: Consider ALL dimensions across ALL layers, compute cosine similarities,
and select the global bottom-k% dimensions (lowest similarity) to use the "good" cache.
"""

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
from mamba_ssm.utils.generation import InferenceParams
from dataset.hotpot import HotpotQAIterator
from tqdm import tqdm


def prefill_from_scratch(model, input_ids, device):
    """
    Prefill from scratch without any existing cache.
    
    Args:
        model: Mamba model
        input_ids: Input token IDs [batch_size, seq_len]
        device: Device to run on
        
    Returns:
        inference_params: Contains SSM hidden states and convolution cache
        next_token_id: The predicted next token ID [batch_size, 1]
    """
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    
    # Initialize inference parameters (cache)
    inference_params = InferenceParams(
        max_seqlen=1024,  # Set a reasonable max length
        max_batch_size=batch_size
    )
    
    # Forward pass to populate cache
    with torch.no_grad():
        logits = model(
            input_ids,
            inference_params=inference_params,
            num_last_tokens=1
        ).logits
    
    # Update seqlen_offset to reflect processed tokens
    inference_params.seqlen_offset += prompt_len
    
    # Get next token
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    return inference_params, next_token_id


def prefill_with_cache(model, input_ids, previous_cache, device):
    """
    Continue prefilling with existing cache.
    
    Args:
        model: Mamba model
        input_ids: Additional input token IDs [batch_size, seq_len]
        previous_cache: Existing InferenceParams with cached states
        device: Device to run on
        
    Returns:
        inference_params: Updated cache
        next_token_id: The predicted next token ID [batch_size, 1]
    """
    additional_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    
    # When cache exists (seqlen_offset > 0), we need to process tokens one by one
    # because the model enters step mode
    logits = None
    for i in range(additional_len):
        current_token = input_ids[:, i:i+1]
        
        # Create position_ids for current position
        position_ids = torch.full(
            (batch_size, 1),
            previous_cache.seqlen_offset,
            dtype=torch.long,
            device=device
        )
        
        with torch.no_grad():
            logits = model(
                current_token,
                position_ids=position_ids,
                inference_params=previous_cache,
                num_last_tokens=1
            ).logits
        
        # Update seqlen_offset after each token
        previous_cache.seqlen_offset += 1
    
    # Get next token from the last logits
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    return previous_cache, next_token_id


def decode_with_cache(model, token_id, cache, device):
    """
    Decode one token using cache.
    
    Args:
        model: Mamba model
        token_id: Current token ID [batch_size, 1]
        cache: InferenceParams with cached states
        device: Device to run on
        
    Returns:
        cache: Updated cache
        next_token_id: The predicted next token ID [batch_size, 1]
    """
    batch_size = token_id.shape[0]
    
    # Create position_ids for current position
    position_ids = torch.full(
        (batch_size, 1),
        cache.seqlen_offset,
        dtype=torch.long,
        device=device
    )
    
    # Forward pass with single token
    with torch.no_grad():
        logits = model(
            token_id,
            position_ids=position_ids,
            inference_params=cache,
            num_last_tokens=1
        ).logits
    
    # Update seqlen_offset
    cache.seqlen_offset += 1
    
    # Get next token
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    return cache, next_token_id


def inference_allangle(
    model,
    tokenizer,
    question,
    doc1,
    doc2,
    top_k_percent=10.0,
    max_new_tokens=30,
    device="cuda",
    verbose=False
):
    """
    Hybrid cache inference using global angle (cosine similarity) mixing strategy.
    
    Strategy:
    - Fast path: few-shot + doc2 (less context, faster)
    - Slow path: few-shot + doc1 + doc2 (full context, slower but better)
    - Hybrid: Consider ALL dimensions across ALL layers, select global bottom-k%
    
    Args:
        model: Mamba model
        tokenizer: Tokenizer
        question: Question string
        doc1: Document 1 string
        doc2: Document 2 string
        top_k_percent: Percentage of ALL dimensions with lowest cosine similarity to use "good" cache (default: 10%)
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        verbose: Print detailed progress (default: False)
        
    Returns:
        answer: Generated answer string
    """
    # Part A: Few-shot examples
    part_A = (
        "Q: Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    
    # Part B: Document 1
    part_B = f"Document 1: {doc1}\n\n"
    
    # Part C: Document 2
    part_C = f"Document 2: {doc2}\n\n"
    
    # Part D: Question
    part_D = f"Q: {question}\n\nA:"
    
    if verbose:
        print("=" * 70)
        print("HYBRID CACHE INFERENCE (GLOBAL ANGLE)")
        print("=" * 70)
        print(f"Bottom-k percent (lowest cosine similarity): {top_k_percent}%")
        print(f"Max new tokens: {max_new_tokens}")
        print()
    
    # ========== Path 1: Fast but poor (only doc2) ==========
    if verbose:
        print("Path 1: Fast (few-shot + doc2)")
    prompt_fast = part_A + part_C
    tokens_fast = tokenizer(prompt_fast, return_tensors="pt")
    input_ids_fast = tokens_fast.input_ids.to(device)
    
    cache_fast, _ = prefill_from_scratch(model, input_ids_fast, device)
    if verbose:
        print(f"  Tokens processed: {cache_fast.seqlen_offset}")
        print(f"  Layers: {len(cache_fast.key_value_memory_dict)}")
    
    # ========== Path 2: Slow but good (doc1 + doc2) ==========
    if verbose:
        print("\nPath 2: Slow (few-shot + doc1 + doc2)")
    prompt_slow = part_A + part_B + part_C
    tokens_slow = tokenizer(prompt_slow, return_tensors="pt")
    input_ids_slow = tokens_slow.input_ids.to(device)
    
    cache_slow, _ = prefill_from_scratch(model, input_ids_slow, device)
    if verbose:
        print(f"  Tokens processed: {cache_slow.seqlen_offset}")
    
    # ========== Create hybrid cache ==========
    if verbose:
        print("\nCreating hybrid cache (global angle)...")
    num_layers = len(cache_fast.key_value_memory_dict)
    
    # Initialize hybrid cache by copying fast cache structure
    cache_hybrid = InferenceParams(
        max_seqlen=cache_fast.max_seqlen,
        max_batch_size=cache_fast.max_batch_size
    )
    cache_hybrid.seqlen_offset = cache_slow.seqlen_offset  # Use slow path's offset
    
    # ========== Vectorized cache mixing with GLOBAL cosine similarity ==========
    # Stack all SSM states into tensors for batch processing
    # Shape: [num_layers, batch, d_inner, d_state]
    ssm_fast_stack = torch.stack([cache_fast.key_value_memory_dict[i][1] for i in range(num_layers)])
    ssm_slow_stack = torch.stack([cache_slow.key_value_memory_dict[i][1] for i in range(num_layers)])
    
    # For each (layer, dimension), treat the d_state vector as a single unit
    # Compute cosine similarity between fast and slow versions
    # Shape: [num_layers, batch, d_inner, d_state]
    
    # Normalize vectors along d_state dimension
    # Shape: [num_layers, batch, d_inner, d_state]
    ssm_fast_norm = F.normalize(ssm_fast_stack, p=2, dim=-1)
    ssm_slow_norm = F.normalize(ssm_slow_stack, p=2, dim=-1)
    
    # Compute cosine similarity (dot product of normalized vectors)
    # Shape: [num_layers, batch, d_inner]
    cosine_sim = (ssm_fast_norm * ssm_slow_norm).sum(dim=-1)
    
    # Flatten ALL dimensions across ALL layers for global bottom-k selection
    # Shape: [num_layers * batch * d_inner]
    num_layers_actual = cosine_sim.shape[0]
    batch_size = cosine_sim.shape[1]
    d_inner = cosine_sim.shape[2]
    
    cosine_global_flat = cosine_sim.view(-1)
    
    # Compute k dimensions globally
    total_dims = cosine_global_flat.shape[0]
    k_dims_global = max(1, int(total_dims * top_k_percent / 100.0))
    
    if verbose:
        print(f"  Total dimensions: {total_dims}")
        print(f"  Selecting bottom {k_dims_global} dimensions ({top_k_percent}%) with lowest cosine similarity")
    
    # Get global bottom-k dimension indices (lowest cosine similarity)
    # Shape: [k_dims_global]
    _, bottom_global_indices = torch.topk(cosine_global_flat, k_dims_global, dim=0, largest=False)
    
    # Create global mask
    # Shape: [num_layers * batch * d_inner]
    global_mask = torch.zeros_like(cosine_global_flat, dtype=torch.bool)
    global_mask.scatter_(0, bottom_global_indices, True)
    
    # Reshape back to [num_layers, batch, d_inner]
    dim_masks = global_mask.view(num_layers_actual, batch_size, d_inner)
    
    # Expand mask to include d_state dimension
    # Shape: [num_layers, batch, d_inner, d_state]
    d_state = ssm_fast_stack.shape[-1]
    full_masks = dim_masks.unsqueeze(-1).expand(-1, -1, -1, d_state)
    
    # Apply masks: use slow cache for bottom-k dimensions (lowest similarity), fast cache for others
    ssm_hybrid_stack = torch.where(full_masks, ssm_slow_stack, ssm_fast_stack)
    
    if verbose:
        selected_count = full_masks.sum().item()
        total_count = full_masks.numel()
        print(f"  Selected elements: {selected_count}/{total_count} ({100*selected_count/total_count:.2f}%)")
    
    # Store hybrid SSM states
    for layer_idx in range(num_layers):
        # For convolution cache: use fast cache (100%)
        conv_fast = cache_fast.key_value_memory_dict[layer_idx][0]
        
        # For SSM state: use hybrid (mixed)
        ssm_hybrid = ssm_hybrid_stack[layer_idx]
        
        cache_hybrid.key_value_memory_dict[layer_idx] = (conv_fast, ssm_hybrid)
    
    # ========== Prefill question with hybrid cache ==========
    if verbose:
        print(f"\nPrefilling question: '{question}'")
    tokens_question = tokenizer(part_D, return_tensors="pt")
    input_ids_question = tokens_question.input_ids.to(device)
    
    cache_hybrid, first_token = prefill_with_cache(model, input_ids_question, cache_hybrid, device)
    if verbose:
        print(f"  Cache offset after question: {cache_hybrid.seqlen_offset}")
    
    # ========== Generate answer ==========
    generated_tokens = [first_token.item()]
    current_token = first_token
    stop_strings = ["\nQ:", "\n\n"]
    
    for _ in range(max_new_tokens - 1):
        cache_hybrid, next_token = decode_with_cache(model, current_token, cache_hybrid, device)
        current_token = next_token
        generated_tokens.append(next_token.item())
        
        # Check for stop strings
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if any(stop_str in generated_text for stop_str in stop_strings):
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
            break
    
    return generated_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global angle (cosine similarity) hybrid cache experiment")
    parser.add_argument('top_k_percent', type=float, help='Percentage of dimensions with lowest cosine similarity to use slow cache (global bottom-k)')
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-2.8b', help='Model name or path')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_new_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--output_dir', type=str, default='./simple_inference/experiments', help='Output directory')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print(f"Loading model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Bottom-k percent: {args.top_k_percent}%\n")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    if args.test_mode:
        print("=" * 70)
        print("TEST MODE")
        print("=" * 70)
        
        question = "What is the capital of France?"
        doc1 = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
        doc2 = "Paris has a population of about 2 million people in the city proper."
        
        answer = inference_allangle(
            model=model,
            tokenizer=tokenizer,
            question=question,
            doc1=doc1,
            doc2=doc2,
            top_k_percent=args.top_k_percent,
            max_new_tokens=args.max_new_tokens,
            device=device,
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)
        print()
    else:
        print("=" * 70)
        print("GLOBAL ANGLE HYBRID CACHE")
        print("=" * 70)
        print(f"Dataset: {args.data_path}")
        print(f"Samples: {args.num_samples}")
        print(f"Bottom-k percent: {args.top_k_percent}%")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Seed: {args.seed}")
        print()
        
        dataset = HotpotQAIterator(args.data_path)
        sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
        
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%d%H%M%S")
        k_str = str(int(args.top_k_percent)) if args.top_k_percent == int(args.top_k_percent) else str(args.top_k_percent).replace('.', 'p')
        output_file = os.path.join(args.output_dir, f"allangle{k_str}_useful_{timestamp}.csv")
        
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
                    decoded = inference_allangle(
                        model=model,
                        tokenizer=tokenizer,
                        question=item.question,
                        doc1=doc1_with_title,
                        doc2=doc2_with_title,
                        top_k_percent=args.top_k_percent,
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
