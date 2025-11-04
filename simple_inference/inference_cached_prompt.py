#!/usr/bin/env python3
"""
Cached inference script for Mamba models with granular cache control.
This script demonstrates:
1. prefill_from_scratch: Create cache from prompt
2. prefill_with_cache: Continue prefilling with existing cache
3. decode_with_cache: Decode one token using cache

Usage: python inference_cached_prompt.py --prompt "Your prompt here"
"""

import argparse
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams


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


def simple_generate(model, input_ids, max_length, device):
    """
    Simple generation using model.generate() for comparison.
    
    Args:
        model: Mamba model
        input_ids: Input token IDs
        max_length: Maximum total length
        device: Device to run on
        
    Returns:
        generated_ids: All generated token IDs
    """
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_length,
            cg=True,
            return_dict_in_generate=True,
            enable_timing=False,
        )
    return output.sequences


def main():
    parser = argparse.ArgumentParser(description="Cached Mamba inference script")
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Input prompt string"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="state-spaces/mamba-2.8b",
        help="Model name or path (default: state-spaces/mamba-2.8b)"
    )
    parser.add_argument(
        "--split-pos",
        type=int,
        default=None,
        help="Position to split prompt (default: middle)"
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=10,
        help="Number of decoding steps (default: 10)"
    )
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Load model
    model = MambaLMHeadModel.from_pretrained(
        args.model_name,
        device=device,
        dtype=dtype
    )
    model.eval()
    
    # Tokenize full prompt
    tokens = tokenizer(args.prompt, return_tensors="pt")
    full_input_ids = tokens.input_ids.to(device=device)
    
    print(f"{'='*70}")
    print(f"Prompt: {args.prompt}")
    print(f"Total tokens: {full_input_ids.shape[1]}")
    
    # Determine split position
    total_len = full_input_ids.shape[1]
    split_pos = args.split_pos if args.split_pos is not None else total_len // 2
    split_pos = max(1, min(split_pos, total_len - 1))  # Ensure valid split
    
    # Split into A and B
    prompt_a = full_input_ids[:, :split_pos]
    prompt_b = full_input_ids[:, split_pos:]
    
    prompt_a_text = tokenizer.decode(prompt_a[0])
    prompt_b_text = tokenizer.decode(prompt_b[0])
    
    print(f"\nSplit at position {split_pos}:")
    print(f"  Part A ({prompt_a.shape[1]} tokens): '{prompt_a_text}'")
    print(f"  Part B ({prompt_b.shape[1]} tokens): '{prompt_b_text}'")
    print(f"{'='*70}\n")
    
    # ========== Test with cache-based approach ==========
    print("=" * 70)
    print("CACHE-BASED APPROACH")
    print("=" * 70)
    
    # Step 1: Prefill part A from scratch
    print(f"\n1. Prefill part A from scratch ({prompt_a.shape[1]} tokens)...")
    cache_a, token_after_a = prefill_from_scratch(model, prompt_a, device)
    print(f"   Cache offset after A: {cache_a.seqlen_offset}")
    print(f"   Next token after A: {tokenizer.decode(token_after_a[0])}")
    
    # Step 2: Prefill part B with cache from A
    print(f"\n2. Prefill part B with cache ({prompt_b.shape[1]} tokens)...")
    cache_ab, token_after_ab = prefill_with_cache(model, prompt_b, cache_a, device)
    print(f"   Cache offset after A+B: {cache_ab.seqlen_offset}")
    print(f"   Next token after A+B: {tokenizer.decode(token_after_ab[0])}")
    
    # Step 3: Decode additional tokens
    print(f"\n3. Decode {args.decode_steps} tokens with cache...")
    current_token = token_after_ab
    generated_tokens_cached = [token_after_ab]
    
    for i in range(args.decode_steps):
        cache_ab, next_token = decode_with_cache(model, current_token, cache_ab, device)
        generated_tokens_cached.append(next_token)
        current_token = next_token
    
    cached_result = torch.cat(generated_tokens_cached, dim=1)
    cached_text = tokenizer.decode(cached_result[0])
    
    print(f"\n   Generated text (cached): {cached_text}")
    print(f"   Final cache offset: {cache_ab.seqlen_offset}")
    
    # ========== Test with simple approach ==========
    print(f"\n{'='*70}")
    print("SIMPLE APPROACH (for comparison)")
    print("=" * 70)
    
    print(f"\nGenerating with simple approach (A+B as one prompt)...")
    simple_result = simple_generate(model, full_input_ids, args.decode_steps + 1, device)
    
    # Extract generated tokens (excluding prompt)
    simple_generated = simple_result[:, full_input_ids.shape[1]:]
    simple_text = tokenizer.decode(simple_generated[0])
    
    print(f"   Generated text (simple): {simple_text}")
    
    # ========== Compare results ==========
    print(f"\n{'='*70}")
    print("COMPARISON")
    print("=" * 70)
    
    # Compare token by token
    min_len = min(cached_result.shape[1], simple_generated.shape[1])
    matches = (cached_result[0, :min_len] == simple_generated[0, :min_len]).sum().item()
    
    print(f"\nCached approach tokens: {cached_result.shape[1]}")
    print(f"Simple approach tokens: {simple_generated.shape[1]}")
    print(f"Matching tokens: {matches}/{min_len}")
    
    if matches == min_len:
        print("\n✅ SUCCESS: Both approaches produce IDENTICAL results!")
    else:
        print(f"\n⚠️  MISMATCH: Results differ at position {matches}")
        print(f"\nCached tokens: {cached_result[0, :min_len].tolist()}")
        print(f"Simple tokens:  {simple_generated[0, :min_len].tolist()}")
        
        # Show where they differ
        for i in range(min_len):
            cached_tok = cached_result[0, i].item()
            simple_tok = simple_generated[0, i].item()
            if cached_tok != simple_tok:
                print(f"\n   Position {i}:")
                print(f"   Cached: {cached_tok} = '{tokenizer.decode([cached_tok])}'")
                print(f"   Simple: {simple_tok} = '{tokenizer.decode([simple_tok])}'")
                break
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
