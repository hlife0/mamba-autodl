#!/usr/bin/env python3
"""
Test utils.py with Mamba2 model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from skip_layer_pre2.utils import prefill_from_scratch, prefill_with_cache, decode_with_cache

def test_basic_inference():
    device = "cuda:2"
    dtype = torch.float16
    
    print("Loading Mamba2 model...")
    model = MambaLMHeadModel.from_pretrained('state-spaces/mamba2-2.7b', device=device, dtype=dtype)
    model.eval()
    print("✓ Model loaded")
    
    # Check all parameters are on correct device
    print(f"Checking model parameters...")
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name}: {param.device}")
    print()
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test 1: Simple prefill
    print("Test 1: Simple prefill from scratch")
    prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Input device: {input_ids.device}")
    
    cache, first_token = prefill_from_scratch(model, input_ids, device)
    print(f"  ✓ Prefill successful")
    print(f"  First token: {first_token.item()}")
    print(f"  Cache seqlen_offset: {cache.seqlen_offset}")
    
    # Check cache device
    for i in range(min(3, len(cache.key_value_memory_dict))):
        conv, ssm = cache.key_value_memory_dict[i]
        print(f"  Layer {i} conv device: {conv.device}, ssm device: {ssm.device}")
    
    # Test 2: Decode with cache
    print("\nTest 2: Decode with cache")
    for step in range(5):
        cache, next_token = decode_with_cache(model, first_token, cache, device)
        first_token = next_token
        print(f"  Step {step+1}: token={next_token.item()}, seqlen_offset={cache.seqlen_offset}")
    
    print("\n✓ All basic tests passed!")
    
    # Test 3: Prefill with cache
    print("\nTest 3: Prefill with existing cache")
    additional_prompt = " and the capital of Germany is"
    tokens2 = tokenizer(additional_prompt, return_tensors="pt")
    input_ids2 = tokens2.input_ids.to(device)
    print(f"  Additional input shape: {input_ids2.shape}")
    
    cache, next_token = prefill_with_cache(model, input_ids2, cache, device)
    print(f"  ✓ Prefill with cache successful")
    print(f"  Next token: {next_token.item()}")
    print(f"  Cache seqlen_offset: {cache.seqlen_offset}")
    
    print("\n✓✓✓ All tests passed successfully! ✓✓✓")

if __name__ == "__main__":
    test_basic_inference()
