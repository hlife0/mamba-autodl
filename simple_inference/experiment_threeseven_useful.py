#!/usr/bin/env python3
"""
ThreeSeven cache baseline: Weighted average of caches (30% doc1, 70% doc2).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime
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


def prefill_with_cache(model, input_ids, previous_cache, device):
    additional_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    logits = None
    for i in range(additional_len):
        current_token = input_ids[:, i:i+1]
        position_ids = torch.full((batch_size, 1), previous_cache.seqlen_offset, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(current_token, position_ids=position_ids, inference_params=previous_cache, num_last_tokens=1).logits
        previous_cache.seqlen_offset += 1
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return previous_cache, next_token_id


def decode_with_cache(model, token_id, cache, device):
    batch_size = token_id.shape[0]
    position_ids = torch.full((batch_size, 1), cache.seqlen_offset, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(token_id, position_ids=position_ids, inference_params=cache, num_last_tokens=1).logits
    cache.seqlen_offset += 1
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return cache, next_token_id


def inference_threeseven(model, tokenizer, question, doc1, doc2, max_new_tokens=30, device="cuda", verbose=False):
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
        print("THREESEVEN CACHE INFERENCE (30% doc1, 70% doc2)")
        print("=" * 70)
    
    # ========== Path 1: fewshot + doc1 ==========
    if verbose:
        print("Path 1: Few-shot + Doc1")
    prompt_1 = part_A + part_B
    tokens_1 = tokenizer(prompt_1, return_tensors="pt")
    input_ids_1 = tokens_1.input_ids.to(device)
    
    cache_1, _ = prefill_from_scratch(model, input_ids_1, device)
    if verbose:
        print(f"  Tokens processed: {cache_1.seqlen_offset}")
    
    # ========== Path 2: fewshot + doc2 ==========
    if verbose:
        print("\nPath 2: Few-shot + Doc2")
    prompt_2 = part_A + part_C
    tokens_2 = tokenizer(prompt_2, return_tensors="pt")
    input_ids_2 = tokens_2.input_ids.to(device)
    
    cache_2, _ = prefill_from_scratch(model, input_ids_2, device)
    if verbose:
        print(f"  Tokens processed: {cache_2.seqlen_offset}")
    
    # ========== Weighted average: 30% doc1, 70% doc2 ==========
    if verbose:
        print("\nWeighted averaging caches (30% doc1, 70% doc2)...")
    num_layers = len(cache_1.key_value_memory_dict)
    
    # Initialize weighted cache
    cache_weighted = InferenceParams(
        max_seqlen=cache_1.max_seqlen,
        max_batch_size=cache_1.max_batch_size
    )
    cache_weighted.seqlen_offset = cache_1.seqlen_offset  # Use same offset (they should be equal)
    
    # ========== Vectorized weighted cache averaging ==========
    # Stack all conv states and SSM states
    # Shape: [num_layers, batch, d_inner, ...]
    conv_1_stack = torch.stack([cache_1.key_value_memory_dict[i][0] for i in range(num_layers)])
    conv_2_stack = torch.stack([cache_2.key_value_memory_dict[i][0] for i in range(num_layers)])
    ssm_1_stack = torch.stack([cache_1.key_value_memory_dict[i][1] for i in range(num_layers)])
    ssm_2_stack = torch.stack([cache_2.key_value_memory_dict[i][1] for i in range(num_layers)])
    
    # Weighted average: 0.3 * cache_1 + 0.7 * cache_2
    conv_weighted_stack = 0.3 * conv_1_stack + 0.7 * conv_2_stack
    ssm_weighted_stack = 0.3 * ssm_1_stack + 0.7 * ssm_2_stack
    
    # Store results in cache_weighted
    for layer_idx in range(num_layers):
        conv_weighted = conv_weighted_stack[layer_idx]
        ssm_weighted = ssm_weighted_stack[layer_idx]
        cache_weighted.key_value_memory_dict[layer_idx] = (conv_weighted, ssm_weighted)
    
    if verbose:
        print(f"  Weighted averaged {num_layers} layers")
    
    # ========== Prefill question with weighted cache ==========
    if verbose:
        print(f"\nPrefilling question: '{question}'")
    tokens_question = tokenizer(part_D, return_tensors="pt")
    input_ids_question = tokens_question.input_ids.to(device)
    
    cache_weighted, first_token = prefill_with_cache(model, input_ids_question, cache_weighted, device)
    if verbose:
        print(f"  Cache offset after question: {cache_weighted.seqlen_offset}")
    
    # ========== Generate answer ==========
    generated_tokens = [first_token.item()]
    current_token = first_token
    stop_strings = ["\nQ:", "\n\n"]
    
    for _ in range(max_new_tokens - 1):
        cache_weighted, next_token = decode_with_cache(model, current_token, cache_weighted, device)
        current_token = next_token
        generated_tokens.append(next_token.item())
        
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if any(stop_str in generated_text for stop_str in stop_strings):
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
            break
    
    return generated_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThreeSeven cache baseline experiment (30% doc1, 70% doc2)")
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
    print(f"Device: {device}\n")
    
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
        
        answer = inference_threeseven(
            model=model,
            tokenizer=tokenizer,
            question=question,
            doc1=doc1,
            doc2=doc2,
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
        print("THREESEVEN CACHE BASELINE (30% doc1, 70% doc2)")
        print("=" * 70)
        print(f"Dataset: {args.data_path}")
        print(f"Samples: {args.num_samples}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Seed: {args.seed}")
        print()
        
        dataset = HotpotQAIterator(args.data_path)
        sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
        
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%d%H%M%S")
        output_file = os.path.join(args.output_dir, f"threeseven_useful_{timestamp}.csv")
        
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
                    decoded = inference_threeseven(
                        model=model,
                        tokenizer=tokenizer,
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
