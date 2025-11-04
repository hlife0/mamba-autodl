#!/usr/bin/env python3
"""
Collect cache differences between different prompts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from dataset.hotpot import HotpotQAIterator
from cache_extraction import extract_cache_diff

def build_prompt(question, doc1, doc2):
    fewshot = (
        "Q: Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    prompt_full = fewshot + f"Document 1: {doc1}\n\n" + f"Document 2: {doc2}\n\n" + f"Q: {question}\n\nA:"
    prompt_doc2 = fewshot + f"Document 2: {doc2}\n\n" + f"Q: {question}\n\nA:"
    return prompt_full, prompt_doc2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect cache differences")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-2.8b', help='Model name or path')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b', help='Tokenizer name')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./PCA_analysis/cache_diffs', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='full_vs_doc2', help='Experiment name')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print("=" * 70)
    print("CACHE DIFFERENCE COLLECTION")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.experiment_name}")
    print()
    
    # Load model and tokenizer ONCE
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Load dataset
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    output_subdir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"Output directory: {output_subdir}\n")
    print("Starting collection...")
    print()
    
    # Collect cache differences
    count = 0
    for idx, item in enumerate(tqdm(sample_dataset, desc="Processing")):
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        
        doc1_with_title = "[" + docs[0]['title'] + "] " + docs[0]['content']
        doc2_with_title = "[" + docs[1]['title'] + "] " + docs[1]['content']
        
        # Build two prompts to compare
        prompt1, prompt2 = build_prompt(item.question, doc1_with_title, doc2_with_title)
        
        try:
            # Extract cache difference (using pre-loaded model and tokenizer)
            cache_diff = extract_cache_diff(
                model=model,
                tokenizer=tokenizer,
                prompt1=prompt1,
                prompt2=prompt2,
                device=device,
                return_tensor=True
            )
            
            # Save the cache difference tensor
            output_file = os.path.join(output_subdir, f"diff_{count:04d}_id_{item.id}.pt")
            torch.save({
                'cache_diff': cache_diff,
                'item_id': item.id,
                'question': item.question,
                'answer': item.answer,
                'doc1_title': docs[0]['title'],
                'doc2_title': docs[1]['title'],
                'prompt1': prompt1,
                'prompt2': prompt2,
            }, output_file)
            
            count += 1
            
        except Exception as e:
            print(f"\nError processing item {item.id}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_subdir}")
    print(f"Collected: {count} cache differences")
    print(f"Shape per diff: [num_layers, d_inner, d_state]")
    print()

