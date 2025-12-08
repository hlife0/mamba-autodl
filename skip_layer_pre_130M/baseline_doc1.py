#!/usr/bin/env python3
"""
Doc1 baseline: Doc1 + Few-shot + Question.
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
from dataset.hotpot import HotpotQAIterator
from skip_layer_pre_130M.utils import prefill_from_scratch, prefill_with_cache, decode_with_cache
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc1 baseline: doc1 + few-shot + question")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-130m', help='Model name or path')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_new_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--output_dir', type=str, default='./skip_layer_pre_130M/experiments', help='Output directory')
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
    
    print("=" * 70)
    print("DOC1 BASELINE")
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
    output_file = os.path.join(args.output_dir, f"baseline_doc1_{timestamp}.csv")
    
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
                # Doc1 baseline: doc1 -> few-shot + question
                doc1_prompt = f"Document 1: {doc1_with_title}\n\n"
                few_shot_prompt = (
                    "Q: Who is older, Alice or Bob?\n"
                    "A: Alice\n\n"
                    "Q: Are cats and dogs both mammals?\n"
                    "A: yes\n\n"
                    "Q: What color do red and blue make?\n"
                    "A: purple\n\n"
                )
                question_prompt = f"Q: {item.question}\n\nA:"
                
                # Prefill doc1
                tokens_doc = tokenizer(doc1_prompt, return_tensors="pt")
                input_ids_doc = tokens_doc.input_ids.to(device)
                cache, _ = prefill_from_scratch(model, input_ids_doc, device)
                
                # Prefill few-shot + question
                tokens_question = tokenizer(few_shot_prompt + question_prompt, return_tensors="pt")
                input_ids_question = tokens_question.input_ids.to(device)
                cache, first_token = prefill_with_cache(model, input_ids_question, cache, device)
                
                # Generate answer
                generated_tokens = [first_token.item()]
                current_token = first_token
                stop_strings = ["\nQ:", "\n\n"]
                
                for _ in range(args.max_new_tokens - 1):
                    cache, next_token = decode_with_cache(model, current_token, cache, device)
                    current_token = next_token
                    generated_tokens.append(next_token.item())
                    
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    if any(stop_str in generated_text for stop_str in stop_strings):
                        for stop_str in stop_strings:
                            if stop_str in generated_text:
                                generated_text = generated_text.split(stop_str)[0]
                        break
                
                decoded = generated_text.strip()
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
