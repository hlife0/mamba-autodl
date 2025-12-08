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
from mamba_ssm.utils.generation import InferenceParams
from dataset.hotpot import HotpotQAIterator
from skip_layer_pre_130M.utils import prefill_from_scratch, prefill_with_cache, decode_with_cache
from tqdm import tqdm


def inference_avgfirst(model, tokenizer, question, doc1, doc2, avg_layers=30, max_new_tokens=30, device="cuda"):
    """
    Hybrid strategy: first N layers use average of doc1 and doc2, remaining layers use slow path (doc1+doc2).
    """
    doc1_prompt = f"Document 1: {doc1}\n\n"
    doc2_prompt = f"Document 2: {doc2}\n\n"
    
    few_shot_prompt = (
        "Q: Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    
    question_prompt = f"Q: {question}\n\nA:"
    
    # ========== Path 1: Only doc1 ==========
    tokens_doc1 = tokenizer(doc1_prompt, return_tensors="pt")
    input_ids_doc1 = tokens_doc1.input_ids.to(device)
    cache_doc1, _ = prefill_from_scratch(model, input_ids_doc1, device)
    
    # ========== Path 2: Only doc2 ==========
    tokens_doc2 = tokenizer(doc2_prompt, return_tensors="pt")
    input_ids_doc2 = tokens_doc2.input_ids.to(device)
    cache_doc2, _ = prefill_from_scratch(model, input_ids_doc2, device)
    
    # ========== Path 3: doc1 + doc2 (slow path) ==========
    prompt_slow = doc1_prompt + doc2_prompt
    tokens_slow = tokenizer(prompt_slow, return_tensors="pt")
    input_ids_slow = tokens_slow.input_ids.to(device)
    cache_slow, _ = prefill_from_scratch(model, input_ids_slow, device)
    
    # ========== Create hybrid cache ==========
    num_layers = len(cache_doc1.key_value_memory_dict)
    
    # Initialize hybrid cache
    cache_hybrid = InferenceParams(
        max_seqlen=cache_slow.max_seqlen,
        max_batch_size=cache_slow.max_batch_size
    )
    cache_hybrid.seqlen_offset = cache_slow.seqlen_offset
    
    # Stack all conv states and SSM states
    conv_doc1_stack = torch.stack([cache_doc1.key_value_memory_dict[i][0] for i in range(num_layers)])
    conv_doc2_stack = torch.stack([cache_doc2.key_value_memory_dict[i][0] for i in range(num_layers)])
    ssm_doc1_stack = torch.stack([cache_doc1.key_value_memory_dict[i][1] for i in range(num_layers)])
    ssm_doc2_stack = torch.stack([cache_doc2.key_value_memory_dict[i][1] for i in range(num_layers)])
    
    conv_slow_stack = torch.stack([cache_slow.key_value_memory_dict[i][0] for i in range(num_layers)])
    ssm_slow_stack = torch.stack([cache_slow.key_value_memory_dict[i][1] for i in range(num_layers)])
    
    # First avg_layers: average doc1 and doc2, rest all use slow path
    conv_hybrid_stack = torch.zeros_like(conv_slow_stack)
    ssm_hybrid_stack = torch.zeros_like(ssm_slow_stack)
    
    # First N layers: average
    conv_hybrid_stack[:avg_layers] = (conv_doc1_stack[:avg_layers] + conv_doc2_stack[:avg_layers]) / 2.0
    ssm_hybrid_stack[:avg_layers] = (ssm_doc1_stack[:avg_layers] + ssm_doc2_stack[:avg_layers]) / 2.0
    
    # Remaining layers: slow path
    conv_hybrid_stack[avg_layers:] = conv_slow_stack[avg_layers:]
    ssm_hybrid_stack[avg_layers:] = ssm_slow_stack[avg_layers:]
    
    # Store results in cache_hybrid
    for layer_idx in range(num_layers):
        conv_hybrid = conv_hybrid_stack[layer_idx]
        ssm_hybrid = ssm_hybrid_stack[layer_idx]
        cache_hybrid.key_value_memory_dict[layer_idx] = (conv_hybrid, ssm_hybrid)
    
    # ========== Prefill few-shot + question with hybrid cache ==========
    tokens_question = tokenizer(few_shot_prompt + question_prompt, return_tensors="pt")
    input_ids_question = tokens_question.input_ids.to(device)
    cache_hybrid, first_token = prefill_with_cache(model, input_ids_question, cache_hybrid, device)
    
    # ========== Generate answer ==========
    generated_tokens = [first_token.item()]
    current_token = first_token
    stop_strings = ["\nQ:", "\n\n"]
    
    for _ in range(max_new_tokens - 1):
        cache_hybrid, next_token = decode_with_cache(model, current_token, cache_hybrid, device)
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
    parser = argparse.ArgumentParser(description="Hybrid experiment: first N layers use average, rest use slow path")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-130m', help='Model name or path')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--avg_layers', type=int, default=30, help='Number of layers to use average')
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
    print("AVERAGE FIRST N LAYERS + SLOW PATH")
    print("=" * 70)
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Strategy: First {args.avg_layers} layers use average(doc1, doc2), rest use slow path")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Seed: {args.seed}")
    print()
    
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    output_file = os.path.join(args.output_dir, f"avgfirst{args.avg_layers}_{timestamp}.csv")
    
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
                decoded = inference_avgfirst(
                    model=model,
                    tokenizer=tokenizer,
                    question=item.question,
                    doc1=doc1_with_title,
                    doc2=doc2_with_title,
                    avg_layers=args.avg_layers,
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
