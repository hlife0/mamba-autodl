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


def skip_last_layer_hybrid_logic(ssm_fast_stack, ssm_slow_stack, skip_last_layers=1):
    """
    Skip last N layers: use slow cache for layers [0, -skip_last_layers), fast cache for layers [-skip_last_layers, end].
    
    Args:
        ssm_fast_stack: Fast cache SSM states [num_layers, batch, d_inner, d_state]
        ssm_slow_stack: Slow cache SSM states [num_layers, batch, d_inner, d_state]
        skip_last_layers: Number of last layers to use fast cache (default: 1)
        
    Returns:
        ssm_hybrid_stack: Mixed SSM states [num_layers, batch, d_inner, d_state]
    """
    num_layers = ssm_fast_stack.shape[0]
    skip_last_layers = min(skip_last_layers, num_layers)  # Ensure skip_last_layers doesn't exceed total layers
    
    # Clone slow stack as base
    ssm_hybrid_stack = ssm_slow_stack.clone()
    
    # Replace last skip_last_layers with fast cache
    ssm_hybrid_stack[-skip_last_layers:] = ssm_fast_stack[-skip_last_layers:]
    
    return ssm_hybrid_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skip-last-layer hybrid cache experiment: last N layers use fast cache, rest use slow cache")
    parser.add_argument('--skip_last_layers', type=int, default=1, help='Number of last layers to use fast cache')
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
    print(f"Skip last layers: {args.skip_last_layers} (last {args.skip_last_layers} layers use fast cache)\n")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    print("=" * 70)
    print("SKIP LAST LAYER HYBRID CACHE")
    print("=" * 70)
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Skip last layers: {args.skip_last_layers} (last {args.skip_last_layers} layers use fast cache)")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Seed: {args.seed}")
    print()
    
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    output_file = os.path.join(args.output_dir, f"skiplastlayers{args.skip_last_layers}_{timestamp}.csv")
    
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
                # Create hybrid logic with skip_last_layers parameter
                def hybrid_logic_func(fast, slow):
                    return skip_last_layer_hybrid_logic(fast, slow, skip_last_layers=args.skip_last_layers)
                
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
