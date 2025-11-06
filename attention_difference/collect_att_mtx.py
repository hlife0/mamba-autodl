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
from attention_extraction import (
    extract_abc, 
    extract_cummulated_Abar_right, 
    extract_alpha_last, 
    extract_beta_last
)

def build_prompt(question, doc1, doc2):
    fewshot = (
        "Q: Alice is twelve years old. Bob is eleven years old. Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    first_half = fewshot + f"Document 1: {doc1}\n\n"
    second_half = f"Document 2: {doc2}\n\n" + f"Q: {question}\n\nA:"
    return first_half, second_half, len(first_half), len(second_half)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect cache differences")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-2.8b', help='Model name or path')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b', help='Tokenizer name')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')   # GPU 0 only allowed for testing
    parser.add_argument('--num_samples', type=int, default=2, help='Number of samples') # for testing
    parser.add_argument('--num_partial_samples', type=int, default=1, help='Number of partial samples') # testing
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./attention_difference/att_mtx', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='alpha_beta_to_last', help='Experiment name')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print("=" * 70)
    print("ATTENTION TO LAST TOKEN COLLECTION")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Partial samples: {args.num_partial_samples}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.experiment_name}")
    print()
    print("Extraction strategy:")
    print("  - Alpha: all tokens → last token (full sequence)")
    print("  - Beta: last 4 tokens of first_half → last token")
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

    count_partial = 0
    partial_docs = []
    for item in sample_dataset:
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        partial_docs.append(("[" + docs[1]['title'] + "] " + docs[1]['content'], item.id))
        count_partial += 1
        if count_partial >= args.num_partial_samples:
            break

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    output_subdir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"Output directory: {output_subdir}\n")
    print("Starting collection...")
    print()

    count = 0
    total_pairs = len([item for item in sample_dataset if len(item.get_useful()) >= 2]) * len(partial_docs)
    pbar = tqdm(total=total_pairs, desc="Processing pairs")
    
    for idx, item in enumerate(sample_dataset):
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        
        doc1_with_title = "[" + docs[0]['title'] + "] " + docs[0]['content']
        doc1_id = item.id
        
        # Cartesian product: each doc1 pairs with all doc2
        for doc2_with_title, doc2_id in partial_docs:
            # Build prompt
            first_half, second_half, len_first_half, len_second_half = build_prompt(
                item.question, doc1_with_title, doc2_with_title
            )
            full_prompt = first_half + second_half
            
            try:
                # Tokenize to get sequence length and first_half token positions
                tokens = tokenizer(full_prompt, return_tensors="pt")
                seq_len = tokens['input_ids'].shape[1]
                
                # Calculate first_half token count
                first_half_tokens = tokenizer(first_half, return_tensors="pt")
                num_first_half_tokens = first_half_tokens['input_ids'].shape[1]
                
                # Last 4 tokens in first_half
                if num_first_half_tokens >= 4:
                    last_4_tokens_first_half = list(range(num_first_half_tokens - 4, num_first_half_tokens))
                else:
                    last_4_tokens_first_half = list(range(num_first_half_tokens))
                
                # All tokens (for alpha)
                all_tokens = list(range(seq_len))
                
                # Extract alpha and beta for all 64 layers
                num_layers = 64
                alpha_all_layers = []
                beta_all_layers = []
                
                for layer_idx in range(num_layers):
                    # Extract ABC for this layer
                    result = extract_abc(model, tokenizer, full_prompt, layer_idx=layer_idx, device=device)
                    
                    # Compute cumulated Abar (O(n) complexity)
                    Abar = extract_cummulated_Abar_right(result['discrete_A'])
                    
                    # Extract alpha: all tokens → last token [seqlen, d_inner]
                    alpha = extract_alpha_last(Abar, result['discrete_B'], result['C'], all_tokens)
                    alpha_all_layers.append(alpha.cpu())  # Move to CPU immediately
                    
                    # Extract beta: last 4 tokens of first_half → last token [4, d_inner, d_state]
                    beta = extract_beta_last(Abar, result['C'], last_4_tokens_first_half)
                    beta_all_layers.append(beta.cpu())  # Move to CPU immediately
                    
                    # Clear GPU memory for this layer
                    del result, Abar, alpha, beta
                    torch.cuda.empty_cache()
                
                # Stack all layers
                alpha_stacked = torch.stack(alpha_all_layers, dim=0)  # [64, seqlen, d_inner]
                beta_stacked = torch.stack(beta_all_layers, dim=0)    # [64, 4, d_inner, d_state]
                
                # Clear intermediate lists
                del alpha_all_layers, beta_all_layers
                
                # Save the attention matrix tensors
                output_file = os.path.join(output_subdir, f"att_mtx_{count:04d}.pt")
                torch.save({
                    'alpha': alpha_stacked,  # [64, seqlen, d_inner] - all tokens → last token
                    'beta': beta_stacked,    # [64, 4, d_inner, d_state] - last 4 of first_half → last token
                    'last_4_tokens_first_half': last_4_tokens_first_half,
                    'len_first_half': len_first_half,
                    'len_second_half': len_second_half,
                    'num_first_half_tokens': num_first_half_tokens,
                    'seq_len': seq_len,
                    'id_doc1_question': doc1_id,
                    'id_doc2_question': doc2_id,
                    'question': item.question,
                }, output_file)
                
                # Clear saved data
                del alpha_stacked, beta_stacked
                torch.cuda.empty_cache()
                
                count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing pair ({doc1_id}, {doc2_id}): {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
                continue
    
    pbar.close()
    
    print("\n" + "=" * 70)
    print("Collection complete!")
    print(f"Total pairs processed: {count}")
    print(f"Output directory: {output_subdir}")
    print("=" * 70)