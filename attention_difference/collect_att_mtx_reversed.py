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
    extract_abc_all_layers_fast,
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
    parser = argparse.ArgumentParser(description="Collect attention statistics (REVERSED: partial as doc1, all as doc2)")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-2.8b', help='Model name or path')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b', help='Tokenizer name')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of all samples (doc2)')
    parser.add_argument('--num_partial_samples', type=int, default=100, help='Number of partial samples (doc1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./attention_difference/experiments', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='alpha_beta_stats_reversed', help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for parallel processing (default: 1)')
    parser.add_argument('--max_doc2_tokens', type=int, default=50, help='Maximum tokens for doc2 (truncation)')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print("=" * 70)
    print("ATTENTION STATISTICS COLLECTION (REVERSED - MEMORY-EFFICIENT)")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"Partial samples (doc1): {args.num_partial_samples}")
    print(f"All samples (doc2): {args.num_samples}")
    print(f"Doc2 max tokens: {args.max_doc2_tokens} (truncated for consistency)")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.experiment_name}")
    print()
    print("Strategy (REVERSED):")
    print("  - For each doc1 (partial):")
    print("    - Pair with all doc2 samples (truncated to 50 tokens)")
    print("    - Extract: alpha (doc2 tokens → doc2 last)")
    print("    - Extract: beta (doc1 last 4 → doc2 last)")
    print("    - Stream computation (Welford's algorithm)")
    print("    - Save statistics only")
    print("  - Memory: O(1) - only statistics in memory")
    print("  - Disk: ~1000x reduction")
    print()
    
    # Load model and tokenizer ONCE
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Load dataset
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    # Collect all doc2 samples (will be truncated)
    all_doc2_samples = []
    for item in sample_dataset:
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        
        # Truncate doc2 to max_doc2_tokens
        doc_text = "[" + docs[1]['title'] + "] " + docs[1]['content']
        # Tokenize and truncate
        doc_tokens = tokenizer(doc_text, return_tensors="pt", truncation=True, max_length=args.max_doc2_tokens)
        doc_truncated = tokenizer.decode(doc_tokens['input_ids'][0], skip_special_tokens=True)
        
        all_doc2_samples.append({
            'doc_with_title': doc_truncated,
            'doc_id': item.id,
            'question': item.question,
        })
        if len(all_doc2_samples) >= args.num_samples:
            break
    
    # Collect partial doc1 samples
    count_partial = 0
    partial_doc1_samples = []
    for item in sample_dataset:
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        partial_doc1_samples.append({
            'doc_with_title': "[" + docs[0]['title'] + "] " + docs[0]['content'],
            'doc_id': item.id,
        })
        count_partial += 1
        if count_partial >= args.num_partial_samples:
            break

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%H%M%S")
    output_subdir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"Output directory: {output_subdir}\n")
    print(f"Total doc1 (partial): {len(partial_doc1_samples)}")
    print(f"Total doc2 (all, truncated): {len(all_doc2_samples)}")
    print(f"Total pairs: {len(partial_doc1_samples) * len(all_doc2_samples)}\n")
    print("Starting collection...")
    print()

    # OUTER LOOP: iterate over doc1 (partial)
    pbar_doc1 = tqdm(total=len(partial_doc1_samples), desc="Processing doc1 (partial)")
    
    for doc1_sample in partial_doc1_samples:
        doc1_with_title = doc1_sample['doc_with_title']
        doc1_id = doc1_sample['doc_id']
        
        # Clear GPU cache before processing new doc1
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Build all pairs for this doc1
        all_pairs_for_doc1 = []
        for doc2_sample in all_doc2_samples:
            first_half, second_half, len_first_half, len_second_half = build_prompt(
                doc2_sample['question'], 
                doc1_with_title,  # doc1 is partial
                doc2_sample['doc_with_title']  # doc2 is all (truncated)
            )
            full_prompt = first_half + second_half
            
            all_pairs_for_doc1.append({
                'full_prompt': full_prompt,
                'first_half': first_half,
                'doc2_id': doc2_sample['doc_id'],
                'doc2_with_title': doc2_sample['doc_with_title'],  # Store for tokenization
                'question': doc2_sample['question'],
                'len_first_half': len_first_half,
                'len_second_half': len_second_half,
            })
        
        # Online computation of mean and variance (Welford's algorithm)
        count = 0
        alpha_mean = None
        alpha_M2 = None
        beta_mean = None
        beta_M2 = None
        max_doc2_seqlen = 0  # Track max doc2 sequence length
        
        # INNER LOOP: process all doc2 in batches
        total_doc2 = len(all_pairs_for_doc1)
        pbar_doc2 = tqdm(total=total_doc2, desc=f"  doc1={doc1_id[:8]}", leave=False)
        
        for batch_start in range(0, total_doc2, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_doc2)
            batch_pairs = all_pairs_for_doc1[batch_start:batch_end]
            actual_batch_size = len(batch_pairs)
            
            try:
                # Tokenize all prompts in batch (with padding)
                prompts = [pair['full_prompt'] for pair in batch_pairs]
                first_halves = [pair['first_half'] for pair in batch_pairs]
                
                # Get tokenized sequences
                batch_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
                input_ids_batch = batch_tokens['input_ids'].to(device)
                attention_mask_batch = batch_tokens['attention_mask'].to(device)
                
                # Get sequence lengths for each sample (before padding)
                seq_lens = attention_mask_batch.sum(dim=1).cpu().tolist()
                
                # Get first_half token counts
                first_half_tokens_batch = tokenizer(first_halves, return_tensors="pt", padding=True)
                num_first_half_tokens_list = first_half_tokens_batch['attention_mask'].sum(dim=1).tolist()
                
                # Extract ABC for ALL 64 layers in a SINGLE forward pass for the ENTIRE BATCH
                extracted_all_layers = extract_abc_all_layers_fast(model, tokenizer, prompts, device=device, keep_on_gpu=True)
                
                # Process each sample in the batch
                for sample_idx in range(actual_batch_size):
                    pair = batch_pairs[sample_idx]
                    seq_len = seq_lens[sample_idx]
                    num_first_half_tokens = num_first_half_tokens_list[sample_idx]
                    
                    # Calculate doc2 token range
                    # doc2 is in second_half, but we ONLY want the doc2 content itself (not question/answer)
                    # The second_half format is: "Document 2: {doc2}\n\nQ: {question}\n\nA:"
                    # We tokenize JUST the doc2 part (with exact formatting) to get its range
                    
                    # Get the doc2 content from the batch_pair (with exact formatting as in prompt)
                    doc2_part_only = f"Document 2: {pair['doc2_with_title']}\n\n"
                    doc2_tokens_only = tokenizer(doc2_part_only, return_tensors="pt")
                    doc2_len_only = doc2_tokens_only['input_ids'].shape[1]
                    
                    # doc2 starts right after first_half
                    doc2_start = num_first_half_tokens
                    doc2_end = doc2_start + doc2_len_only
                    doc2_len = doc2_len_only
                    
                    # Last 4 tokens in first_half (doc1)
                    if num_first_half_tokens >= 4:
                        last_4_tokens_first_half = list(range(num_first_half_tokens - 4, num_first_half_tokens))
                    else:
                        last_4_tokens_first_half = list(range(num_first_half_tokens))
                    
                    # All tokens in doc2 (for alpha)
                    doc2_tokens = list(range(doc2_start, doc2_end))
                    
                    # Process each layer for this sample
                    alpha_all_layers = []
                    beta_all_layers = []
                    
                    for layer_data in extracted_all_layers:
                        # Extract this sample's data from the batch
                        discrete_A = layer_data['discrete_A'][sample_idx:sample_idx+1, :, :seq_len, :]
                        discrete_B = layer_data['discrete_B'][sample_idx:sample_idx+1, :, :seq_len, :]
                        C = layer_data['C'][sample_idx:sample_idx+1, :, :seq_len]
                        
                        # Compute cumulated Abar
                        Abar = extract_cummulated_Abar_right(discrete_A)
                        
                        # Extract alpha: doc2 tokens → doc2 last token [doc2_len, d_inner]
                        alpha = extract_alpha_last(Abar, discrete_B, C, doc2_tokens)
                        alpha_all_layers.append(alpha.cpu())
                        
                        # Extract beta: doc1 last 4 tokens → doc2 last token [4, d_inner, d_state]
                        beta = extract_beta_last(Abar, C, last_4_tokens_first_half)
                        beta_all_layers.append(beta.cpu())
                        
                        # Clear GPU memory for this layer
                        del discrete_A, discrete_B, C, Abar, alpha, beta
                    
                    # Stack all layers
                    alpha_stacked = torch.stack(alpha_all_layers, dim=0).float()  # [64, doc2_len, d_inner]
                    beta_stacked = torch.stack(beta_all_layers, dim=0).float()    # [64, 4, d_inner, d_state]
                    del alpha_all_layers, beta_all_layers
                    
                    # Track max doc2 sequence length
                    current_doc2_seqlen = alpha_stacked.shape[1]
                    if current_doc2_seqlen > max_doc2_seqlen:
                        # Expand previous statistics if needed
                        if alpha_mean is not None:
                            old_seqlen = alpha_mean.shape[1]
                            if current_doc2_seqlen > old_seqlen:
                                pad_size = current_doc2_seqlen - old_seqlen
                                alpha_mean = torch.cat([alpha_mean, torch.zeros(64, pad_size, 5120)], dim=1)
                                alpha_M2 = torch.cat([alpha_M2, torch.zeros(64, pad_size, 5120)], dim=1)
                        max_doc2_seqlen = current_doc2_seqlen
                    
                    # Pad current alpha if needed
                    if current_doc2_seqlen < max_doc2_seqlen:
                        pad_size = max_doc2_seqlen - current_doc2_seqlen
                        alpha_stacked = torch.cat([alpha_stacked, torch.zeros(64, pad_size, 5120)], dim=1)
                    
                    # Welford's online algorithm
                    count += 1
                    
                    if alpha_mean is None:
                        # First sample
                        alpha_mean = alpha_stacked.clone()
                        alpha_M2 = torch.zeros_like(alpha_stacked)
                        beta_mean = beta_stacked.clone()
                        beta_M2 = torch.zeros_like(beta_stacked)
                        del alpha_stacked, beta_stacked
                    else:
                        # Update alpha
                        delta = alpha_stacked - alpha_mean
                        alpha_mean += delta / count
                        delta2 = alpha_stacked - alpha_mean
                        alpha_M2 += delta * delta2
                        
                        # Update beta
                        delta_beta = beta_stacked - beta_mean
                        beta_mean += delta_beta / count
                        delta2_beta = beta_stacked - beta_mean
                        beta_M2 += delta_beta * delta2_beta
                        
                        del alpha_stacked, beta_stacked, delta, delta2, delta_beta, delta2_beta
                    
                    pbar_doc2.update(1)
                
                # Clear batch data
                del extracted_all_layers, input_ids_batch, attention_mask_batch, prompts, first_halves
                del batch_tokens, seq_lens, first_half_tokens_batch, num_first_half_tokens_list
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            except Exception as e:
                print(f"\nError processing batch starting at {batch_start}: {e}")
                import traceback
                traceback.print_exc()
                pbar_doc2.update(actual_batch_size)
                continue
        
        pbar_doc2.close()
        
        # Finalize statistics computation
        try:
            print(f"\n  Finalizing statistics for doc1={doc1_id[:8]}...")
            
            if count == 0:
                print(f"  ✗ No data collected for doc1={doc1_id}, skipping...")
                continue
            
            # Compute variance from M2
            if count > 1:
                alpha_var = alpha_M2 / (count - 1)
                beta_var = beta_M2 / (count - 1)
            else:
                alpha_var = torch.zeros_like(alpha_mean)
                beta_var = torch.zeros_like(beta_mean)
            
            # Save statistics
            output_file = os.path.join(output_subdir, f"{doc1_id}.pt")
            torch.save({
                'alpha_mean': alpha_mean,
                'alpha_var': alpha_var,
                'beta_mean': beta_mean,
                'beta_var': beta_var,
                'doc1_id': doc1_id,
                'num_doc2_samples': count,
                'max_doc2_seqlen': max_doc2_seqlen,
            }, output_file)
            
            print(f"  ✓ Saved statistics to {doc1_id}.pt (n={count}, max_doc2_seqlen={max_doc2_seqlen})")
            
            # Clear memory
            del alpha_mean, alpha_M2, alpha_var, beta_mean, beta_M2, beta_var
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError finalizing statistics for doc1={doc1_id}: {e}")
            import traceback
            traceback.print_exc()
        
        pbar_doc1.update(1)
    
    pbar_doc1.close()
    
    print("\n" + "=" * 70)
    print("Collection complete!")
    print(f"Total doc1 processed: {len(partial_doc1_samples)}")
    print(f"Output directory: {output_subdir}")
    print("=" * 70)

