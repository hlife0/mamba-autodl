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
    parser = argparse.ArgumentParser(description="Collect cache differences")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba-130m', help='Model name or path')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b', help='Tokenizer name')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--num_partial_samples', type=int, default=100, help='Number of partial samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./attention_difference/experiments', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='alpha_beta_stats-130M', help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for parallel processing (default: 2, max recommended: 2)')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print("=" * 70)
    print("ATTENTION STATISTICS COLLECTION (MEMORY-EFFICIENT)")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"All samples (doc1): {args.num_samples}")
    print(f"Partial samples (doc2): {args.num_partial_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.experiment_name}")
    print()
    print("Strategy:")
    print("  - For each doc2 (partial):")
    print("    - Pair with all doc1 samples")
    print("    - Stream alpha/beta computation (Welford's algorithm)")
    print("    - Calculate mean & variance online (no storage of raw data)")
    print("    - Save statistics only")
    print("  - Memory: O(1) - only statistics in memory"  )
    print("  - Disk: ~1000x reduction")
    print()
    
    # Load model and tokenizer ONCE
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Extract model dimensions
    n_layer = len(model.backbone.layers)
    d_inner = model.backbone.layers[0].mixer.d_inner
    d_state = model.backbone.layers[0].mixer.d_state
    print(f"\nModel Architecture:")
    print(f"  Number of layers: {n_layer}")
    print(f"  d_inner: {d_inner}")
    print(f"  d_state: {d_state}")
    print()
    
    # Load dataset
    dataset = HotpotQAIterator(args.data_path)
    sample_dataset = dataset.random_choose(args.num_samples, seed=args.seed)
    
    # Collect all doc1 samples
    all_doc1_samples = []
    for item in sample_dataset:
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        all_doc1_samples.append({
            'doc_with_title': "[" + docs[0]['title'] + "] " + docs[0]['content'],
            'doc_id': item.id,
            'question': item.question,
        })
        if len(all_doc1_samples) >= args.num_samples:
            break
    
    # Collect partial doc2 samples
    count_partial = 0
    partial_doc2_samples = []
    for item in sample_dataset:
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        partial_doc2_samples.append({
            'doc_with_title': "[" + docs[1]['title'] + "] " + docs[1]['content'],
            'doc_id': item.id,
            'question': item.question
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
    print(f"Total doc1 (all): {len(all_doc1_samples)}")
    print(f"Total doc2 (partial): {len(partial_doc2_samples)}")
    print(f"Total pairs: {len(all_doc1_samples) * len(partial_doc2_samples)}\n")
    print("Starting collection...")
    print()

    # OUTER LOOP: iterate over doc2 (partial)
    pbar_doc2 = tqdm(total=len(partial_doc2_samples), desc="Processing doc2 (partial)")
    
    for doc2_sample in partial_doc2_samples:
        doc2_id = doc2_sample['doc_id']
        
        # Clear GPU cache before processing new doc2
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Build all pairs for this doc2
        all_pairs_for_doc2 = []
        for doc1_sample in all_doc1_samples:
            first_half, second_half, len_first_half, len_second_half = build_prompt(
                doc2_sample['question'], 
                doc1_sample['doc_with_title'], 
                doc2_sample['doc_with_title']
            )
            full_prompt = first_half + second_half
            
            all_pairs_for_doc2.append({
                'full_prompt': full_prompt,
                'first_half': first_half,
                'doc1_id': doc1_sample['doc_id'],
                'question': doc2_sample['question']
            })
        
        # Online computation of mean and variance (Welford's algorithm)
        # Don't store all samples - compute statistics on-the-fly
        count = 0
        alpha_mean = None
        alpha_M2 = None  # For variance computation
        beta_mean = None
        beta_M2 = None
        max_seqlen = 0
        
        # INNER LOOP: process all doc1 in batches
        total_doc1 = len(all_pairs_for_doc2)
        pbar_doc1 = tqdm(total=total_doc1, desc=f"  doc2={doc2_id[:8]}", leave=False)
        
        for batch_start in range(0, total_doc1, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_doc1)
            batch_pairs = all_pairs_for_doc2[batch_start:batch_end]
            actual_batch_size = len(batch_pairs)
            
            try:
                # Tokenize all prompts in batch (with padding)
                prompts = [pair['full_prompt'] for pair in batch_pairs]
                first_halves = [pair['first_half'] for pair in batch_pairs]
                questions = [f"Q: {pair['question']}\n\nA:" for pair in batch_pairs]
                
                # Get tokenized sequences
                batch_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
                input_ids_batch = batch_tokens['input_ids'].to(device)
                attention_mask_batch = batch_tokens['attention_mask'].to(device)
                
                # Get sequence lengths for each sample (before padding)
                seq_lens = attention_mask_batch.sum(dim=1).cpu().tolist()
                
                # Get first_half token counts
                first_half_tokens_batch = tokenizer(first_halves, return_tensors="pt", padding=True)
                num_first_half_tokens_list = first_half_tokens_batch['attention_mask'].sum(dim=1).tolist()
                
                # Extract ABC for ALL layers in a SINGLE forward pass for the ENTIRE BATCH
                extracted_all_layers = extract_abc_all_layers_fast(model, tokenizer, prompts, device=device, keep_on_gpu=True)
                
                # Process each sample in the batch
                for sample_idx in range(actual_batch_size):
                    pair = batch_pairs[sample_idx]
                    seq_len = seq_lens[sample_idx]
                    num_first_half_tokens = num_first_half_tokens_list[sample_idx]
                    
                    # Last 4 tokens in first_half
                    if num_first_half_tokens >= 4:
                        last_4_tokens_first_half = list(range(num_first_half_tokens - 4, num_first_half_tokens))
                    else:
                        last_4_tokens_first_half = list(range(num_first_half_tokens))
                    
                    # All tokens (for alpha)
                    all_tokens_of_second_half = list(range(num_first_half_tokens, seq_len))
                    
                    # Process each layer for this sample
                    alpha_all_layers = []
                    beta_all_layers = []
                    
                    for layer_data in extracted_all_layers:
                        # Extract this sample's data from the batch
                        # Data shape: discrete_A/B: [batch, d_inner, max_seqlen, d_state], C: [batch, d_state, max_seqlen]
                        discrete_A = layer_data['discrete_A'][sample_idx:sample_idx+1, :, :seq_len, :]
                        discrete_B = layer_data['discrete_B'][sample_idx:sample_idx+1, :, :seq_len, :]
                        C = layer_data['C'][sample_idx:sample_idx+1, :, :seq_len]
                        
                        # Compute cumulated Abar (O(n) complexity)
                        Abar = extract_cummulated_Abar_right(discrete_A)
                        
                        # Extract alpha: all tokens → last token [seqlen, d_inner]
                        alpha = extract_alpha_last(Abar, discrete_B, C, all_tokens_of_second_half)
                        alpha_all_layers.append(alpha.cpu())
                        
                        # Extract beta: last 4 tokens of first_half → last token
                        beta = extract_beta_last(Abar, C, last_4_tokens_first_half)
                        beta_all_layers.append(beta.cpu())
                        
                        # Clear GPU memory for this layer
                        del discrete_A, discrete_B, C, Abar, alpha, beta
                    
                    # Stack all layers
                    alpha_stacked = torch.stack(alpha_all_layers, dim=0).float()  # [n_layer, seqlen, d_inner]
                    beta_stacked = torch.stack(beta_all_layers, dim=0).float()    # [n_layer, 4, d_inner, d_state]
                    del alpha_all_layers, beta_all_layers
                    
                    # Update max_seqlen
                    current_seqlen = alpha_stacked.shape[1]
                    if current_seqlen > max_seqlen:
                        # Need to expand previous statistics
                        if alpha_mean is not None:
                            old_seqlen = alpha_mean.shape[1]
                            if current_seqlen > old_seqlen:
                                # Pad previous statistics
                                pad_size = current_seqlen - old_seqlen
                                alpha_mean = torch.cat([alpha_mean, torch.zeros(n_layer, pad_size, d_inner)], dim=1)
                                alpha_M2 = torch.cat([alpha_M2, torch.zeros(n_layer, pad_size, d_inner)], dim=1)
                        max_seqlen = current_seqlen
                    
                    # Pad current alpha if needed
                    if current_seqlen < max_seqlen:
                        pad_size = max_seqlen - current_seqlen
                        alpha_stacked = torch.cat([alpha_stacked, torch.zeros(n_layer, pad_size, d_inner)], dim=1)
                    
                    # Welford's online algorithm for mean and variance
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
                    
                    pbar_doc1.update(1)
                
                # Clear batch data
                del extracted_all_layers, input_ids_batch, attention_mask_batch, prompts, first_halves
                del batch_tokens, seq_lens, first_half_tokens_batch, num_first_half_tokens_list
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            except Exception as e:
                print(f"\nError processing batch starting at {batch_start}: {e}")
                import traceback
                traceback.print_exc()
                pbar_doc1.update(actual_batch_size)
                continue
        
        pbar_doc1.close()
        
        # Finalize statistics computation
        try:
            print(f"\n  Finalizing statistics for doc2={doc2_id[:8]}...")
            
            if count == 0:
                print(f"  ✗ No data collected for doc2={doc2_id}, skipping...")
                continue
            
            # Compute variance from M2 (Welford's algorithm)
            if count > 1:
                alpha_var = alpha_M2 / (count - 1)  # Sample variance
                beta_var = beta_M2 / (count - 1)
            else:
                alpha_var = torch.zeros_like(alpha_mean)
                beta_var = torch.zeros_like(beta_mean)
            
            # Save statistics
            output_file = os.path.join(output_subdir, f"{doc2_id}.pt")
            torch.save({
                'alpha_mean': alpha_mean,
                'alpha_var': alpha_var,
                'beta_mean': beta_mean,
                'beta_var': beta_var,
                'doc2_id': doc2_id,
                'num_doc1_samples': count,
                'max_seqlen': max_seqlen,
            }, output_file)
            
            print(f"  ✓ Saved statistics to {doc2_id}.pt (n={count}, max_seqlen={max_seqlen})")
            
            # Clear memory
            del alpha_mean, alpha_M2, alpha_var, beta_mean, beta_M2, beta_var
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError finalizing statistics for doc2={doc2_id}: {e}")
            import traceback
            traceback.print_exc()
        
        pbar_doc2.update(1)
    
    pbar_doc2.close()
    
    print("\n" + "=" * 70)
    print("Collection complete!")
    print(f"Total doc2 processed: {len(partial_doc2_samples)}")
    print(f"Output directory: {output_subdir}")
    print("=" * 70)