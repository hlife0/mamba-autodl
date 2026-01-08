"""
Collect Alpha Statistics for Mamba2

Collects attention statistics (alpha matrices) from Mamba2 models on HotpotQA dataset.
Uses memory-efficient online computation (Welford's algorithm) to calculate mean and variance.

Run from project root: python attention_difference2/collect_alpha.py
"""
import sys
import os
import argparse
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Add parent directory to path for dataset import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.hotpot import HotpotQAIterator

# Import from current directory
from fast_extraction import extract_alpha_all


def build_prompt(question, doc1, doc2):
    """Build few-shot prompt with two documents
    
    Returns:
        first_half: fewshot + doc1
        second_half: doc2 + question
        doc2_start_marker: string marking where doc2 starts
        doc2_end_marker: string marking where doc2 ends
    """
    fewshot = (
        "Q: Alice is twelve years old. Bob is eleven years old. Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    first_half = fewshot + f"Document 1: {doc1}\n\n"
    doc2_content = f"Document 2: {doc2}"
    second_half = doc2_content + f"\n\nQ: {question}\n\nA:"
    
    return first_half, second_half, doc2_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect alpha statistics for Mamba2")
    parser.add_argument('--model_path', type=str, default='state-spaces/mamba2-1.3b', 
                        help='Model name or path')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b', 
                        help='Tokenizer name')
    parser.add_argument('--data_path', type=str, default='./dataset/HotpotQA/hotpot_train_v1.1.json', 
                        help='Path to HotpotQA dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, 
                        help='Number of samples for doc1 (all)')
    parser.add_argument('--num_partial_samples', type=int, default=100, 
                        help='Number of samples for doc2 (partial)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./attention_difference2/experiments', 
                        help='Output directory (relative to project root)')
    parser.add_argument('--experiment_name', type=str, default='alpha_stats_mamba2', 
                        help='Experiment name')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print("=" * 70)
    print("ALPHA STATISTICS COLLECTION FOR MAMBA2 (MEMORY-EFFICIENT)")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"All samples (doc1): {args.num_samples}")
    print(f"Partial samples (doc2): {args.num_partial_samples}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.experiment_name}")
    print()
    print("Strategy:")
    print("  - For each doc2 (partial):")
    print("    - Pair with all doc1 samples")
    print("    - Stream alpha computation (Welford's algorithm)")
    print("    - Calculate mean & variance online (no storage of raw data)")
    print("    - Save statistics only")
    print("  - Memory: O(1) - only statistics in memory")
    print("  - Alpha: [j→last] for each token to last token")
    print()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Extract model dimensions
    n_layer = len(model.backbone.layers)
    nheads = model.backbone.layers[0].mixer.nheads
    headdim = model.backbone.layers[0].mixer.headdim
    print(f"\nModel Architecture:")
    print(f"  Number of layers: {n_layer}")
    print(f"  Number of heads: {nheads}")
    print(f"  Head dimension: {headdim}")
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
        
        # Online computation of mean and variance (Welford's algorithm)
        count = 0
        alpha_mean = None  # [max_doc2_len, nheads, headdim] - only doc2 tokens
        alpha_M2 = None    # For variance computation
        max_doc2_len = 0   # Maximum length of doc2 across samples
        
        # INNER LOOP: process all doc1
        total_doc1 = len(all_doc1_samples)
        pbar_doc1 = tqdm(total=total_doc1, desc=f"  doc2={doc2_id[:8]}", leave=False)
        
        for doc1_sample in all_doc1_samples:
            try:
                # Build prompt
                first_half, second_half, doc2_content = build_prompt(
                    doc2_sample['question'], 
                    doc1_sample['doc_with_title'], 
                    doc2_sample['doc_with_title']
                )
                full_prompt = first_half + second_half
                
                # Get token positions for doc2
                first_half_tokens = tokenizer(first_half, return_tensors="pt")
                num_first_half_tokens = first_half_tokens['input_ids'].shape[1]
                
                doc2_tokens = tokenizer(doc2_content, return_tensors="pt")
                num_doc2_tokens = doc2_tokens['input_ids'].shape[1]
                
                # doc2 starts at position num_first_half_tokens
                doc2_start = num_first_half_tokens
                doc2_end = doc2_start + num_doc2_tokens
                
                # Extract alpha for ALL layers in a SINGLE forward pass
                # Returns: [n_layer, seqlen, nheads, headdim]
                alpha_all_layers = extract_alpha_all(model, tokenizer, full_prompt, device=device)
                
                # Extract only doc2 portion: [n_layer, num_doc2_tokens, nheads, headdim]
                alpha_doc2 = alpha_all_layers[:, doc2_start:doc2_end, :, :]
                
                current_doc2_len = alpha_doc2.shape[1]
                
                # Update max_doc2_len
                if current_doc2_len > max_doc2_len:
                    # Need to expand previous statistics
                    if alpha_mean is not None:
                        old_len = alpha_mean.shape[1]
                        if current_doc2_len > old_len:
                            # Pad previous statistics
                            pad_size = current_doc2_len - old_len
                            alpha_mean = torch.cat([
                                alpha_mean, 
                                torch.zeros(n_layer, pad_size, nheads, headdim)
                            ], dim=1)
                            alpha_M2 = torch.cat([
                                alpha_M2, 
                                torch.zeros(n_layer, pad_size, nheads, headdim)
                            ], dim=1)
                    max_doc2_len = current_doc2_len
                
                # Pad current alpha if needed
                if current_doc2_len < max_doc2_len:
                    pad_size = max_doc2_len - current_doc2_len
                    alpha_doc2 = torch.cat([
                        alpha_doc2, 
                        torch.zeros(n_layer, pad_size, nheads, headdim)
                    ], dim=1)
                
                # Convert to float32 for statistics computation
                alpha_doc2 = alpha_doc2.float()
                
                # Welford's online algorithm for mean and variance
                count += 1
                
                if alpha_mean is None:
                    # First sample
                    alpha_mean = alpha_doc2.clone()
                    alpha_M2 = torch.zeros_like(alpha_doc2)
                    del alpha_doc2
                else:
                    # Update statistics
                    delta = alpha_doc2 - alpha_mean
                    alpha_mean += delta / count
                    delta2 = alpha_doc2 - alpha_mean
                    alpha_M2 += delta * delta2
                    del alpha_doc2, delta, delta2
                
                # Cleanup
                del alpha_all_layers
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError processing doc1={doc1_sample['doc_id'][:8]}: {e}")
                import traceback
                traceback.print_exc()
            
            pbar_doc1.update(1)
        
        pbar_doc1.close()
        
        # Finalize statistics computation
        try:
            print(f"\n  Finalizing statistics for doc2={doc2_id[:8]}...")
            
            if count == 0:
                print(f"  ✗ No data collected for doc2={doc2_id}, skipping...")
                pbar_doc2.update(1)
                continue
            
            # Compute variance from M2 (Welford's algorithm)
            if count > 1:
                alpha_var = alpha_M2 / (count - 1)  # Sample variance
            else:
                alpha_var = torch.zeros_like(alpha_mean)
            
            # Save statistics
            output_file = os.path.join(output_subdir, f"{doc2_id}.pt")
            torch.save({
                'alpha_mean': alpha_mean,
                'alpha_var': alpha_var,
                'doc2_id': doc2_id,
                'num_doc1_samples': count,
                'max_doc2_len': max_doc2_len,
                'n_layer': n_layer,
                'nheads': nheads,
                'headdim': headdim,
                'note': 'alpha_mean shape: [n_layer, max_doc2_len, nheads, headdim]. Only doc2 tokens are included.',
            }, output_file)
            
            print(f"  ✓ Saved statistics to {doc2_id}.pt (n={count}, max_doc2_len={max_doc2_len})")
            
            # Clear memory
            del alpha_mean, alpha_M2, alpha_var
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
