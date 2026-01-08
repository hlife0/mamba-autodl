"""
Collect Beta Statistics for Mamba2

Collects beta state propagation weights from doc1's last token to the final token.
Uses memory-efficient online computation (Welford's algorithm) to calculate mean and variance.

Beta measures how doc1's final state propagates through the model to influence the last token.

Run from project root: python attention_difference2/collect_beta.py
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
from fast_extraction import extract_beta_all


def build_prompt(question, doc1, doc2):
    """Build few-shot prompt with two documents
    
    Returns:
        first_half: fewshot + doc1
        second_half: doc2 + question
        doc1_end_marker: position where doc1 ends
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
    second_half = f"Document 2: {doc2}\n\nQ: {question}\n\nA:"
    
    return first_half, second_half


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect beta statistics for Mamba2")
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
    parser.add_argument('--experiment_name', type=str, default='beta_stats_mamba2', 
                        help='Experiment name')
    args = parser.parse_args()
    
    device = args.device
    dtype = torch.float16 if 'cuda' in device else torch.float32
    
    print("=" * 70)
    print("BETA STATISTICS COLLECTION FOR MAMBA2 (MEMORY-EFFICIENT)")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"All samples (doc1): {args.num_samples}")
    print(f"Partial samples (doc2): {args.num_partial_samples}")
    print(f"Seed: {args.seed}")
    print(f"Experiment: {args.experiment_name}")
    print()
    print("Target: doc1's LAST TOKEN → final sequence token state propagation")
    print("Strategy:")
    print("  - For each doc2 (partial):")
    print("    - Pair with all doc1 samples")
    print("    - Extract beta: doc1_last_token → sequence_last_token")
    print("    - Stream computation (Welford's algorithm)")
    print("    - Calculate mean & variance online")
    print("  - Memory: O(1) - only statistics")
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
    d_state = model.backbone.layers[0].mixer.d_state
    print(f"\nModel Architecture:")
    print(f"  Number of layers: {n_layer}")
    print(f"  Number of heads: {nheads}")
    print(f"  Head dimension: {headdim}")
    print(f"  State dimension: {d_state}")
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
            'question': item.question,
            'doc_with_title': docs[0],
            'doc_id': item.id
        })
    
    print(f"Collected {len(all_doc1_samples)} doc1 samples")
    
    # Select partial doc2 samples
    partial_dataset = dataset.random_choose(args.num_partial_samples, seed=args.seed + 1)
    doc2_samples = []
    for item in partial_dataset:
        docs = item.get_useful()
        if len(docs) < 2:
            continue
        doc2_samples.append({
            'question': item.question,
            'doc_with_title': docs[1],
            'doc_id': item.id
        })
    
    print(f"Selected {len(doc2_samples)} doc2 samples")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%m%d%H%M")
    exp_name = f"{args.experiment_name}_{timestamp}"
    output_path = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")
    print()
    
    # Process each doc2
    for doc2_idx, doc2_sample in enumerate(tqdm(doc2_samples, desc="Processing doc2 samples")):
        doc2_id = doc2_sample['doc_id']
        
        # Initialize statistics
        count = 0
        beta_mean = None
        beta_M2 = None  # For variance computation
        
        # Process all doc1 samples for this doc2
        for doc1_sample in tqdm(all_doc1_samples, desc=f"  doc2={doc2_id[:8]}", leave=False):
            try:
                # Build prompt
                first_half, second_half = build_prompt(
                    doc2_sample['question'],
                    doc1_sample['doc_with_title'],
                    doc2_sample['doc_with_title']
                )
                full_prompt = first_half + second_half
                
                # Tokenize to find doc1_last_token position
                first_half_tokens = tokenizer(first_half, return_tensors="pt")
                doc1_last_token_idx = first_half_tokens['input_ids'].shape[1] - 1
                
                # Extract beta: doc1_last_token → sequence_last_token
                beta = extract_beta_all(
                    model, 
                    tokenizer, 
                    full_prompt, 
                    device=device, 
                    token_idx=doc1_last_token_idx
                )  # [n_layer, nheads, headdim, d_state]
                
                beta = beta.float()
                
                # Welford's online algorithm
                count += 1
                
                if beta_mean is None:
                    # First sample
                    beta_mean = beta.clone()
                    beta_M2 = torch.zeros_like(beta)
                else:
                    # Update
                    delta = beta - beta_mean
                    beta_mean += delta / count
                    delta2 = beta - beta_mean
                    beta_M2 += delta * delta2
                
                # Cleanup
                del beta
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n  Error processing doc1={doc1_sample['doc_id'][:8]}: {e}")
                continue
        
        # Calculate variance
        if count > 1:
            beta_var = beta_M2 / count
        else:
            beta_var = torch.zeros_like(beta_mean)
        
        # Save results for this doc2
        result = {
            'doc2_id': doc2_id,
            'doc2_text': doc2_sample['doc_with_title'],
            'question': doc2_sample['question'],
            'num_doc1_samples': count,
            'beta_mean': beta_mean,
            'beta_var': beta_var
        }
        
        # Save to file
        output_file = os.path.join(output_path, f'{doc2_id}.pt')
        torch.save(result, output_file)
        
        # Cleanup
        del beta_mean, beta_M2, beta_var, result
        torch.cuda.empty_cache()
    
    print()
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Saved {len(doc2_samples)} files to: {output_path}")
    print()
    print("Each file contains:")
    print("  - beta_mean: [n_layer, nheads, headdim, d_state]")
    print("  - beta_var:  [n_layer, nheads, headdim, d_state]")
    print()
    print(f"Total doc2 samples: {len(doc2_samples)}")
    print(f"Doc1 samples per doc2: {len(all_doc1_samples)}")
    print(f"Total computations: {len(doc2_samples) * len(all_doc1_samples)}")
