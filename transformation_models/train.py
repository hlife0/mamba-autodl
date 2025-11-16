#!/usr/bin/env python3
"""
GPU-optimized training script for SSM state transformation using single Mamba model
Shared model approach to avoid memory issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from models import SimpleSSMTransformer
from dataset.hotpot_ssm_state import extract_cache_single


def prepare_cache_2(device, gpu_id):
    """
    Prepare doc2 cache for the fixed document
    Returns: (mamba_model, tokenizer, doc2_cache, doc2_content)
    """
    fixed_doc2_id = "5a7a06935542990198eaf050"

    # Get fixed doc2
    from dataset.hotpot import HotpotQAIterator
    hotpot_iterator = HotpotQAIterator("dataset/HotpotQA/hotpot_train_v1.1.json")
    fixed_doc2_item = hotpot_iterator.get_by_id(fixed_doc2_id)

    if not fixed_doc2_item or len(fixed_doc2_item.context) < 2:
        raise ValueError(f"Fixed doc2 ID {fixed_doc2_id} not found or has no doc2")

    doc2_content = f"Document 2: {fixed_doc2_item.context[1].title}\n{fixed_doc2_item.context[1].get_full_text()}\n\n"
    print(f"Using fixed doc2: {fixed_doc2_item.context[1].title}")

    # Load Mamba model and tokenizer
    print("Loading Mamba model...")
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    mamba_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device=device)
    mamba_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Pre-compute doc2 cache once
    print("Pre-computing doc2 cache...")
    doc2_cache = extract_cache_single(mamba_model, tokenizer, doc2_content, device=device)
    print(f"Doc2 cache computed: {doc2_cache.shape}")

    return mamba_model, tokenizer, doc2_cache, doc2_content, hotpot_iterator


if __name__ == "__main__":
    print("Starting training with shared Mamba model...")

    # Configuration
    num_samples = 10
    num_epochs = 3
    gpu_id = 2

    # Clear GPU memory
    torch.cuda.empty_cache()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare doc2 cache and get hotpot_iterator
    mamba_model, tokenizer, doc2_cache, doc2_content, hotpot_iterator = prepare_cache_2(device, gpu_id)

    # Create training model
    transformer_model = SimpleSSMTransformer().to(device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    print(f"Transformer model parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")

    # Get sample indices (use fixed random seed)
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(hotpot_iterator)), num_samples)

    print(f"Selected {num_samples} random samples for training")

    # Training loop
    for epoch in range(num_epochs):
        transformer_model.train()
        total_loss = 0.0

        for sample_idx, i in enumerate(sample_indices):
            # Get item directly
            item = hotpot_iterator[i]
            if item.id == "5a7a06935542990198eaf050":  # Fixed doc2 ID
                continue

            # Create doc1 content
            doc1_content = f"Document 1: {item.context[0].title}\n{item.context[0].get_full_text()}\n\n"

            # Get doc1 cache using shared model
            doc1_cache = extract_cache_single(mamba_model, tokenizer, doc1_content, device=device)

            # Get doc1+doc2 cache using shared model
            combined_content = doc1_content + doc2_content
            combined_cache = extract_cache_single(mamba_model, tokenizer, combined_content, device=device)

            # Calculate target difference: (doc1+doc2) - doc2
            target_diff = combined_cache - doc2_cache

            # Training step
            optimizer.zero_grad()

            # Add batch dimension
            doc1_batch = doc1_cache.unsqueeze(0)
            target_batch = target_diff.unsqueeze(0)

            predicted_diff = transformer_model(doc1_batch)
            loss = criterion(predicted_diff, target_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if sample_idx % 10 == 0:
                print(f"  Sample {sample_idx+1}/{num_samples}, Loss: {loss.item():.6f}")

            # Clear intermediate tensors
            del doc1_cache, combined_cache, target_diff
            del doc1_batch, target_batch, predicted_diff, loss

        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

        # Clear cache between epochs
        torch.cuda.empty_cache()

    # Save model
    torch.save(transformer_model.state_dict(), "ssm_transformer_shared.pth")
    print("Training completed! Model saved to ssm_transformer_shared.pth")