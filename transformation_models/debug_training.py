#!/usr/bin/env python3
"""
Debug why training loss is always 0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import SimpleMLP
from dataset.hotpot_ssm_state import HotpotDoc1CacheIterator, HotpotDoc1PlusCacheIterator, extract_cache_single


def debug_training_loss():
    """Debug why loss is 0"""
    print("🔍 Debugging training loss issue...")

    # Basic setup
    device = "cuda:2"
    num_samples = 4
    batch_size = 2

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Prepare doc2 cache
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer
    from dataset.hotpot import HotpotQAIterator

    fixed_doc2_id = "5a7a06935542990198eaf050"
    hotpot_iterator = HotpotQAIterator("dataset/HotpotQA/hotpot_train_v1.1.json")
    fixed_doc2_item = hotpot_iterator.get_by_id(fixed_doc2_id)

    # Check if the issue is with content format
    doc2_simple_content = fixed_doc2_item.context[1].get_full_text()
    doc2_formatted_content = f"Document 2: {fixed_doc2_item.context[1].title}\n{fixed_doc2_item.context[1].get_full_text()}\n\n"

    print(f"Fixed doc2: {fixed_doc2_item.context[1].title}")
    print(f"Doc2 simple content length: {len(doc2_simple_content)}")
    print(f"Doc2 formatted content length: {len(doc2_formatted_content)}")
    print(f"Doc2 formatted content preview: {doc2_formatted_content[:200]}...")

    doc2_content = doc2_formatted_content

    # Load models
    print("Loading models...")
    mamba_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device="cuda:3")
    mamba_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

  # Test both simple and formatted doc2 content
    print("Testing different doc2 content formats...")

    # Test 1: Just the formatted doc2 content (what we currently use)
    doc2_cache_formatted = extract_cache_single(mamba_model, tokenizer, doc2_formatted_content, device="cuda:3")
    doc2_cache_formatted = doc2_cache_formatted.to(device)
    print(f"Formatted doc2 cache range: [{doc2_cache_formatted.min():.6f}, {doc2_cache_formatted.max():.6f}]")
    print(f"Formatted doc2 cache norm: {torch.norm(doc2_cache_formatted).item():.6f}")

    # The issue is clear: doc2 cache is all zeros! Let's continue with the debug...

    doc2_cache = doc2_cache_formatted

    transformer_model = SimpleMLP().to(device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Doc2 cache shape: {doc2_cache.shape}")
    print(f"Doc2 cache range: [{doc2_cache.min():.6f}, {doc2_cache.max():.6f}]")

    # Create iterators
    doc1_iterator = HotpotDoc1CacheIterator(
        "dataset/HotpotQA/hotpot_train_v1.1.json",
        gpu_id=3,
        num_samples=num_samples,
        random_seed=42,
        external_model=mamba_model,
        external_tokenizer=tokenizer
    )

    doc1_plus_iterator = HotpotDoc1PlusCacheIterator(
        "dataset/HotpotQA/hotpot_train_v1.1.json",
        plus_content=doc2_content,
        gpu_id=3,
        num_samples=num_samples,
        random_seed=42,
        external_model=mamba_model,
        external_tokenizer=tokenizer
    )

    # Debug: Check first few samples
    print("\n🔍 Checking first few samples...")
    for i, (doc1_cache, doc1_plus_cache) in enumerate(zip(doc1_iterator, doc1_plus_iterator)):
        if i >= 2:
            break

        print(f"\nSample {i+1}:")
        print(f"  Doc1 cache range: [{doc1_cache.min():.6f}, {doc1_cache.max():.6f}]")
        print(f"  Doc1+ cache range: [{doc1_plus_cache.min():.6f}, {doc1_plus_cache.max():.6f}]")

        # Move to training device
        doc1_cache = doc1_cache.to(device)
        doc1_plus_cache = doc1_plus_cache.to(device)

        # Calculate target difference
        target_diff = doc1_plus_cache - doc2_cache
        print(f"  Target diff range: [{target_diff.min():.6f}, {target_diff.max():.6f}]")
        print(f"  Target diff mean: {target_diff.mean():.6f}, std: {target_diff.std():.6f}")

        # Check the individual components to understand the issue
        print(f"  Doc1 cache range: [{doc1_cache.min():.6f}, {doc1_cache.max():.6f}]")
        print(f"  Doc2 cache range: [{doc2_cache.min():.6f}, {doc2_cache.max():.6f}]")
        print(f"  Doc1+ cache range: [{doc1_plus_cache.min():.6f}, {doc1_plus_cache.max():.6f}]")

        # Check if doc1_plus_cache ≈ doc2_cache (which would make the diff ≈ 0)
        doc1_plus_vs_doc2_diff = torch.abs(doc1_plus_cache - doc2_cache)
        print(f"  |Doc1+ - Doc2| range: [{doc1_plus_vs_doc2_diff.min():.6f}, {doc1_plus_vs_doc2_diff.max():.6f}]")
        print(f"  |Doc1+ - Doc2| mean: {doc1_plus_vs_doc2_diff.mean():.6f}")

        # Check relative magnitudes
        doc1_plus_norm = torch.norm(doc1_plus_cache).item()
        doc2_norm = torch.norm(doc2_cache).item()
        target_norm = torch.norm(target_diff).item()
        print(f"  Norms - Doc1+: {doc1_plus_norm:.6f}, Doc2: {doc2_norm:.6f}, Target: {target_norm:.6f}")
        print(f"  Target/Doc2+ ratio: {target_norm/(doc1_plus_norm+1e-9):.9f}")

        # Test model prediction
        transformer_model.eval()
        with torch.no_grad():
            doc1_batch = doc1_cache.unsqueeze(0)  # [1, 64, 5120, 16]
            predicted_diff = transformer_model(doc1_batch)
            print(f"  Predicted diff range: [{predicted_diff.min():.6f}, {predicted_diff.max():.6f}]")
            print(f"  Predicted diff mean: {predicted_diff.mean():.6f}, std: {predicted_diff.std():.6f}")

            # Calculate loss
            loss = criterion(predicted_diff, target_diff.unsqueeze(0))
            print(f"  Loss: {loss.item():.8f}")

        # Check if target diff is zero (possible issue)
        if torch.allclose(target_diff, torch.zeros_like(target_diff)):
            print("  ⚠️ Target diff is essentially zero - this explains the zero loss!")

        # Check if model output is zero
        if torch.allclose(predicted_diff, torch.zeros_like(predicted_diff)):
            print("  ⚠️ Model output is essentially zero!")

        break


if __name__ == "__main__":
    debug_training_loss()