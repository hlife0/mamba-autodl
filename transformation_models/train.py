import sys
import os
from typing import Dict
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
from models import SimpleMLP
from dataset.hotpot import HotpotQAIterator
from dataset.hotpot_ssm_state import HotpotDoc1CacheIterator, HotpotDoc1PlusCacheIterator
from utils import COMPARISON_FIXED_ID, generate_doc1_prompt, generate_doc2_prompt, generate_doc12_prompt, extract_cache_single

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSM state transformation model")
    parser.add_argument("--fixed_doc2_id", type=str, default=COMPARISON_FIXED_ID, help="Fixed doc2 ID for cache preparation")
    parser.add_argument("--exp_name", type=str, default="dbg", help="Experiment name for saving models")
    parser.add_argument("--gpu_train", type=int, default=2, help="GPU for training transformer")
    parser.add_argument("--gpu_mamba", type=int, default=3, help="GPU for Mamba model and data generation")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="/home/hlife/Mamba-experiment/transformation_models/experiments", help="Directory to save models")

    args = parser.parse_args()

    print("Starting training with existing iterators...")
    print(f"Experiment: {args.exp_name}")
    print(f"Training GPU: {args.gpu_train}, Mamba GPU: {args.gpu_mamba}")
    print(f"Samples: {args.num_samples}, Epochs: {args.num_epochs}, Batch Size: {args.batch_size}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.exp_name}_model.pth")

    # Clear GPU memory for Mamba GPU
    torch.cuda.empty_cache()

    # Set up devices
    mamba_device = f"cuda:{args.gpu_mamba}" if torch.cuda.is_available() else "cpu"
    train_device = f"cuda:{args.gpu_train}" if torch.cuda.is_available() else "cpu"
    mamba_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device=mamba_device)
    mamba_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # prepare fixed doc2 content and cache
    doc2_item_useful = HotpotQAIterator("dataset/HotpotQA/hotpot_train_v1.1.json").get_by_id(args.fixed_doc2_id).get_useful()
    doc2_prompt = generate_doc2_prompt(doc2_item_useful)
    doc2_cache = extract_cache_single(mamba_model, mamba_tokenizer, doc2_prompt, device=mamba_device, return_tensor=True)
    doc2_cache = doc2_cache.to(train_device)

    # Create iterators using existing classes
    print("Creating iterators...")
    doc1_iterator = HotpotDoc1CacheIterator(
        "dataset/HotpotQA/hotpot_train_v1.1.json",
        gpu_id=args.gpu_mamba,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        external_model=mamba_model,
        external_tokenizer=mamba_tokenizer
    )

    doc1_plus_iterator = HotpotDoc1PlusCacheIterator(
        "dataset/HotpotQA/hotpot_train_v1.1.json",
        plus_content=doc2_prompt,
        gpu_id=args.gpu_mamba,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        external_model=mamba_model,
        external_tokenizer=mamba_tokenizer
    )

    # Create training model on training GPU
    transformer_model = SimpleMLP().to(train_device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    print(f"Transformer model parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    print(f"Effective batch size: {args.batch_size}")

    # Training loop using iterators with real batching
    for epoch in range(args.num_epochs):
        transformer_model.train()
        total_loss = 0.0
        num_batches = 0

        # Collect batches
        batch_doc1 = []
        batch_doc1_plus = []

        for doc1_cache, doc1_plus_cache in zip(doc1_iterator, doc1_plus_iterator):
            # Move to training GPU
            doc1_cache = doc1_cache.to(train_device)
            doc1_plus_cache = doc1_plus_cache.to(train_device)

            batch_doc1.append(doc1_cache)
            batch_doc1_plus.append(doc1_plus_cache)

            # When batch is full, process it
            if len(batch_doc1) >= args.batch_size:
                # Stack to create batch tensors
                doc1_batch = torch.stack(batch_doc1)  # [batch_size, 64, 5120, 16]
                doc1_plus_batch = torch.stack(batch_doc1_plus)

                # Calculate target difference: (doc1+doc2) - doc2
                target_diff = doc1_plus_batch - doc2_cache.unsqueeze(0)

                # Training step
                optimizer.zero_grad()

                predicted_diff = transformer_model(doc1_batch)
                loss = criterion(predicted_diff, target_diff)

  
                # Check for NaN
                if torch.isnan(loss):
                    print(f"⚠️ NaN detected in loss at batch {num_batches}")
                    break

                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Clear batch
                batch_doc1 = []
                batch_doc1_plus = []

                if num_batches % 5 == 0:
                    print(f"  Batch {num_batches}, Loss: {loss.item():.6f}")

        # Process remaining items in last incomplete batch
        if len(batch_doc1) > 0:
            doc1_batch = torch.stack(batch_doc1)
            doc1_plus_batch = torch.stack(batch_doc1_plus)
            target_diff = doc1_plus_batch - doc2_cache.unsqueeze(0)

            optimizer.zero_grad()
            predicted_diff = transformer_model(doc1_batch)
            loss = criterion(predicted_diff, target_diff)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.6f}, Batches: {num_batches}")

        # Clear cache between epochs
        torch.cuda.empty_cache()

    # Save model
    torch.save(transformer_model.state_dict(), save_path)
    print(f"Training completed! Model saved to {save_path}")