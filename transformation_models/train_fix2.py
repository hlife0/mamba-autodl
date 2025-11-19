import sys
import os
import logging
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
from utils import COMPARISON_FIXED_ID, generate_doc2_prompt, extract_cache_single

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSM state transformation model")
    parser.add_argument("--fixed_doc2_id", type=str, default=COMPARISON_FIXED_ID, help="Fixed doc2 ID for cache preparation")
    parser.add_argument("--exp_name", type=str, default="dbg", help="Experiment name for saving models")
    parser.add_argument("--gpu_train", type=int, default=2, help="GPU for training transformer")
    parser.add_argument("--gpu_mamba", type=int, default=3, help="GPU for Mamba model and data generation")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_percent", type=int, default=20, help="Percentage of data to use for evaluation (0-100)")
    parser.add_argument("--save_dir", type=str, default="/home/hlife/Mamba-experiment/transformation_models/experiments", help="Directory to save models")
    parser.add_argument("--scale_factor", type=float, default=100.0, help="Scaling factor for target differences to address small magnitude issue")

    args = parser.parse_args()

    # Create experiment directory structure
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(exp_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting training with existing iterators...")
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Training GPU: {args.gpu_train}, Mamba GPU: {args.gpu_mamba}")
    logger.info(f"Samples: {args.num_samples}, Epochs: {args.num_epochs}, Batch Size: {args.batch_size}")
    logger.info(f"Evaluation split: {args.eval_percent}%")
    logger.info(f"Scale factor: {args.scale_factor}")
    logger.info(f"Experiment directory: {exp_dir}")

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

    # Calculate train/eval split
    total_samples = args.num_samples
    eval_samples = int(total_samples * args.eval_percent / 100)
    train_samples = total_samples - eval_samples

    logger.info(f"Splitting data: {train_samples} training, {eval_samples} evaluation samples")

    # Create iterators using existing classes
    logger.info("Creating iterators...")

    # Training iterators (only training samples)
    doc1_train_iterator = HotpotDoc1CacheIterator(
        "dataset/HotpotQA/hotpot_train_v1.1.json",
        gpu_id=args.gpu_mamba,
        num_samples=train_samples,  # Only generate training samples
        random_seed=args.random_seed,
        external_model=mamba_model,
        external_tokenizer=mamba_tokenizer
    )

    doc1_plus_train_iterator = HotpotDoc1PlusCacheIterator(
        "dataset/HotpotQA/hotpot_train_v1.1.json",
        plus_content=doc2_prompt,
        gpu_id=args.gpu_mamba,
        num_samples=train_samples,  # Only generate training samples
        random_seed=args.random_seed,  # Same seed for alignment
        external_model=mamba_model,
        external_tokenizer=mamba_tokenizer
    )

    logger.info(f"Created training iterators with {len(doc1_train_iterator)} samples each")

    # Create training model on training GPU
    transformer_model = SimpleMLP().to(train_device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Save model architecture
    model_arch_path = os.path.join(exp_dir, "model_architecture.txt")
    with open(model_arch_path, 'w') as f:
        f.write(str(transformer_model))
    logger.info(f"Model architecture saved to {model_arch_path}")

    # Save model as pickle for easy loading
    import pickle
    model_pickle_path = os.path.join(exp_dir, "model.pkl")
    with open(model_pickle_path, 'wb') as f:
        pickle.dump(transformer_model, f)
    logger.info(f"Model structure saved to {model_pickle_path}")

    logger.info(f"Training model parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    logger.info(f"Effective batch size: {args.batch_size}")

    # Streaming approach: evaluate by skipping training samples
    logger.info(f"Split setup: {train_samples} training, {eval_samples} evaluation samples")

    def stream_evaluation(model, skip_samples, eval_samples):
        """Stream evaluation data without loading all into memory"""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            # Skip training samples, then use eval_samples
            doc1_eval_iterator = HotpotDoc1CacheIterator(
                "dataset/HotpotQA/hotpot_train_v1.1.json",
                gpu_id=args.gpu_mamba,
                num_samples=skip_samples + eval_samples,
                random_seed=args.random_seed,
                external_model=mamba_model,
                external_tokenizer=mamba_tokenizer
            )

            doc1_plus_eval_iterator = HotpotDoc1PlusCacheIterator(
                "dataset/HotpotQA/hotpot_train_v1.1.json",
                plus_content=doc2_prompt,
                gpu_id=args.gpu_mamba,
                num_samples=skip_samples + eval_samples,
                random_seed=args.random_seed,
                external_model=mamba_model,
                external_tokenizer=mamba_tokenizer
            )

            # Skip training samples
            for _ in range(skip_samples):
                next(doc1_eval_iterator)
                next(doc1_plus_eval_iterator)

            batch_doc1 = []
            batch_doc1_plus = []

            for doc1_cache, doc1_plus_cache in zip(doc1_eval_iterator, doc1_plus_eval_iterator):
                doc1_cache = doc1_cache.to(train_device)
                doc1_plus_cache = doc1_plus_cache.to(train_device)

                batch_doc1.append(doc1_cache)
                batch_doc1_plus.append(doc1_plus_cache)

                if len(batch_doc1) >= args.batch_size:
                    doc1_batch = torch.stack(batch_doc1)
                    doc1_plus_batch = torch.stack(batch_doc1_plus)
                    target_diff = doc1_plus_batch - doc2_cache.unsqueeze(0)
                    # Apply scaling to target difference for evaluation consistency
                    target_diff_scaled = target_diff * args.scale_factor

                    predicted_diff_scaled = model(doc1_batch)
                    loss = criterion(predicted_diff_scaled, target_diff_scaled)

                    total_loss += loss.item()
                    num_batches += 1

                    batch_doc1 = []
                    batch_doc1_plus = []

            # Process remaining items
            if len(batch_doc1) > 0:
                doc1_batch = torch.stack(batch_doc1)
                doc1_plus_batch = torch.stack(batch_doc1_plus)
                target_diff = doc1_plus_batch - doc2_cache.unsqueeze(0)
                # Apply scaling to target difference for evaluation consistency
                target_diff_scaled = target_diff * args.scale_factor

                predicted_diff_scaled = model(doc1_batch)
                loss = criterion(predicted_diff_scaled, target_diff_scaled)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    # Training loop with evaluation
    for epoch in range(args.num_epochs):
        transformer_model.train()
        total_loss = 0.0
        num_batches = 0

        # Collect batches
        batch_doc1 = []
        batch_doc1_plus = []

        for doc1_cache, doc1_plus_cache in zip(doc1_train_iterator, doc1_plus_train_iterator):
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
                # Apply scaling to target difference to address small magnitude issue
                target_diff_scaled = target_diff * args.scale_factor

                # Training step
                optimizer.zero_grad()

                predicted_diff_scaled = transformer_model(doc1_batch)
                # Loss is calculated on scaled targets, model learns to predict scaled differences
                loss = criterion(predicted_diff_scaled, target_diff_scaled)

                # Check for NaN
                if torch.isnan(loss):
                    logger.warning(f"⚠️ NaN detected in loss at batch {num_batches}")
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
                    logger.info(f"  Batch {num_batches}, Loss: {loss.item():.6f}")

        # Process remaining items in last incomplete batch
        if len(batch_doc1) > 0:
            doc1_batch = torch.stack(batch_doc1)
            doc1_plus_batch = torch.stack(batch_doc1_plus)
            target_diff = doc1_plus_batch - doc2_cache.unsqueeze(0)
            # Apply scaling to target difference to address small magnitude issue
            target_diff_scaled = target_diff * args.scale_factor

            optimizer.zero_grad()
            predicted_diff_scaled = transformer_model(doc1_batch)
            # Loss is calculated on scaled targets, model learns to predict scaled differences
            loss = criterion(predicted_diff_scaled, target_diff_scaled)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {avg_train_loss:.6f}, Batches: {num_batches}")

        # Save checkpoint for each epoch
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(exp_dir, f"checkpoint_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'scale_factor': args.scale_factor
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Evaluation using streaming approach
        if eval_samples > 0:
            eval_loss = stream_evaluation(transformer_model, train_samples, eval_samples)
            logger.info(f"  Evaluation Loss: {eval_loss:.6f}")
        else:
            logger.info("  No evaluation data available")

        # Clear cache between epochs
        torch.cuda.empty_cache()

    logger.info("Training completed!")
    logger.info(f"All checkpoints saved in {exp_dir}")