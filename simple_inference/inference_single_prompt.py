#!/usr/bin/env python3
"""
Simple inference script for Mamba models.
Usage: python inference_single_prompt.py --prompt "Your prompt here"
"""

import argparse
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def main():
    parser = argparse.ArgumentParser(description="Simple Mamba inference script")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Once upon a time in a land far, far away,",
        # required=True,
        help="Input prompt string"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="state-spaces/mamba-2.8B",
        help="Model name or path (default: state-spaces/mamba-2.8B)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length (default: 100)"
    )
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}")
    
    # Load tokenizer (Mamba uses GPT-NeoX tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Load model
    model = MambaLMHeadModel.from_pretrained(
        args.model_name,
        device=device,
        dtype=dtype
    )
    model.eval()
    
    # Tokenize input
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    # Generate (greedy decoding by default)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + args.max_length,
            cg=True,
            return_dict_in_generate=True,
            enable_timing=False,
        )
    
    # Decode and print result
    # Get only the newly generated tokens (excluding the input prompt)
    generated_tokens = output.sequences[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("Generated text:")
    print(generated_text)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
