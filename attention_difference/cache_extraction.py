import argparse
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

def extract_cache_single(model, tokenizer, prompt, device="cuda:0", return_tensor=True):
    """
    Extract cache from a single prompt using pre-loaded model and tokenizer.
    
    Args:
        model: Pre-loaded MambaLMHeadModel
        tokenizer: Pre-loaded tokenizer
        prompt: Input text prompt
        device: Device to use
        return_tensor: If True, return stacked SSM states; else return InferenceParams
    
    Returns:
        If return_tensor=True: torch.Tensor of shape [num_layers, d_inner, d_state]
        If return_tensor=False: InferenceParams object
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    inference_params = InferenceParams(
        max_seqlen=prompt_len + 8,
        max_batch_size=batch_size
    )

    with torch.no_grad():
        logits = model(input_ids, inference_params=inference_params).logits
    
    if return_tensor:
        kv_dict = inference_params.key_value_memory_dict
        
        ssm_states = []
        for layer_idx in sorted(kv_dict.keys()):
            conv_state, ssm_state = kv_dict[layer_idx]
            ssm_states.append(ssm_state.squeeze(0))
        stacked_states = torch.stack(ssm_states, dim=0)
        return stacked_states
    else:
        return inference_params

def extract_cache_diff(model, tokenizer, prompt1, prompt2, device="cuda:0", return_tensor=True):
    """
    Extract cache difference between two prompts using pre-loaded model and tokenizer.
    
    Args:
        model: Pre-loaded MambaLMHeadModel
        tokenizer: Pre-loaded tokenizer
        prompt1: First input text prompt
        prompt2: Second input text prompt
        device: Device to use
        return_tensor: If True, return stacked SSM states; else return InferenceParams
    
    Returns:
        torch.Tensor of shape [num_layers, d_inner, d_state] representing cache1 - cache2
    """
    cache1 = extract_cache_single(model, tokenizer, prompt1, device, return_tensor)
    cache2 = extract_cache_single(model, tokenizer, prompt2, device, return_tensor)
    return cache1 - cache2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--return_tensor", type=bool, default=True)
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32
    
    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    model.eval()
    
    print("Model loaded. Extracting cache...\n")

    result = extract_cache_single(
        model,
        tokenizer,
        args.prompt, 
        device,
        return_tensor=args.return_tensor
    )

    if args.return_tensor:
        print(f"SSM States Tensor Shape: {result.shape}")
        print(f"Expected: [num_layers, d_inner, d_state]")
        print(f"Data type: {result.dtype}")
        print(f"Device: {result.device}")
        print(f"\nStatistics:")
        print(f"  Min: {result.min().item():.6f}")
        print(f"  Max: {result.max().item():.6f}")
        print(f"  Mean: {result.mean().item():.6f}")
        print(f"  Std: {result.std().item():.6f}")
    else:
        cache = result
        print("max_seqlen: ", cache.max_seqlen)
        print("max_batch_size: ", cache.max_batch_size)
        print("seqlen_offset: ", cache.seqlen_offset)
        print("batch_size_offset: ", cache.batch_size_offset)
        print("key_value_memory_dict: ", type(cache.key_value_memory_dict), len(cache.key_value_memory_dict), cache.key_value_memory_dict.keys())

