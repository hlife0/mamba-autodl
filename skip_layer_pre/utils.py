import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from dataset.hotpot import HotpotQAIterator


def prefill_from_scratch(model, input_ids, device):
    """
    Prefill from scratch without any existing cache.
    
    Args:
        model: Mamba model
        input_ids: Input token IDs [batch_size, seq_len]
        device: Device to run on
        
    Returns:
        inference_params: Contains SSM hidden states and convolution cache
        next_token_id: The predicted next token ID [batch_size, 1]
    """
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    
    # Initialize inference parameters (cache)
    inference_params = InferenceParams(
        max_seqlen=1024,  # Set a reasonable max length
        max_batch_size=batch_size
    )
    
    # Forward pass to populate cache
    with torch.no_grad():
        logits = model(
            input_ids,
            inference_params=inference_params,
            num_last_tokens=1
        ).logits
    
    # Update seqlen_offset to reflect processed tokens
    inference_params.seqlen_offset += prompt_len
    
    # Get next token
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    return inference_params, next_token_id


def prefill_with_cache(model, input_ids, previous_cache, device):
    """
    Continue prefilling with existing cache.
    
    Args:
        model: Mamba model
        input_ids: Additional input token IDs [batch_size, seq_len]
        previous_cache: Existing InferenceParams with cached states
        device: Device to run on
        
    Returns:
        inference_params: Updated cache
        next_token_id: The predicted next token ID [batch_size, 1]
    """
    additional_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    
    # When cache exists (seqlen_offset > 0), we need to process tokens one by one
    # because the model enters step mode
    logits = None
    for i in range(additional_len):
        current_token = input_ids[:, i:i+1]
        
        # Create position_ids for current position
        position_ids = torch.full(
            (batch_size, 1),
            previous_cache.seqlen_offset,
            dtype=torch.long,
            device=device
        )
        
        with torch.no_grad():
            logits = model(
                current_token,
                position_ids=position_ids,
                inference_params=previous_cache,
                num_last_tokens=1
            ).logits
        
        # Update seqlen_offset after each token
        previous_cache.seqlen_offset += 1
    
    # Get next token from the last logits
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    return previous_cache, next_token_id


def decode_with_cache(model, token_id, cache, device):
    """
    Decode one token using cache.
    
    Args:
        model: Mamba model
        token_id: Current token ID [batch_size, 1]
        cache: InferenceParams with cached states
        device: Device to run on
        
    Returns:
        cache: Updated cache
        next_token_id: The predicted next token ID [batch_size, 1]
    """
    batch_size = token_id.shape[0]
    
    # Create position_ids for current position
    position_ids = torch.full(
        (batch_size, 1),
        cache.seqlen_offset,
        dtype=torch.long,
        device=device
    )
    
    # Forward pass with single token
    with torch.no_grad():
        logits = model(
            token_id,
            position_ids=position_ids,
            inference_params=cache,
            num_last_tokens=1
        ).logits
    
    # Update seqlen_offset
    cache.seqlen_offset += 1
    
    # Get next token
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    return cache, next_token_id


def inference_logic(
    model,
    tokenizer,
    hybrid_logic,
    question,
    doc1,
    doc2,
    max_new_tokens=30,
    device="cuda"
):
    # Part A: Document 1
    doc1_prompt = f"Document 1: {doc1}\n\n"
    
    # Part B: Document 2
    doc2_prompt = f"Document 2: {doc2}\n\n"
    
    # Part C: Few-shot examples
    few_shot_prompt = (
        "Q: Who is older, Alice or Bob?\n"
        "A: Alice\n\n"
        "Q: Are cats and dogs both mammals?\n"
        "A: yes\n\n"
        "Q: What color do red and blue make?\n"
        "A: purple\n\n"
    )
    
    # Part D: Question
    question_prompt = f"Q: {question}\nA:"
    
    # ========== Path 1: Fast but poor (only doc2) ==========
    prompt_fast = doc2_prompt
    tokens_fast = tokenizer(prompt_fast, return_tensors="pt")
    input_ids_fast = tokens_fast.input_ids.to(device)
    cache_fast, _ = prefill_from_scratch(model, input_ids_fast, device)
    
    # ========== Path 2: Slow but good (doc1 + doc2) ==========
    prompt_slow = doc1_prompt + doc2_prompt
    tokens_slow = tokenizer(prompt_slow, return_tensors="pt")
    input_ids_slow = tokens_slow.input_ids.to(device)
    cache_slow, _ = prefill_from_scratch(model, input_ids_slow, device)
    
    # ========== Create hybrid cache ==========
    num_layers = len(cache_fast.key_value_memory_dict)
    cache_hybrid = InferenceParams(
        max_seqlen=cache_slow.max_seqlen,
        max_batch_size=cache_slow.max_batch_size
    )
    cache_hybrid.seqlen_offset = cache_slow.seqlen_offset
    
    # ========== Vectorized cache and apply hybrid logic ==========
    # Stack all SSM states into tensors for batch processing
    # Shape: [num_layers, batch, d_inner, d_state]
    ssm_fast_stack = torch.stack([cache_fast.key_value_memory_dict[i][1] for i in range(num_layers)])
    ssm_slow_stack = torch.stack([cache_slow.key_value_memory_dict[i][1] for i in range(num_layers)])
    
    # Apply masks: use slow cache for bottom-k dimensions (lowest similarity), fast cache for others
    ssm_hybrid_stack = hybrid_logic(ssm_fast_stack, ssm_slow_stack)
    
    # Store hybrid SSM states
    for layer_idx in range(num_layers):
        # For convolution cache: use fast cache (100%)
        conv_fast = cache_fast.key_value_memory_dict[layer_idx][0]
        
        # For SSM state: use hybrid (mixed)
        ssm_hybrid = ssm_hybrid_stack[layer_idx]
        
        cache_hybrid.key_value_memory_dict[layer_idx] = (conv_fast, ssm_hybrid)
    
    # ========== Prefill question with hybrid cache ==========
    tokens_question = tokenizer(few_shot_prompt + question_prompt, return_tensors="pt")
    input_ids_question = tokens_question.input_ids.to(device)
    cache_hybrid, first_token = prefill_with_cache(model, input_ids_question, cache_hybrid, device)
    
    # ========== Generate answer ==========
    generated_tokens = [first_token.item()]
    current_token = first_token
    stop_strings = ["\nQ:", "\n\n"]
    
    for _ in range(max_new_tokens - 1):
        cache_hybrid, next_token = decode_with_cache(model, current_token, cache_hybrid, device)
        current_token = next_token
        generated_tokens.append(next_token.item())
        
        # Check for stop strings
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if any(stop_str in generated_text for stop_str in stop_strings):
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
            break
    
    return generated_text.strip()