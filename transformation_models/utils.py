from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba.mamba_ssm.utils.generation import InferenceParams
import torch

COMPARISON_FIXED_ID = "5a7a06935542990198eaf050" # Obtained by HotpotQAIterator("/home/hlife/Mamba-experiment/dataset/HotpotQA/hotpot_train_v1.1.json").get_by_type("comparison")[0].id

def generate_doc1_prompt(item: list[dict[str, str]]) -> str:
    """
    Generate prompt for Document 1 from the given item. Currently only support Comparison type of data. Format:
    
    Document 1: {title}
    
    {content}
    
    args:
        item: List of two dictionaries, each with 'title' and 'content' keys. This can be directly obtained by HotpotQAItem.get_useful().
        
    returns:
        A formatted string prompt for Document 1, with two newlines at the end.
    """
    if len(item) != 2:
        raise ValueError("Error in generating doc1 prompt: Expected exactly two documents in the item")
    if 'title' not in item[0] or 'content' not in item[0]:
        raise ValueError("Error in generating doc1 prompt: Missing 'title' or 'content' in the first document")
    return f"Document 1: {item[0]['title']}\n{item[0]['content']}\n\n"

def generate_doc2_prompt(item: list[dict[str, str]]) -> str:
    """
    Generate prompt for Document 2 from the given item. Currently only support Comparison type of data. Format:
    
    Document 2: {title}
    
    {content}
    
    args:
        item: List of two dictionaries, each with 'title' and 'content' keys. This can be directly obtained by HotpotQAItem.get_useful().
        
    returns:
        A formatted string prompt for Document 2, with two newlines at the end.
    """
    if len(item) != 2:
        raise ValueError("Error in generating doc2 prompt: Expected exactly two documents in the item")
    if 'title' not in item[1] or 'content' not in item[1]:
        raise ValueError("Error in generating doc2 prompt: Missing 'title' or 'content' in the second document")
    return f"Document 2: {item[1]['title']}\n{item[1]['content']}\n\n"

def generate_doc12_prompt(item: list[dict[str, str]]) -> str:
    """
    Generate prompt for Document 1 and Document 2 from the given item. Currently only support Comparison type of data. Format:
    
    Document 1: {title}
    
    {content}


    Document 2: {title}
    
    {content}
    
    args:
        item: List of two dictionaries, each with 'title' and 'content' keys. This can be directly obtained by HotpotQAItem.get_useful().
        
    returns:
        A formatted string prompt for Document 1 and Document 2, with two newlines at the end.
    """
    return generate_doc1_prompt(item) + generate_doc2_prompt(item)

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