"""
Fast Alpha/Beta Matrix Extraction for Mamba2 (Last Token Only)

Memory-efficient version: Only extracts attention from all tokens to the LAST token.
This reduces memory usage from O(seqlen²) to O(seqlen).

Main Functions:
- extract_alpha(model, tokenizer, prompt, device, layer_idx): Extract alpha for last token only
  Returns: [seqlen, nheads, headdim] - attention from all tokens to LAST token

- extract_beta(model, tokenizer, prompt, device, layer_idx): Extract beta for last token only
  Returns: [seqlen, nheads, headdim, d_state] - state propagation weights to LAST token

- extract_alpha_all(model, tokenizer, prompt, device): Extract alpha for all layers at once
  Returns: [num_layers, seqlen, nheads, headdim]

- extract_alpha_batch(model, tokenizer, prompts, device, layer_idx): Batch extraction for multiple prompts
  Returns: list of [seqlen, nheads, headdim]
"""
import torch
import torch.nn.functional as F
from einops import rearrange


def extract_alpha(model, tokenizer, prompt, device="cuda:0", layer_idx=0):
    """
    Extract alpha attention weights from all tokens to the LAST token only.
    
    Memory-efficient: O(seqlen) instead of O(seqlen²)
    
    Args:
        model: Mamba2 model
        tokenizer: tokenizer
        prompt: input text (string)
        device: device to use
        layer_idx: layer index to extract from
    
    Returns:
        alpha_last: [seqlen, nheads, headdim] - attention from all tokens to LAST token
    """
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    result = {'alpha': None}
    
    def extract_hook(module, input, output):
        hidden_states = input[0]
        batch, seqlen, _ = hidden_states.shape
        last_idx = seqlen - 1
        
        # Project and split
        zxbcdt = module.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * module.d_ssm - 2 * module.ngroups * module.d_state - module.nheads) // 2
        
        # Extract only what we need: xBC and dt
        _, _, _, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, module.d_ssm, module.d_ssm + 2 * module.ngroups * module.d_state, module.nheads],
            dim=-1
        )
        
        # Conv1d
        if module.activation in ["silu", "swish"]:
            xBC = module.act(module.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
        
        # Split into B and C
        _, B, C = torch.split(
            xBC, 
            [module.d_ssm, module.ngroups * module.d_state, module.ngroups * module.d_state], 
            dim=-1
        )
        
        # Reshape
        B = rearrange(B, "b l (g n) -> b l g n", g=module.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=module.ngroups)
        
        # Process dt
        dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))
        
        # Get A
        A = -torch.exp(module.A_log.float())
        
        # Compute discrete_A and discrete_B efficiently
        dt_expanded = dt.unsqueeze(-1)  # [batch, seqlen, nheads, 1]
        A_expanded = A.view(1, 1, -1, 1)  # [1, 1, nheads, 1]
        
        discrete_A_scalar = torch.exp(A_expanded * dt_expanded)  # [batch, seqlen, nheads, 1]
        discrete_A = discrete_A_scalar.unsqueeze(-1).expand(
            batch, seqlen, module.nheads, module.headdim, module.d_state
        )
        
        # discrete_B
        dt_for_B = dt.unsqueeze(-1).unsqueeze(-1)
        B_expanded = B.unsqueeze(2)
        discrete_B = dt_for_B * B_expanded
        
        # Get D
        D = module.D
        
        # Calculate alpha for LAST token only (assume batch=1, ngroups=1)
        A_vals = discrete_A[0].float()  # [seqlen, nheads, headdim, d_state]
        B_vals = discrete_B[0, :, :, 0, :].float()  # [seqlen, nheads, d_state]
        C_last = C[0, last_idx, 0, :].float()  # [d_state] - only last token's C
        
        # Compute cumulative products from each token j to last token
        # Abar[j] = prod(A[k] for k in range(j+1, last+1))
        log_A = torch.log(A_vals + 1e-10)  # [seqlen, nheads, headdim, d_state]
        log_A_cumsum = torch.cumsum(log_A, dim=0)  # [seqlen, nheads, headdim, d_state]
        
        # Abar[j] = exp(log_A_cumsum[last] - log_A_cumsum[j])
        log_A_cumsum_last = log_A_cumsum[last_idx]  # [nheads, headdim, d_state]
        log_A_cumsum_j = log_A_cumsum  # [seqlen, nheads, headdim, d_state]
        
        # For j < last: Abar[j] = exp(sum(log_A[k]) for k in j+1..last)
        #             = exp(log_A_cumsum[last] - log_A_cumsum[j])
        # For j == last: Abar[last] = 1 (empty product)
        Abar = torch.exp(log_A_cumsum_last.unsqueeze(0) - log_A_cumsum_j)  # [seqlen, nheads, headdim, d_state]
        
        # Compute effective B: B_eff[j] = Abar[j] * B[j]
        B_eff = Abar * B_vals.unsqueeze(2)  # [seqlen, nheads, headdim, d_state]
        
        # Compute alpha: α[j,h,p] = Σ_n C_last[n] * B_eff[j,h,p,n]
        alpha_last = torch.einsum('n,jhpn->jhp', C_last, B_eff)  # [seqlen, nheads, headdim]
        
        # Add D to diagonal (only for last token, which is alpha_last[last])
        if D is not None:
            if D.dim() == 1:
                alpha_last[last_idx, :, :] += D.unsqueeze(-1)
            else:
                alpha_last[last_idx, :, :] += D
        
        # Store result
        result['alpha'] = alpha_last
        
        # Aggressive cleanup
        del zxbcdt, xBC, dt, B, C, dt_expanded, A_expanded, discrete_A_scalar, discrete_A
        del dt_for_B, B_expanded, discrete_B, A_vals, B_vals, C_last
        del log_A, log_A_cumsum, log_A_cumsum_last, log_A_cumsum_j, Abar, B_eff
        torch.cuda.empty_cache()
    
    # Register hook
    target_layer = model.backbone.layers[layer_idx].mixer
    hook_handle = target_layer.register_forward_hook(extract_hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Cleanup
    hook_handle.remove()
    
    return result['alpha']


def extract_beta(model, tokenizer, prompt, device="cuda:0", layer_idx=0, token_idx=0):
    """
    Extract beta state propagation weight from a single token to the LAST token only.
    
    Beta formula: beta[h, p, s] = Abar[token_idx, h, p, s] * C[last, s]
    where Abar[token_idx] = A[token_idx+1] × A[token_idx+2] × ... × A[last]
    
    Args:
        model: Mamba2 model
        tokenizer: tokenizer
        prompt: input text (string)
        device: device to use
        layer_idx: layer index to extract from
        token_idx: int - single token position to extract beta for (e.g., a doc1 token position)
    
    Returns:
        beta_last: [nheads, headdim, d_state] - state propagation weights from token_idx to LAST token
    """
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    result = {'beta': None}
    
    def extract_hook(module, input, output):
        hidden_states = input[0]
        batch, seqlen, _ = hidden_states.shape
        last_idx = seqlen - 1
        
        # Project and split
        zxbcdt = module.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * module.d_ssm - 2 * module.ngroups * module.d_state - module.nheads) // 2
        
        # Extract only what we need: xBC and dt
        _, _, _, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, module.d_ssm, module.d_ssm + 2 * module.ngroups * module.d_state, module.nheads],
            dim=-1
        )
        
        # Conv1d
        if module.activation in ["silu", "swish"]:
            xBC = module.act(module.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
        
        # Split into C
        _, _, C = torch.split(
            xBC, 
            [module.d_ssm, module.ngroups * module.d_state, module.ngroups * module.d_state], 
            dim=-1
        )
        
        # Reshape C
        C = rearrange(C, "b l (g n) -> b l g n", g=module.ngroups)
        
        # Process dt
        dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))
        
        # Get A
        A = -torch.exp(module.A_log.float())
        
        # Compute discrete_A
        dt_expanded = dt.unsqueeze(-1)  # [batch, seqlen, nheads, 1]
        A_expanded = A.view(1, 1, -1, 1)  # [1, 1, nheads, 1]
        
        discrete_A_scalar = torch.exp(A_expanded * dt_expanded)  # [batch, seqlen, nheads, 1]
        discrete_A = discrete_A_scalar.unsqueeze(-1).expand(
            batch, seqlen, module.nheads, module.headdim, module.d_state
        )
        
        # Calculate beta for LAST token only (assume batch=1, ngroups=1)
        A_vals = discrete_A[0].float()  # [seqlen, nheads, headdim, d_state]
        C_last = C[0, last_idx, 0, :].float()  # [d_state] - only last token's C
        
        # Compute cumulative products from each token j to last token
        # Abar[j] = prod(A[k] for k in range(j+1, last+1))
        log_A = torch.log(A_vals + 1e-10)  # [seqlen, nheads, headdim, d_state]
        log_A_cumsum = torch.cumsum(log_A, dim=0)  # [seqlen, nheads, headdim, d_state]
        
        # Abar[j] = exp(log_A_cumsum[last] - log_A_cumsum[j])
        log_A_cumsum_last = log_A_cumsum[last_idx]  # [nheads, headdim, d_state]
        log_A_cumsum_j = log_A_cumsum  # [seqlen, nheads, headdim, d_state]
        
        # Abar[j] = A[j+1] × A[j+2] × ... × A[last]
        Abar = torch.exp(log_A_cumsum_last.unsqueeze(0) - log_A_cumsum_j)  # [seqlen, nheads, headdim, d_state]
        
        # Select only the specified token
        Abar_token = Abar[token_idx]  # [nheads, headdim, d_state]
        
        # Compute beta: β[h,p,s] = Abar[token_idx,h,p,s] * C_last[s]
        beta_last = Abar_token * C_last.unsqueeze(0).unsqueeze(0)  # [nheads, headdim, d_state]
        
        # Store result
        result['beta'] = beta_last
        
        # Cleanup
        del zxbcdt, xBC, C, dt, dt_expanded, A_expanded, discrete_A_scalar, discrete_A
        del A_vals, C_last, log_A, log_A_cumsum, log_A_cumsum_last, log_A_cumsum_j, Abar
        torch.cuda.empty_cache()
    
    # Register hook
    target_layer = model.backbone.layers[layer_idx].mixer
    hook_handle = target_layer.register_forward_hook(extract_hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Cleanup
    hook_handle.remove()
    
    return result['beta']


def extract_beta_all(model, tokenizer, prompt, device="cuda:0", token_idx=0):
    """
    Extract beta state propagation weights from a single token to the LAST token for ALL layers.
    
    Single forward pass extracts beta from all layers simultaneously.
    
    Args:
        model: Mamba2 model
        tokenizer: tokenizer
        prompt: input text (string)
        device: device to use
        token_idx: int - single token position to extract beta for (e.g., last token of doc1)
    
    Returns:
        beta_all: [num_layers, nheads, headdim, d_state] - beta from token_idx to last token for all layers
    """
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    num_layers = len(model.backbone.layers)
    results = [{'beta': None} for _ in range(num_layers)]
    
    def make_extract_hook(layer_idx):
        def extract_hook(module, input, output):
            hidden_states = input[0]
            batch, seqlen, _ = hidden_states.shape
            last_idx = seqlen - 1
            
            # Validate token_idx
            if token_idx < 0 or token_idx >= seqlen:
                raise ValueError(f"token_idx={token_idx} is out of bounds for sequence length {seqlen}")
            
            # Project and split
            zxbcdt = module.in_proj(hidden_states)
            d_mlp = (zxbcdt.shape[-1] - 2 * module.d_ssm - 2 * module.ngroups * module.d_state - module.nheads) // 2
            
            _, _, _, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, module.d_ssm, module.d_ssm + 2 * module.ngroups * module.d_state, module.nheads],
                dim=-1
            )
            
            # Conv1d
            if module.activation in ["silu", "swish"]:
                xBC = module.act(module.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
            
            # Split into C
            _, _, C = torch.split(
                xBC, 
                [module.d_ssm, module.ngroups * module.d_state, module.ngroups * module.d_state], 
                dim=-1
            )
            
            C = rearrange(C, "b l (g n) -> b l g n", g=module.ngroups)
            
            # Process dt
            dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))
            
            # Get A
            A = -torch.exp(module.A_log.float())
            
            # Compute discrete_A
            dt_expanded = dt.unsqueeze(-1)
            A_expanded = A.view(1, 1, -1, 1)
            
            discrete_A_scalar = torch.exp(A_expanded * dt_expanded)
            discrete_A = discrete_A_scalar.unsqueeze(-1).expand(
                batch, seqlen, module.nheads, module.headdim, module.d_state
            )
            
            # Calculate beta
            A_vals = discrete_A[0].float()
            C_last = C[0, last_idx, 0, :].float()
            
            log_A = torch.log(A_vals + 1e-10)
            log_A_cumsum = torch.cumsum(log_A, dim=0)
            
            log_A_cumsum_last = log_A_cumsum[last_idx]
            log_A_cumsum_j = log_A_cumsum
            
            Abar = torch.exp(log_A_cumsum_last.unsqueeze(0) - log_A_cumsum_j)
            
            # Select only the specified token
            Abar_token = Abar[token_idx]
            
            # Compute beta
            beta_last = Abar_token * C_last.unsqueeze(0).unsqueeze(0)
            
            # Store result (move to CPU to save GPU memory)
            results[layer_idx]['beta'] = beta_last.cpu()
            
            # Cleanup
            del zxbcdt, xBC, C, dt, dt_expanded, A_expanded, discrete_A_scalar, discrete_A
            del A_vals, C_last, log_A, log_A_cumsum, log_A_cumsum_last, log_A_cumsum_j, Abar
            torch.cuda.empty_cache()
        
        return extract_hook
    
    # Register hooks for all layers
    hooks = []
    for layer_idx in range(num_layers):
        target_layer = model.backbone.layers[layer_idx].mixer
        hook = target_layer.register_forward_hook(make_extract_hook(layer_idx))
        hooks.append(hook)
    
    # Single forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Stack results
    beta_all = torch.stack([r['beta'] for r in results], dim=0)  # [num_layers, nheads, headdim, d_state]
    
    return beta_all


def extract_alpha_all(model, tokenizer, prompt, device="cuda:0"):
    """
    Extract alpha attention weights from all layers in a single forward pass.
    Only extracts attention to the LAST token (memory-efficient).
    
    Args:
        model: Mamba2 model
        tokenizer: tokenizer
        prompt: input text (string)
        device: device to use
    
    Returns:
        alphas: [num_layers, seqlen, nheads, headdim] - alpha for LAST token in all layers
    """
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    num_layers = len(model.backbone.layers)
    results = {i: None for i in range(num_layers)}
    
    def make_extract_hook(layer_idx):
        def extract_hook(module, input, output):
            hidden_states = input[0]
            batch, seqlen, _ = hidden_states.shape
            last_idx = seqlen - 1
            
            # Project and split
            zxbcdt = module.in_proj(hidden_states)
            d_mlp = (zxbcdt.shape[-1] - 2 * module.d_ssm - 2 * module.ngroups * module.d_state - module.nheads) // 2
            
            # Extract only what we need: xBC and dt
            _, _, _, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, module.d_ssm, module.d_ssm + 2 * module.ngroups * module.d_state, module.nheads],
                dim=-1
            )
            
            # Conv1d
            if module.activation in ["silu", "swish"]:
                xBC = module.act(module.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
            
            # Split into B and C
            _, B, C = torch.split(
                xBC, 
                [module.d_ssm, module.ngroups * module.d_state, module.ngroups * module.d_state], 
                dim=-1
            )
            
            # Reshape
            B = rearrange(B, "b l (g n) -> b l g n", g=module.ngroups)
            C = rearrange(C, "b l (g n) -> b l g n", g=module.ngroups)
            
            # Process dt
            dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))
            
            # Get A
            A = -torch.exp(module.A_log.float())
            
            # Compute discrete_A and discrete_B efficiently
            dt_expanded = dt.unsqueeze(-1)
            A_expanded = A.view(1, 1, -1, 1)
            
            discrete_A_scalar = torch.exp(A_expanded * dt_expanded)
            discrete_A = discrete_A_scalar.unsqueeze(-1).expand(
                batch, seqlen, module.nheads, module.headdim, module.d_state
            )
            
            # discrete_B
            dt_for_B = dt.unsqueeze(-1).unsqueeze(-1)
            B_expanded = B.unsqueeze(2)
            discrete_B = dt_for_B * B_expanded
            
            # Get D
            D = module.D
            
            # Calculate alpha for LAST token only (assume batch=1, ngroups=1)
            A_vals = discrete_A[0].float()
            B_vals = discrete_B[0, :, :, 0, :].float()
            C_last = C[0, last_idx, 0, :].float()  # Only last token's C
            
            # Compute cumulative products from each token j to last token
            log_A = torch.log(A_vals + 1e-10)
            log_A_cumsum = torch.cumsum(log_A, dim=0)
            
            log_A_cumsum_last = log_A_cumsum[last_idx]
            log_A_cumsum_j = log_A_cumsum
            
            Abar = torch.exp(log_A_cumsum_last.unsqueeze(0) - log_A_cumsum_j)
            
            # Compute effective B
            B_eff = Abar * B_vals.unsqueeze(2)
            
            # Compute alpha for last token
            alpha_last = torch.einsum('n,jhpn->jhp', C_last, B_eff)
            
            # Add D to diagonal (only for last token)
            if D is not None:
                if D.dim() == 1:
                    alpha_last[last_idx, :, :] += D.unsqueeze(-1)
                else:
                    alpha_last[last_idx, :, :] += D
            
            # Store result
            results[layer_idx] = alpha_last.cpu()  # Move to CPU to save GPU memory
            
            # Cleanup
            del zxbcdt, xBC, dt, B, C, dt_expanded, A_expanded, discrete_A_scalar, discrete_A
            del dt_for_B, B_expanded, discrete_B, A_vals, B_vals, C_last
            del log_A, log_A_cumsum, log_A_cumsum_last, log_A_cumsum_j, Abar, B_eff
        
        return extract_hook
    
    # Register hooks for all layers
    hook_handles = []
    for layer_idx in range(num_layers):
        target_layer = model.backbone.layers[layer_idx].mixer
        hook_handle = target_layer.register_forward_hook(make_extract_hook(layer_idx))
        hook_handles.append(hook_handle)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Cleanup hooks
    for hook_handle in hook_handles:
        hook_handle.remove()
    
    # Stack results
    alphas = torch.stack([results[i] for i in range(num_layers)], dim=0)
    
    return alphas


def extract_alpha_batch(model, tokenizer, prompts, device="cuda:0", layer_idx=0):
    """
    Extract alpha matrices for multiple prompts efficiently.
    Only extracts attention to the LAST token (memory-efficient).
    
    Args:
        model: Mamba2 model
        tokenizer: tokenizer
        prompts: list of input texts
        device: device to use
        layer_idx: layer index to extract from
    
    Returns:
        alphas: list of [seqlen, nheads, headdim] - alpha for LAST token in each prompt
    """
    alphas = []
    for prompt in prompts:
        alpha = extract_alpha(model, tokenizer, prompt, device, layer_idx)
        alphas.append(alpha.cpu())  # Move to CPU to save GPU memory
        torch.cuda.empty_cache()
    
    return alphas


if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    
    print("="*80)
    print("Fast Alpha Extraction Test")
    print("="*80)
    
    device = "cuda:0"
    dtype = torch.float32
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=device, dtype=dtype)
    model.eval()
    
    # Test single extraction
    prompt = "Hello world! This is a test."
    print(f"\nPrompt: '{prompt}'")
    print("Extracting alpha matrix...")
    
    start_time = time.time()
    alpha = extract_alpha(model, tokenizer, prompt, device=device, layer_idx=0)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Extraction complete!")
    print(f"   Time: {elapsed:.4f} seconds")
    print(f"   Alpha shape: {alpha.shape}")
    print(f"   Alpha[0,0,0,0]: {alpha[0,0,0,0]:.6f}")
    print(f"   Alpha[1,0,0,0]: {alpha[1,0,0,0]:.6f}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    # Test batch extraction
    print(f"\n{'='*80}")
    print("Testing batch extraction...")
    prompts = ["Hello world", "Test prompt 2", "Another example text"]
    
    start_time = time.time()
    alphas = extract_alpha_batch(model, tokenizer, prompts, device=device, layer_idx=0)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Batch extraction complete!")
    print(f"   Time: {elapsed:.4f} seconds ({elapsed/len(prompts):.4f} sec/prompt)")
    print(f"   Number of alphas: {len(alphas)}")
    print(f"   Shapes: {[a.shape for a in alphas]}")
    
    print(f"\n{'='*80}")
    print("✅ All tests passed!")
