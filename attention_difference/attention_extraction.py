import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def extract_abc(model, tokenizer, prompt, layer_idx=0, device="cuda:0"):
    """
    Extract discrete A, B, C matrices from a Mamba layer.
    
    Returns: dict with keys [discrete_A, discrete_B, C]
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    extracted = {}
    
    def extract_hook(module, input, output):
        hidden_states = input[0]
        batch, seqlen, dim = hidden_states.shape
        
        # Get x projection
        xz = rearrange(
            module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if module.in_proj.bias is not None:
            xz = xz + rearrange(module.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        x, z = xz.chunk(2, dim=1)
        
        if module.activation in ["silu", "swish"]:
            x = F.silu(module.conv1d(x)[..., :seqlen])
        
        # Get dt, B, C
        x_dbl = module.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1)
        
        dt = module.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        if module.dt_proj.bias is not None:
            dt = dt + module.dt_proj.bias.float().view(1, -1, 1)
        dt = F.softplus(dt)
        
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # Compute discrete A and B
        A = -torch.exp(module.A_log.float())
        A_expanded = A[None, :, None, :]
        dt_expanded = dt[:, :, :, None]
        
        discrete_A = torch.exp(A_expanded * dt_expanded)
        discrete_B = dt_expanded * B[:, None, :, :].transpose(2, 3)
        
        extracted['discrete_A'] = discrete_A
        extracted['discrete_B'] = discrete_B
        extracted['C'] = C
        
        # Clean up
        del xz, x, z, x_dbl, dt, B, A, A_expanded, dt_expanded
    
    target_layer = model.backbone.layers[layer_idx].mixer
    hook_handle = target_layer.register_forward_hook(extract_hook)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    hook_handle.remove()
    
    return extracted

def extract_abc_all_layers_fast(model, tokenizer, prompt, device="cuda:0", keep_on_gpu=False):
    """
    Extract discrete A, B, C matrices from ALL layers in a SINGLE forward pass.
    Much faster than calling extract_abc 64 times.
    
    Args:
        model: Mamba model
        tokenizer: tokenizer
        prompt: input text OR list of texts (for batch processing)
        device: device to use
        keep_on_gpu: if True, keep data on GPU; if False, move to CPU
    
    Returns: list of 64 dicts, each with keys [discrete_A, discrete_B, C]
    """
    # Handle both single and batch inputs
    if isinstance(prompt, str):
        tokens = tokenizer(prompt, return_tensors="pt")
    else:  # List of prompts (batch)
        tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
    
    input_ids = tokens['input_ids'].to(device)
    
    extracted_all = []
    hooks = []
    
    def make_extract_hook(layer_idx):
        def extract_hook(module, input, output):
            hidden_states = input[0]
            batch, seqlen, dim = hidden_states.shape
            
            # Get x projection
            xz = rearrange(
                module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if module.in_proj.bias is not None:
                xz = xz + rearrange(module.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            
            x, z = xz.chunk(2, dim=1)
            
            if module.activation in ["silu", "swish"]:
                x = F.silu(module.conv1d(x)[..., :seqlen])
            
            # Get dt, B, C
            x_dbl = module.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1)
            
            dt = module.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            if module.dt_proj.bias is not None:
                dt = dt + module.dt_proj.bias.float().view(1, -1, 1)
            dt = F.softplus(dt)
            
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            
            # Compute discrete A and B
            A = -torch.exp(module.A_log.float())
            A_expanded = A[None, :, None, :]
            dt_expanded = dt[:, :, :, None]
            
            discrete_A = torch.exp(A_expanded * dt_expanded)
            discrete_B = dt_expanded * B[:, None, :, :].transpose(2, 3)
            
            # Store on GPU or CPU based on flag
            if keep_on_gpu:
                extracted_all[layer_idx] = {
                    'discrete_A': discrete_A,
                    'discrete_B': discrete_B,
                    'C': C
                }
            else:
                extracted_all[layer_idx] = {
                    'discrete_A': discrete_A.cpu(),
                    'discrete_B': discrete_B.cpu(),
                    'C': C.cpu()
                }
                # Clean up GPU memory only if moving to CPU
                del discrete_A, discrete_B, C
            
            # Always clean up intermediate variables
            del xz, x, z, x_dbl, dt, B, A, A_expanded, dt_expanded
        
        return extract_hook
    
    # Register hooks for all 64 layers
    num_layers = len(model.backbone.layers)
    extracted_all = [None] * num_layers
    
    for layer_idx in range(num_layers):
        target_layer = model.backbone.layers[layer_idx].mixer
        hook = target_layer.register_forward_hook(make_extract_hook(layer_idx))
        hooks.append(hook)
    
    # Single forward pass extracts all layers
    with torch.no_grad():
        _ = model(input_ids)
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    return extracted_all

def extract_cummulated_Abar_right(discrete_A):
    """
    Perform the function A => A.scanRight(_ elem-wise product _)
    
    Computes: Abar[b, ch, i, s] = ∏_{j=i+1}^{seqlen-1} A[b, ch, j, s]
    
    This is the cumulative product from position i+1 to the last position (seqlen-1).
    Used to compute attention scores from each token to the LAST token:
      - Abar * C[last_token] → beta (state propagation weights)
      - beta * B[token] → alpha (final attention score)
    
    Args:
        discrete_A: [batch, d_inner, seqlen, d_state]
    
    Returns:
        Abar: [batch, d_inner, seqlen, d_state]
              Abar[i] = A[i+1] * A[i+2] * ... * A[seqlen-1]
              Abar[seqlen-1] = 1 (no tokens after)
    """
    # Use log-space to avoid numerical issues
    log_A = torch.log(discrete_A + 1e-10)
    
    # Cumulative sum from right (reverse cumsum)
    # Step 1: flip along seqlen dimension
    log_A_flipped = torch.flip(log_A, dims=[2])
    
    # Step 2: cumsum (now accumulating from what was the right)
    log_cumsum_flipped = torch.cumsum(log_A_flipped, dim=2)
    
    # Step 3: flip back
    log_cumsum = torch.flip(log_cumsum_flipped, dims=[2])
    
    # Now log_cumsum[i] = sum(log_A[i:]) = log(∏_{j=i}^{seqlen-1} A[j])
    # We want log(∏_{j=i+1}^{seqlen-1} A[j]) = log_cumsum[i] - log_A[i]
    log_Abar = log_cumsum - log_A
    
    # Convert back from log-space
    Abar = torch.exp(log_Abar)
    
    # Clean up
    del log_A, log_A_flipped, log_cumsum_flipped, log_cumsum, log_Abar
    return Abar

def extract_beta_last(cumulated_Abar, C, tokens):
    """
    Extract beta (state propagation weights) from specified tokens to the LAST token.
    
    Formula: beta[j, ch, s] = Abar[ch, j, s] * C[s, last]
    
    This represents how state dimension s at token j propagates to the output at the last token,
    for each channel ch.
    
    Args:
        cumulated_Abar: [batch, d_inner, seqlen, d_state] - from extract_cummulated_Abar_right
        C: [batch, d_state, seqlen] - C matrix
        tokens: List[int] - token positions to extract beta for
    
    Returns:
        beta: [len(tokens), d_inner, d_state]
              beta[i, ch, s] = state propagation weight from tokens[i] to last token
    """
    batch, d_inner, seqlen, d_state = cumulated_Abar.shape
    last_idx = seqlen - 1
    
    # Get C for the last token: [d_state]
    C_last = C[0, :, last_idx].float()  # [d_state]
    
    # Get Abar for specified tokens: [len(tokens), d_inner, d_state]
    Abar_selected = cumulated_Abar[0, :, tokens, :].float()  # [d_inner, len(tokens), d_state]
    Abar_selected = Abar_selected.permute(1, 0, 2)  # [len(tokens), d_inner, d_state]
    
    # Compute beta: element-wise multiply with C_last
    # beta[i, ch, s] = Abar[ch, tokens[i], s] * C[s, last]
    beta = Abar_selected * C_last[None, None, :]  # [len(tokens), d_inner, d_state]
    
    return beta

def extract_alpha_last(cumulated_Abar, discrete_B, C, tokens):
    """
    Extract alpha (attention scores) from specified tokens to the LAST token.
    
    Formula: alpha[j, ch] = sum_s (Abar[ch, j, s] * B[ch, j, s] * C[s, last])
    
    This represents the attention weight from token j to the last token, for each channel ch.
    
    Args:
        cumulated_Abar: [batch, d_inner, seqlen, d_state] - from extract_cummulated_Abar_right
        discrete_B: [batch, d_inner, seqlen, d_state] - discrete B matrix
        C: [batch, d_state, seqlen] - C matrix
        tokens: List[int] - token positions to extract alpha for
    
    Returns:
        alpha: [len(tokens), d_inner]
               alpha[i, ch] = attention score from tokens[i] to last token
    """
    batch, d_inner, seqlen, d_state = cumulated_Abar.shape
    last_idx = seqlen - 1
    
    # Get C for the last token: [d_state]
    C_last = C[0, :, last_idx].float()  # [d_state]
    
    # Get Abar and B for specified tokens
    Abar_selected = cumulated_Abar[0, :, tokens, :].float()  # [d_inner, len(tokens), d_state]
    B_selected = discrete_B[0, :, tokens, :].float()  # [d_inner, len(tokens), d_state]
    
    # Compute alpha: einsum over state dimension
    # alpha[i, ch] = sum_s (Abar[ch, i, s] * B[ch, i, s] * C[s])
    alpha = torch.einsum('cjs,cjs,s->jc', Abar_selected, B_selected, C_last)  # [len(tokens), d_inner]
    
    return alpha


def calculate_alpha_all_channels(discrete_A, discrete_B, C):
    """
    Calculate alpha attention matrix for all channels (vectorized).
    
    α_{i,j,ch} = Σ_s C_{i,s} * (∏_{k=j+1}^{i} A_{k,ch,s}) * discrete_B_{j,ch,s}
    
    Note: Original B and C are shared across all channels, but discrete_B = dt * B
    is channel-specific since dt varies per channel. Only A and dt are truly independent.
    
    Returns: [seqlen, seqlen, d_inner] where alpha[i,j,ch] is a scalar weight.
    """
    A = discrete_A[0].float()
    B = discrete_B[0].float()
    C_mat = C[0].float()
    
    d_inner, seq_len, d_state = A.shape
    
    log_A = torch.log(A + 1e-10)
    log_A_cumsum = torch.cumsum(log_A, dim=1)
    
    log_A_cumsum_i = log_A_cumsum[:, :, None, :]
    log_A_cumsum_j = log_A_cumsum[:, None, :, :]
    
    log_A_cumprod = log_A_cumsum_i - log_A_cumsum_j
    A_cumprod = torch.exp(log_A_cumprod)
    
    mask = torch.triu(torch.ones(seq_len, seq_len, device=A.device, dtype=torch.bool), diagonal=1)
    A_cumprod[:, mask, :] = 0
    
    # Release intermediate tensors
    del log_A, log_A_cumsum, log_A_cumsum_i, log_A_cumsum_j, log_A_cumprod, mask
    
    B_effective = A_cumprod * B[:, None, :, :]
    del A_cumprod  # Release after creating B_effective
    
    alpha_matrix = torch.einsum('si,cijs->ijc', C_mat, B_effective)
    del B_effective  # Release after einsum
    
    diagonal_alpha = torch.einsum('si,cis->ic', C_mat, B)
    alpha_matrix.diagonal(dim1=0, dim2=1).copy_(diagonal_alpha.T)
    
    # Release remaining tensors
    del A, B, C_mat, diagonal_alpha
    
    return alpha_matrix

def calculate_alpha_last_tokens(discrete_A, discrete_B, C, last_tokens):
    """
    Calculate alpha attention matrix for ONLY the last N tokens (memory efficient version).
    
    This function computes attention only for the last `last_tokens` positions as queries,
    while keeping all positions as keys. This reduces memory from O(seqlen²) to O(last_tokens × seqlen).
    
    α_{i,j,ch} = Σ_s C_{i,s} * (∏_{k=j+1}^{i} A_{k,ch,s}) * discrete_B_{j,ch,s}
    
    Args:
        discrete_A: [batch, d_inner, seqlen, d_state]
        discrete_B: [batch, d_inner, seqlen, d_state]
        C: [batch, d_state, seqlen]
        last_tokens: Number of last tokens to compute attention for (as queries)
    
    Returns: 
        [last_tokens, seqlen, d_inner] where alpha[i,j,ch] represents attention from 
        query position (seqlen-last_tokens+i) to key position j for channel ch.
        
    Example:
        If seqlen=651 and last_tokens=100:
        - Output shape: [100, 651, 5120]
        - Output[0] = attention from position 551 to all positions
        - Output[99] = attention from position 650 to all positions
    """
    A = discrete_A[0].float()
    B = discrete_B[0].float()
    C_mat = C[0].float()
    
    d_inner, seq_len, d_state = A.shape
    
    # Determine query positions (last N tokens)
    query_len = min(last_tokens, seq_len)
    query_start = seq_len - query_len
    
    log_A = torch.log(A + 1e-10)
    log_A_cumsum = torch.cumsum(log_A, dim=1)
    
    # Only compute for selected query positions
    log_A_cumsum_i = log_A_cumsum[:, query_start:, None, :]  # [d_inner, query_len, 1, d_state]
    log_A_cumsum_j = log_A_cumsum[:, None, :, :]              # [d_inner, 1, seq_len, d_state]
    
    log_A_cumprod = log_A_cumsum_i - log_A_cumsum_j
    A_cumprod = torch.exp(log_A_cumprod)  # [d_inner, query_len, seq_len, d_state]
    
    # Mask: prevent attending to future tokens
    mask = torch.zeros(query_len, seq_len, device=A.device, dtype=torch.bool)
    for i_rel in range(query_len):
        i_abs = query_start + i_rel
        mask[i_rel, i_abs+1:] = True
    A_cumprod[:, mask, :] = 0
    
    B_effective = A_cumprod * B[:, None, :, :]  # [d_inner, query_len, seq_len, d_state]
    
    # Release large intermediate tensors
    del log_A, log_A_cumsum, log_A_cumsum_i, log_A_cumsum_j, log_A_cumprod, A_cumprod, mask
    
    # Compute alpha for selected query positions
    C_selected = C_mat[:, query_start:]  # [d_state, query_len]
    alpha_matrix = torch.einsum('si,cijs->ijc', C_selected, B_effective)  # [query_len, seq_len, d_inner]
    
    # Release B_effective after einsum
    del B_effective, C_selected
    
    # Diagonal elements (i == j)
    for i_rel in range(query_len):
        i_abs = query_start + i_rel
        diagonal_alpha = torch.einsum('s,cs->c', C_mat[:, i_abs], B[:, i_abs, :])  # [d_inner]
        alpha_matrix[i_rel, i_abs, :] = diagonal_alpha
    
    # Release remaining large tensors
    del A, B, C_mat
    
    return alpha_matrix

def calculate_beta_all_channels(discrete_A, C):
    """
    Calculate beta state propagation matrix for all channels (vectorized).
    
    β_{i,j,ch,s} = C_{i,s} * (∏_{k=j+1}^{i} A_{k,ch,s})
    
    Represents how each dimension of state h_j propagates to output y_i at channel ch.
    Relation: α_{i,j,ch} = Σ_s β_{i,j,ch,s} * discrete_B_{j,ch,s}
    
    Note: C is shared across all channels, but beta is channel-specific due to
    channel-specific A (state transition). discrete_A = exp(A * dt) varies per channel.
    
    Returns: [seqlen, seqlen, d_inner, d_state] where beta[i,j,ch,:] is a d_state-dimensional vector.
    """
    A = discrete_A[0].float()
    C_mat = C[0].float()
    
    d_inner, seq_len, d_state = A.shape
    
    log_A = torch.log(A + 1e-10)
    log_A_cumsum = torch.cumsum(log_A, dim=1)
    
    log_A_cumsum_i = log_A_cumsum[:, :, None, :]
    log_A_cumsum_j = log_A_cumsum[:, None, :, :]
    
    log_A_cumprod = log_A_cumsum_i - log_A_cumsum_j
    A_cumprod = torch.exp(log_A_cumprod)
    
    mask = torch.triu(torch.ones(seq_len, seq_len, device=A.device, dtype=torch.bool), diagonal=1)
    A_cumprod[:, mask, :] = 0
    
    # β[i,j,ch] = C[i] * A_cumprod[ch,i,j]
    # Shape: [seq_len, seq_len, d_inner, d_state]
    beta_matrix = torch.einsum('si,cijs->ijcs', C_mat, A_cumprod)
    
    # Diagonal: no accumulation
    diagonal_beta = C_mat.T.unsqueeze(1).expand(seq_len, d_inner, d_state)  # [seqlen, d_inner, d_state]
    for i in range(seq_len):
        beta_matrix[i, i, :, :] = diagonal_beta[i, :, :]
    
    return beta_matrix


if __name__ == "__main__":
    # Simple test - no arguments needed
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device=device, dtype=dtype)
    model.eval()
    
    # Test with SHORT sequence for verification
    prompt = "Hello world! This is a test."
    
    print()
    
    result = extract_abc(model, tokenizer, prompt, layer_idx=0, device=device)
    seq_len = result['discrete_A'].shape[2]
    print(f"Sequence length: {seq_len} tokens")
    print(f"Shape of discrete_A: {result['discrete_A'].shape}")
    print(f"Shape of discrete_B: {result['discrete_B'].shape}")
    print(f"Shape of C: {result['C'].shape}")
    
    # Test extract_cummulated_Abar_right
    print("\n" + "="*80)
    print("Testing extract_cummulated_Abar_right (O(n) method)...")
    print("="*80)
    Abar = extract_cummulated_Abar_right(result['discrete_A'])
    print(f"Abar shape: {list(Abar.shape)}")
    
    # Compute attention to last token using Abar (O(n) complexity!)
    last_idx = seq_len - 1
    C_last = result['C'][0, :, last_idx].float()  # [d_state]
    discrete_B_data = result['discrete_B'][0].float()  # [d_inner, seqlen, d_state]
    Abar_data = Abar[0].float()  # [d_inner, seqlen, d_state]
    
    # alpha[j, ch] = sum_s (Abar[ch, j, s] * B[ch, j, s] * C_last[s])
    alpha_to_last_fast = torch.einsum('cjs,cjs,s->jc', Abar_data, discrete_B_data, C_last)  # [seqlen, d_inner]
    print(f"Alpha (all tokens → last token) shape: {list(alpha_to_last_fast.shape)}")
    
    # Verify against the trusted method from extract_ABC.py
    print("\n" + "="*80)
    print("Verifying against trusted iterative method (extract_ABC.py style)...")
    print("="*80)
    
    A = result['discrete_A'][0].float()  # [d_inner, seqlen, d_state]
    B = result['discrete_B'][0].float()  # [d_inner, seqlen, d_state]
    C = result['C'][0].float()  # [d_state, seqlen]
    d_inner, seq_len_check, d_state = A.shape
    
    # Calculate alpha for the LAST token (i = last_idx) using verified method
    alpha_to_last_verified = torch.zeros(seq_len, d_inner, device=A.device, dtype=A.dtype)
    
    for ch in range(d_inner):
        # For output position i = last_idx
        i = last_idx
        A_prod = torch.ones(d_state, device=A.device, dtype=A.dtype)
        
        # alpha[i, i] = C[i] · B[ch, i]
        alpha_ii = torch.dot(C[:, i], B[ch, i, :])
        alpha_to_last_verified[i, ch] = alpha_ii
        
        # alpha[i, j] for j < i
        for j in range(i - 1, -1, -1):
            # A_prod *= A[ch, j+1]
            A_prod = A_prod * A[ch, j + 1, :]
            # alpha[i, j] = C[i] · (A_prod * B[ch, j])
            B_effective = A_prod * B[ch, j, :]
            alpha_ij = torch.dot(C[:, i], B_effective)
            alpha_to_last_verified[j, ch] = alpha_ij
    
    print("Verified alpha calculation complete.")
    
    # Compare results
    print("\n" + "="*80)
    print("Comparison Results:")
    print("="*80)
    diff = (alpha_to_last_fast - alpha_to_last_verified).abs()
    print(f"Max difference: {diff.max():.2e}")
    print(f"Mean difference: {diff.mean():.2e}")
    
    if diff.max() < 1e-4:
        print("✓ PASS: extract_cummulated_Abar_right matches verified method!")
    else:
        print("✗ FAIL: Results differ significantly!")
    
    print(f"\nDiagonal element (last token to itself):")
    print(f"  Fast method: {alpha_to_last_fast[-1, 0]:.6f}")
    print(f"  Verified:    {alpha_to_last_verified[-1, 0]:.6f}")
    
    # Test extract_alpha_last and extract_beta_last
    print("\n" + "="*80)
    print("Testing extract_alpha_last and extract_beta_last...")
    print("="*80)
    
    # Select some tokens to extract (e.g., first, middle, last)
    test_tokens = [0, seq_len // 2, seq_len - 1]
    print(f"Extracting attention for tokens: {test_tokens}")
    
    # Extract alpha and beta for selected tokens
    alpha_selected = extract_alpha_last(Abar, result['discrete_B'], result['C'], test_tokens)
    beta_selected = extract_beta_last(Abar, result['C'], test_tokens)
    
    print(f"\nAlpha shape: {list(alpha_selected.shape)} (expected: [{len(test_tokens)}, {d_inner}])")
    print(f"Beta shape: {list(beta_selected.shape)} (expected: [{len(test_tokens)}, {d_inner}, {d_state}])")
    
    # Verify alpha matches the full computation
    print("\nVerifying alpha_selected matches alpha_to_last_fast:")
    for i, token_idx in enumerate(test_tokens):
        diff = (alpha_selected[i] - alpha_to_last_fast[token_idx]).abs()
        print(f"  Token {token_idx}: max diff = {diff.max():.2e}, mean diff = {diff.mean():.2e}")
    
    # Verify beta relationship: alpha = sum_s(beta * B)
    print("\nVerifying relationship: alpha = sum_s(beta * B):")
    for i, token_idx in enumerate(test_tokens):
        # Reconstruct alpha from beta
        B_token = result['discrete_B'][0, :, token_idx, :].float()  # [d_inner, d_state]
        alpha_reconstructed = (beta_selected[i] * B_token).sum(dim=-1)  # [d_inner]
        diff = (alpha_reconstructed - alpha_selected[i]).abs()
        print(f"  Token {token_idx}: max diff = {diff.max():.2e}, mean diff = {diff.mean():.2e}")
    
    print("\n✓ All tests complete!")



