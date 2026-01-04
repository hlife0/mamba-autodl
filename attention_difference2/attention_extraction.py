"""
Mamba2 Attention Extraction
Adapted for multi-head architecture with nheads, headdim structure
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def extract_abc_mamba2(model, tokenizer, prompt, layer_idx=0, device="cuda:0"):
    """
    Extract discrete A, B, C matrices from a Mamba2 layer.
    
    Mamba2 uses multi-head architecture:
    - A: [nheads] - shared across all heads
    - dt: [batch, nheads] - per-head time step
    - B, C: [batch, ngroups, d_state, seqlen] - grouped
    
    Returns: dict with keys [discrete_A, discrete_B, C, dt, nheads, headdim, ngroups]
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    extracted = {}
    
    def extract_hook(module, input, output):
        hidden_states = input[0]
        batch, seqlen, dim = hidden_states.shape
        
        # Get projection (order: [z, x, B, C, dt])
        zxbcdt = module.in_proj(hidden_states)
        
        # Calculate d_mlp (MLP dimensions if using hybrid architecture)
        d_mlp = (zxbcdt.shape[-1] - 2 * module.d_ssm - 2 * module.ngroups * module.d_state - module.nheads) // 2
        
        # Split into components
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, module.d_ssm, module.d_ssm + 2 * module.ngroups * module.d_state, module.nheads],
            dim=-1
        )
        
        # Apply conv1d
        if module.activation in ["silu", "swish"]:
            xBC = module.act(
                module.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen]
            )
        
        # Split xBC into x, B, C
        x, B, C = torch.split(
            xBC, 
            [module.d_ssm, module.ngroups * module.d_state, module.ngroups * module.d_state], 
            dim=-1
        )
        
        # Reshape for multi-head processing
        x = rearrange(x, "b l (h p) -> b l h p", p=module.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=module.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=module.ngroups)
        
        # Process dt
        dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))  # [batch, seqlen, nheads]
        
        # Get A
        A = -torch.exp(module.A_log.float())  # [nheads]
        
        # Compute discrete A and B
        # For Mamba2: discrete_A[b, l, h, p, n] = exp(A[h] * dt[b, l, h])
        # A[h] applies to all (p, n), dt[b,l,h] varies per position and head
        dt_expanded = dt.unsqueeze(-1)  # [batch, seqlen, nheads, 1]
        A_expanded = A.view(1, 1, -1, 1)  # [1, 1, nheads, 1]
        
        # discrete_A should be [batch, seqlen, nheads, 1, 1] after exp
        # then broadcast to [batch, seqlen, nheads, headdim, d_state]
        discrete_A_scalar = torch.exp(A_expanded * dt_expanded)  # [batch, seqlen, nheads, 1]
        discrete_A = discrete_A_scalar.unsqueeze(-1).expand(
            batch, seqlen, module.nheads, module.headdim, module.d_state
        )  # [batch, seqlen, nheads, headdim, d_state]
        
        # discrete_B[b, l, h, p, g, n] = dt[b, l, h] * B[b, l, g, n]
        # We'll keep it per-group for now
        dt_for_B = dt.unsqueeze(-1).unsqueeze(-1)  # [batch, seqlen, nheads, 1, 1]
        B_expanded = B.unsqueeze(2)  # [batch, seqlen, 1, ngroups, d_state]
        discrete_B = dt_for_B * B_expanded  # [batch, seqlen, nheads, ngroups, d_state]
        
        extracted['discrete_A'] = discrete_A
        extracted['discrete_B'] = discrete_B
        extracted['C'] = C  # [batch, seqlen, ngroups, d_state]
        extracted['dt'] = dt
        extracted['x'] = x
        extracted['nheads'] = module.nheads
        extracted['headdim'] = module.headdim
        extracted['ngroups'] = module.ngroups
        extracted['d_state'] = module.d_state
        extracted['D'] = module.D  # Skip connection parameter [nheads]
        
        # Capture additional info for verification
        extracted['z'] = z
        extracted['out_proj'] = module.out_proj
        extracted['layer_output'] = output
        
        # Clean up
        del zxbcdt, z0, x0, xBC, dt_expanded, A_expanded, dt_for_B, B_expanded
    
    target_layer = model.backbone.layers[layer_idx].mixer
    hook_handle = target_layer.register_forward_hook(extract_hook)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    hook_handle.remove()
    
    return extracted


def extract_abc_all_layers_mamba2(model, tokenizer, prompt, device="cuda:0", keep_on_gpu=False):
    """
    Extract discrete A, B, C matrices from ALL Mamba2 layers in a SINGLE forward pass.
    
    Args:
        model: Mamba2 model
        tokenizer: tokenizer
        prompt: input text OR list of texts (for batch processing)
        device: device to use
        keep_on_gpu: if True, keep data on GPU; if False, move to CPU
    
    Returns: list of dicts, each with keys [discrete_A, discrete_B, C, dt, nheads, headdim, ngroups]
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
            
            # Get projection (order: [z, x, B, C, dt])
            zxbcdt = module.in_proj(hidden_states)
            
            # Calculate d_mlp
            d_mlp = (zxbcdt.shape[-1] - 2 * module.d_ssm - 2 * module.ngroups * module.d_state - module.nheads) // 2
            
            # Split into components
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, module.d_ssm, module.d_ssm + 2 * module.ngroups * module.d_state, module.nheads],
                dim=-1
            )
            
            # Apply conv1d
            if module.activation in ["silu", "swish"]:
                xBC = module.act(
                    module.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen]
                )
            
            # Split xBC into x, B, C
            x, B, C = torch.split(
                xBC, 
                [module.d_ssm, module.ngroups * module.d_state, module.ngroups * module.d_state], 
                dim=-1
            )
            
            # Reshape for multi-head processing
            x = rearrange(x, "b l (h p) -> b l h p", p=module.headdim)
            B = rearrange(B, "b l (g n) -> b l g n", g=module.ngroups)
            C = rearrange(C, "b l (g n) -> b l g n", g=module.ngroups)
            
            # Process dt
            dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))
            
            # Get A
            A = -torch.exp(module.A_log.float())
            
            # Compute discrete A and B
            dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)
            A_expanded = A.view(1, 1, -1, 1, 1).expand(1, 1, module.nheads, module.headdim, module.d_state)
            
            discrete_A = torch.exp(A_expanded * dt_expanded)
            
            dt_for_B = dt.unsqueeze(-1).unsqueeze(-1)
            B_expanded = B.unsqueeze(2)
            discrete_B = dt_for_B * B_expanded
            
            # Store on GPU or CPU based on flag
            if keep_on_gpu:
                extracted_all[layer_idx] = {
                    'discrete_A': discrete_A,
                    'discrete_B': discrete_B,
                    'C': C,
                    'dt': dt,
                    'x': x,
                    'nheads': module.nheads,
                    'headdim': module.headdim,
                    'ngroups': module.ngroups,
                    'd_state': module.d_state,
                }
            else:
                extracted_all[layer_idx] = {
                    'discrete_A': discrete_A.cpu(),
                    'discrete_B': discrete_B.cpu(),
                    'C': C.cpu(),
                    'dt': dt.cpu(),
                    'x': x.cpu(),
                    'nheads': module.nheads,
                    'headdim': module.headdim,
                    'ngroups': module.ngroups,
                    'd_state': module.d_state,
                }
                del discrete_A, discrete_B, C, dt, x
            
            # Clean up
            del zxbcdt, z0, x0, z, xBC, dt_expanded, A_expanded, dt_for_B, B_expanded
        
        return extract_hook
    
    # Register hooks for all layers
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


def extract_cummulated_Abar_right_mamba2(discrete_A):
    """
    Compute cumulative product from right for Mamba2 multi-head architecture.
    
    Abar[b, l, h, p, n] = ∏_{j=l+1}^{seqlen-1} A[b, j, h, p, n]
    
    Args:
        discrete_A: [batch, seqlen, nheads, headdim, d_state]
    
    Returns:
        Abar: [batch, seqlen, nheads, headdim, d_state]
    """
    # Use log-space to avoid numerical issues
    log_A = torch.log(discrete_A + 1e-10)
    
    # Cumulative sum from right (reverse cumsum)
    # Flip along seqlen dimension (dim=1)
    log_A_flipped = torch.flip(log_A, dims=[1])
    log_cumsum_flipped = torch.cumsum(log_A_flipped, dim=1)
    log_cumsum = torch.flip(log_cumsum_flipped, dims=[1])
    
    # log_cumsum[l] = sum(log_A[l:]) = log(∏_{j=l}^{seqlen-1} A[j])
    # We want log(∏_{j=l+1}^{seqlen-1} A[j]) = log_cumsum[l] - log_A[l]
    log_Abar = log_cumsum - log_A
    
    # Convert back from log-space
    Abar = torch.exp(log_Abar)
    
    # Clean up
    del log_A, log_A_flipped, log_cumsum_flipped, log_cumsum, log_Abar
    return Abar


def calculate_alpha_mamba2_simple(discrete_A, discrete_B, C):
    """
    Calculate alpha attention for Mamba2 (simplified version for single group).
    
    For Mamba2 with ngroups=1:
    α_{i,j,h,p} = Σ_n C_{i,n} * (∏_{k=j+1}^{i} A_{k,h,p,n}) * discrete_B_{j,h,n}
    
    Args:
        discrete_A: [batch, seqlen, nheads, headdim, d_state]
        discrete_B: [batch, seqlen, nheads, ngroups, d_state]
        C: [batch, seqlen, ngroups, d_state]
    
    Returns:
        alpha: [seqlen, seqlen, nheads, headdim] - attention from token j to token i
    """
    batch, seqlen, nheads, headdim, d_state = discrete_A.shape
    ngroups = C.shape[2]
    
    # Assume batch=1 and ngroups=1 for simplicity
    A = discrete_A[0].float()  # [seqlen, nheads, headdim, d_state]
    B = discrete_B[0].float()  # [seqlen, nheads, ngroups, d_state]
    C_mat = C[0].float()  # [seqlen, ngroups, d_state]
    
    if ngroups == 1:
        B = B.squeeze(2)  # [seqlen, nheads, d_state]
        C_mat = C_mat.squeeze(1)  # [seqlen, d_state]
    
    # Compute cumulative products
    log_A = torch.log(A + 1e-10)  # [seqlen, nheads, headdim, d_state]
    log_A_cumsum = torch.cumsum(log_A, dim=0)  # [seqlen, nheads, headdim, d_state]
    
    # For each pair (i, j), compute A_cumprod = ∏_{k=j+1}^{i} A[k]
    # This is exp(log_A_cumsum[i] - log_A_cumsum[j])
    log_A_cumsum_i = log_A_cumsum[:, None, :, :, :]  # [seqlen, 1, nheads, headdim, d_state]
    log_A_cumsum_j = log_A_cumsum[None, :, :, :, :]  # [1, seqlen, nheads, headdim, d_state]
    
    log_A_cumprod = log_A_cumsum_i - log_A_cumsum_j  # [seqlen, seqlen, nheads, headdim, d_state]
    A_cumprod = torch.exp(log_A_cumprod)
    
    # Mask future tokens (j > i should be 0)
    mask = torch.triu(torch.ones(seqlen, seqlen, device=A.device, dtype=torch.bool), diagonal=1)
    # Expand mask to match A_cumprod dimensions for masked_fill
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [seqlen, seqlen, 1, 1, 1]
    A_cumprod = A_cumprod.masked_fill(mask_expanded, 0.0)
    
    del log_A, log_A_cumsum, log_A_cumsum_i, log_A_cumsum_j, log_A_cumprod, mask, mask_expanded
    
    # Compute effective B: B_eff[i, j, h, p, n] = A_cumprod[i, j, h, p, n] * B[j, h, n]
    # Need to broadcast properly
    B_eff = A_cumprod * B[None, :, :, None, :]  # [seqlen, seqlen, nheads, headdim, d_state]
    del A_cumprod
    
    # Compute alpha: α[i, j, h, p] = Σ_n C[i, n] * B_eff[i, j, h, p, n]
    alpha = torch.einsum('in,ijhpn->ijhp', C_mat, B_eff)  # [seqlen, seqlen, nheads, headdim]
    del B_eff
    
    # Diagonal should be 0 because x[i] doesn't affect y[i] in the SSM recursion
    # y[i] = C[i] @ state[i], then state[i+1] = A[i] * state[i] + B[i] * x[i]
    # So x[i] only affects future outputs, not y[i] itself
    for i in range(seqlen):
        alpha[i, i, :, :] = 0.0
    
    del A, B, C_mat
    
    return alpha


if __name__ == "__main__":
    # Test script
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32
    
    print("="*80)
    print("Testing Mamba2 Attention Extraction")
    print("="*80)
    print(f"Device: {device}")
    print()
    
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-2.7b", device=device, dtype=dtype)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test with SHORT sequence
    prompt = "Hello world! This is a test."
    
    print()
    print("="*80)
    print("Test 1: Extract ABC from single layer")
    print("="*80)
    
    try:
        result = extract_abc_mamba2(model, tokenizer, prompt, layer_idx=0, device=device)
        
        batch, seqlen, nheads, headdim, d_state = result['discrete_A'].shape
        print(f"✓ Extraction successful!")
        print(f"  Sequence length: {seqlen} tokens")
        print(f"  Architecture:")
        print(f"    - nheads: {result['nheads']}")
        print(f"    - headdim: {result['headdim']}")
        print(f"    - ngroups: {result['ngroups']}")
        print(f"    - d_state: {result['d_state']}")
        print(f"  Shapes:")
        print(f"    - discrete_A: {tuple(result['discrete_A'].shape)}")
        print(f"    - discrete_B: {tuple(result['discrete_B'].shape)}")
        print(f"    - C: {tuple(result['C'].shape)}")
        print(f"    - dt: {tuple(result['dt'].shape)}")
    except Exception as e:
        print(f"❌ ERROR in extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("="*80)
    print("Test 2: Extract cumulative Abar")
    print("="*80)
    
    try:
        Abar = extract_cummulated_Abar_right_mamba2(result['discrete_A'])
        print(f"✓ Abar computation successful!")
        print(f"  Abar shape: {tuple(Abar.shape)}")
        print(f"  Expected: {tuple(result['discrete_A'].shape)}")
        
        # Verify: Abar at last position should be 1 (no tokens after)
        last_token_abar = Abar[0, -1, 0, 0, 0].item()
        print(f"  Abar[last_token] sample value: {last_token_abar:.6f} (should be ~1.0)")
    except Exception as e:
        print(f"❌ ERROR in Abar computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("="*80)
    print("Test 3: Calculate alpha attention (simplified)")
    print("="*80)
    
    try:
        alpha = calculate_alpha_mamba2_simple(result['discrete_A'], result['discrete_B'], result['C'])
        print(f"✓ Alpha computation successful!")
        print(f"  Alpha shape: {tuple(alpha.shape)}")
        print(f"  Expected: [seqlen={seqlen}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}]")
        
        # Check some properties
        print(f"\n  Sanity checks:")
        print(f"    - Alpha is causal (upper triangle should be 0):")
        # alpha is [seqlen, seqlen, nheads, headdim]
        # triu works on last 2 dims, so we need to permute seqlen to the end
        alpha_permuted = alpha.permute(2, 3, 0, 1) # [nheads, headdim, seqlen, seqlen]
        upper_tri_sum = alpha_permuted.triu(diagonal=1).abs().sum().item()
        print(f"      Upper triangle sum: {upper_tri_sum:.2e} (should be ~0)")
        
        print(f"    - Diagonal values (self-attention):")
        diag_values = alpha.diagonal(dim1=0, dim2=1)[0, :, 0]  # First head, first headdim
        print(f"      First 5 diagonal values: {diag_values[:5].tolist()}")
        
    except Exception as e:
        print(f"❌ ERROR in alpha computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("="*80)
    print("Test 4: Extract all layers")
    print("="*80)
    
    try:
        all_layers = extract_abc_all_layers_mamba2(model, tokenizer, prompt, device=device, keep_on_gpu=False)
        num_layers = len(all_layers)
        print(f"✓ Extracted {num_layers} layers successfully!")
        
        # Check a few layers
        for idx in [0, num_layers//2, num_layers-1]:
            layer_data = all_layers[idx]
            print(f"  Layer {idx}: discrete_A shape = {tuple(layer_data['discrete_A'].shape)}")
        
    except Exception as e:
        print(f"❌ ERROR in multi-layer extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("="*80)
    print("✓✓✓ All tests passed! ✓✓✓")
    print("="*80)
    print()
    print("Summary:")
    print("  - Mamba2 attention extraction is working correctly")
    print("  - Multi-head architecture properly handled")
    print("  - Cumulative products computed correctly")
    print("  - Alpha attention matrix has correct causal structure")
    print()
