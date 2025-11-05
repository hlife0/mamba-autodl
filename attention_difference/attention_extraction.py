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
    
    Note on Mamba architecture:
    - Original B, C: shared across all d_inner channels [batch, d_state, seqlen]
    - A, dt: channel-specific [d_inner, ...] 
    - discrete_A = exp(A * dt): channel-specific due to channel-specific A and dt
    - discrete_B = dt * B: channel-specific due to channel-specific dt (despite shared B)
    
    Returns: dict with keys [ssm_input, ssm_output, discrete_A, discrete_B, C, dt, A_continuous, D]
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    
    extracted = {}
    
    def extract_hook(module, input, output):
        hidden_states = input[0]
        batch, seqlen, dim = hidden_states.shape
        
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
        
        x_dbl = module.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1)
        
        dt = module.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        
        if module.dt_proj.bias is not None:
            dt = dt + module.dt_proj.bias.float().view(1, -1, 1)
        dt = F.softplus(dt)
        
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        A = -torch.exp(module.A_log.float())
        
        A_expanded = A[None, :, None, :]
        dt_expanded = dt[:, :, :, None]
        
        discrete_A = torch.exp(A_expanded * dt_expanded)
        discrete_B = dt_expanded * B[:, None, :, :].transpose(2, 3)
        
        # Pre-convert to target dtype to avoid repeated conversions
        discrete_A_conv = discrete_A.to(x.dtype)
        discrete_B_conv = discrete_B.to(x.dtype)
        C_conv = C.to(x.dtype)
        
        ssm_state = torch.zeros(
            (batch, module.d_inner, module.d_state),
            device=hidden_states.device, 
            dtype=x.dtype
        )
        
        # Preallocate output tensor instead of list append
        scan_output = torch.zeros((batch, module.d_inner, seqlen), device=x.device, dtype=x.dtype)
        
        for t in range(seqlen):
            ssm_state = discrete_A_conv[:, :, t, :] * ssm_state + \
                       discrete_B_conv[:, :, t, :] * x[:, :, t, None]
            scan_output[:, :, t] = torch.einsum('bdn,bn->bd', ssm_state, C_conv[:, :, t])
        
        extracted['ssm_input'] = x
        extracted['ssm_output'] = scan_output
        extracted['discrete_A'] = discrete_A
        extracted['discrete_B'] = discrete_B
        extracted['C'] = C
        extracted['dt'] = dt
        extracted['A_continuous'] = A
        extracted['D'] = module.D
    
    target_layer = model.backbone.layers[layer_idx].mixer
    hook_handle = target_layer.register_forward_hook(extract_hook)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    hook_handle.remove()
    
    return extracted

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
    
    B_effective = A_cumprod * B[:, None, :, :]
    alpha_matrix = torch.einsum('si,cijs->ijc', C_mat, B_effective)
    
    diagonal_alpha = torch.einsum('si,cis->ic', C_mat, B)
    alpha_matrix.diagonal(dim1=0, dim2=1).copy_(diagonal_alpha.T)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--prompt", type=str, default="Hello world! This is a test.")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verify", action="store_true", help="Verify against slow version")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32
    
    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    model.eval()
    
    print(f"Extracting from layer {args.layer_idx}...\n")
    
    result = extract_abc(
        model,
        tokenizer,
        args.prompt, 
        layer_idx=args.layer_idx,
        device=device
    )
    
    print("Extracted matrices:")
    for name, tensor in result.items():
        print(f"  {name:20s}: {list(tensor.shape)}")
    
    # Test alpha-beta relationship
    print("\n" + "="*60)
    print("Testing Alpha-Beta relationship")
    print("="*60)
    
    seq_len = result['discrete_A'].shape[2]
    d_inner = result['discrete_A'].shape[1]
    d_state = result['discrete_A'].shape[3]
    
    # Memory estimate: beta is [seq, seq, d_inner, d_state] in float32
    beta_size_mb = (seq_len * seq_len * d_inner * d_state * 4) / (1024**2)
    
    if beta_size_mb < 500:  # Only skip if > 500MB
        print(f"Computing alpha and beta (estimated memory: {beta_size_mb:.1f}MB)...")
        
        alpha = calculate_alpha_all_channels(
            result['discrete_A'],
            result['discrete_B'],
            result['C']
        )
        print(f"  Alpha shape: {list(alpha.shape)}")
        
        beta = calculate_beta_all_channels(
            result['discrete_A'],
            result['C']
        )
        print(f"  Beta shape: {list(beta.shape)}")
        
        # Reconstruct alpha from beta: α[i,j,ch] = Σ_s β[i,j,ch,s] * B[ch,j,s]
        discrete_B_data = result['discrete_B'][0].float()  # [d_inner, seqlen, d_state]
        alpha_reconstructed = torch.einsum('ijcs,cjs->ijc', beta, discrete_B_data)
        
        diff = (alpha - alpha_reconstructed).abs()
        print(f"\nVerification: α = β · discrete_B")
        print(f"  Max difference: {diff.max():.2e}")
        print(f"  Mean difference: {diff.mean():.2e}")
        
        if diff.max() < 1e-4:
            print("  ✓ Relationship verified: α_{i,j,ch} = Σ_s β_{i,j,ch,s} * B_{j,ch,s}")
        else:
            print(f"  ⚠ Warning: large difference detected")
    else:
        print(f"Skipping test (estimated memory: {beta_size_mb:.1f}MB > 500MB)")
    
    print("="*60)
    
    if args.verify:
        print("\n" + "="*60)
        print("Verification: comparing fast vs slow implementation")
        print("="*60)
        
        from attention_difference.extract_ABC import calculate_alpha_for_one_channel_verified
        import time
        
        # Slow version
        print("\n[1] Running slow version (single channel)...")
        t0 = time.time()
        alpha_slow = calculate_alpha_for_one_channel_verified(result, channel_idx=0)
        t_slow = time.time() - t0
        
        # Fast version
        print(f"\n[2] Running fast version (single channel)...")
        t0 = time.time()
        alpha_fast = calculate_alpha_single_channel(
            result['discrete_A'], 
            result['discrete_B'], 
            result['C'],
            channel_idx=0
        )
        t_fast = time.time() - t0
        
        # Compare
        diff = (alpha_slow - alpha_fast).abs()
        print(f"\n[3] Comparison:")
        print(f"  Max difference: {diff.max():.2e}")
        print(f"  Mean difference: {diff.mean():.2e}")
        print(f"  Time (slow): {t_slow:.4f}s")
        print(f"  Time (fast): {t_fast:.4f}s")
        print(f"  Speedup: {t_slow/t_fast:.2f}x")
        
        if diff.max() < 1e-4:
            print("  ✓ Results match!")
        else:
            print(f"  ⚠ Warning: large difference detected")
        
        # Test all channels (if sequence is short enough)
        seq_len = result['discrete_A'].shape[2]
        d_inner = result['discrete_A'].shape[1]
        if seq_len < 50 and d_inner < 1000:
            print(f"\n[4] Testing all channels ({d_inner} channels)...")
            t0 = time.time()
            alpha_all_fast = calculate_alpha_all_channels(
                result['discrete_A'], 
                result['discrete_B'], 
                result['C']
            )
            t_all = time.time() - t0
            print(f"  Alpha matrix shape: {list(alpha_all_fast.shape)}")
            print(f"  Time: {t_all:.4f}s")
            print(f"  Channel 0 matches single-channel result: {(alpha_all_fast[:, :, 0] - alpha_fast).abs().max():.2e}")
        else:
            print(f"\n[4] Skipping all-channels test (seq_len={seq_len}, d_inner={d_inner} too large)")

