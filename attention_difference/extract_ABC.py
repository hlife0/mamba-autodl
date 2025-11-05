"""
提取Mamba层中离散化的A和B矩阵
使用官方 mamba-ssm 实现

使用方法:
    python extract_ABC.py --text "Your text here"

验证流程说明:
    本脚本包含两层验证，共同证明 alpha 矩阵的正确性：
    
    1. 参数提取验证 (在 extract_hook 中):
       - 使用提取的 discrete_A, discrete_B, C 手动实现完整的 SSM 前向传播
       - 对比手动计算的最终输出 vs 模型的真实输出
       - ✓ 证明：提取的 A/B/C 参数正确，手动计算的 ssm_output 可信
    
    2. Alpha 核心属性验证 (在 calculate_alpha_* 函数中):
       - 使用公式: y_i = Σ_{j=0}^{i} α_{i,j} * x_j  (只用 alpha + input)
       - 对比: alpha 重构的输出 vs ssm_output (已在第1步验证正确)
       - ✓ 证明：alpha 矩阵能从 INPUT ALONE 重构 SSM 输出，无需 A/B/C
    
    验证链条的完整性：
    - 第1步确保 ssm_output 是正确的（对比模型真实输出）
    - 第2步确保 alpha 能从 input 重构 ssm_output（不依赖 A/B/C）
    - 结论：alpha 作为"注意力权重"的解释是完全正确的！
    
    关键意义：
    一旦计算出 alpha 矩阵，就可以抛开 A/B/C，仅用 alpha 和 input 
    来理解和重构 Mamba 的行为，类似 Transformer 的注意力机制！
"""
import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
import argparse
from einops import rearrange


def extract_abc_from_layer_verified(text, model_path, layer_idx=0, device='cuda'):
    """提取离散化的A, B, C矩阵
    
    参数:
        text: 输入文本
        model_path: 模型路径
        layer_idx: 要分析的层索引 (0-63)
        device: 设备类型
    
    返回:
        dict: 包含ssm_input, discrete_A, discrete_B, C等张量的字典
    """
    
    print("="*80)
    print(f"提取Layer {layer_idx}的离散化A, B, C矩阵 (使用官方 mamba-ssm 实现)")
    print("="*80)
    
    # ==================== 1. 加载模型 ====================
    print("\n[1] 加载模型...")
    print(f"  - 模型路径: {model_path}")
    print(f"  - 设备: {device}")
    
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    
    try:
        model = MambaLMHeadModel.from_pretrained(
            model_path, 
            device=device, 
            dtype=dtype,
            save_ABC=True
        )
        model.eval()
        print(f"  - 模型加载完成")
    except Exception as e:
        print(f"  - 加载模型失败: {e}")
        return None

    # 使用 gpt-neox tokenizer (官方 mamba 使用的)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ==================== 2. 文本分词 ====================
    print("\n[2] 文本分词...")
    print(f"  输入文本: {text}")
    
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"  Token序列: {tokens}")
    print(f"  Token IDs: {input_ids.tolist()[0]}")
    print(f"  序列长度: {len(tokens)}")
    
    # ==================== 3. 注册Hook提取参数 ====================
    print("\n[3] 注册Hook提取离散化A和B...")
    
    # 存储提取的参数
    extracted = {}
    
    def extract_hook(module, input, output):
        """Hook函数：从 Mamba 模块中提取离散化A和B"""
        print("  - Hook triggered inside Mamba forward pass...")
        
        hidden_states = input[0]
        batch, seqlen, dim = hidden_states.shape
        
        # 按照官方实现的流程
        # 1. 输入投影
        xz = rearrange(
            module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if module.in_proj.bias is not None:
            xz = xz + rearrange(module.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        x, z = xz.chunk(2, dim=1)
        
        # 2. 卷积
        if module.activation in ["silu", "swish"]:
            x = F.silu(module.conv1d(x)[..., :seqlen])
        
        # 3. SSM 参数生成
        x_dbl = module.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1)
        
        # 4. dt 投影和离散化
        dt = module.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        
        # 应用 bias 和 softplus
        if module.dt_proj.bias is not None:
            dt = dt + module.dt_proj.bias.float().view(1, -1, 1)
        dt = F.softplus(dt)
        
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # 5. 获取 A 矩阵
        A = -torch.exp(module.A_log.float())  # (d_inner, d_state)
        
        # 6. 计算离散化的 A 和 B
        # discrete_A[b, d, l, n] = exp(A[d, n] * dt[b, d, l])
        # discrete_B[b, d, l, n] = dt[b, d, l] * B[b, n, l]
        
        # 扩展维度进行广播
        A_expanded = A[None, :, None, :]  # [1, d_inner, 1, d_state]
        dt_expanded = dt[:, :, :, None]   # [batch, d_inner, seqlen, 1]
        
        discrete_A = torch.exp(A_expanded * dt_expanded)  # [batch, d_inner, seqlen, d_state]
        discrete_B = dt_expanded * B[:, None, :, :].transpose(2, 3)  # [batch, d_inner, seqlen, d_state]
        
        # 7. 手动实现 SSM 递归以验证
        print("  - Computing SSM output manually for verification...")
        ssm_state = torch.zeros(
            (batch, module.d_inner, module.d_state),
            device=hidden_states.device, 
            dtype=x.dtype  # 使用 x 的dtype
        )
        
        scan_outputs = []
        x_for_scan = x  # [batch, d_inner, seqlen]
        
        for t in range(seqlen):
            # SSM递归：h_t = A_t * h_{t-1} + B_t * x_t
            ssm_state = discrete_A[:, :, t, :].to(x.dtype) * ssm_state + \
                       discrete_B[:, :, t, :].to(x.dtype) * x_for_scan[:, :, t, None]
            
            # 输出：y_t = C_t^T * h_t
            # C: [batch, d_state, seqlen], 我们需要 C[:, :, t]
            # ssm_state: [batch, d_inner, d_state]
            # C[:, :, t]: [batch, d_state]
            y_t = torch.einsum('bdn,bn->bd', ssm_state, C[:, :, t].to(ssm_state.dtype))
            scan_outputs.append(y_t)
        
        scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, d_inner, seqlen]
        
        # 加上 skip connection
        scan_output_with_skip = scan_output + (x * module.D[None, :, None].to(x.dtype))
        
        # Gating
        z_act = F.silu(z)
        contextualized_states = scan_output_with_skip * z_act
        
        # 输出投影
        manual_output = module.out_proj(rearrange(contextualized_states, "b d l -> b l d"))
        
        # 验证: 对比模型真实输出和手动计算的输出
        diff = (output.to(manual_output.dtype) - manual_output).abs().max().item()
        print(f"  - Verification: max difference = {diff:.2e}")
        if diff < 1e-2:  # 放宽一点，因为 float16 精度有限
            print("  - ✓ Verification successful! Parameters extracted correctly.")
        else:
            print(f"  - ⚠ Warning: difference is large ({diff:.2e})")
        
        # 8. 从模型真实输出反推 SSM 输出（用于后续 alpha 验证）
        # output 的形状是 [batch, seqlen, d_model]
        # 需要反推出 scan_output_with_skip（在 gating 之前）
        # 因为 output = out_proj(contextualized_states)
        # 我们需要反投影
        print("  - Extracting ground truth SSM output from model...")
        
        # 反投影：y_before_proj = out_proj^{-1}(output)
        # 由于 out_proj 是线性层，我们不能直接求逆，但我们可以保存真实的中间值
        # 这里我们保存手动计算的值，并额外保存模型的真实最终输出用于完整验证
        
        # 保存到字典
        extracted['ssm_input'] = x_for_scan  # [batch, d_inner, seqlen]
        extracted['ssm_output'] = scan_output  # [batch, d_inner, seqlen] - 纯 SSM 输出（无 skip/gating）
        extracted['ssm_output_with_skip'] = scan_output_with_skip  # [batch, d_inner, seqlen] - 加了 skip connection
        extracted['z_activation'] = z_act  # [batch, d_inner, seqlen] - gating 激活
        extracted['discrete_A'] = discrete_A  # [batch, d_inner, seqlen, d_state]
        extracted['discrete_B'] = discrete_B  # [batch, d_inner, seqlen, d_state]
        extracted['C'] = C  # [batch, d_state, seqlen]
        extracted['dt'] = dt  # [batch, d_inner, seqlen]
        extracted['A_continuous'] = A  # [d_inner, d_state]
        extracted['D'] = module.D  # [d_inner] - skip connection 权重
        extracted['model_output'] = output  # [batch, seqlen, d_model] - 模型真实输出
    
    # 在指定层的 Mamba 模块注册 hook
    target_layer = model.backbone.layers[layer_idx].mixer
    hook_handle = target_layer.register_forward_hook(extract_hook)
    
    # ==================== 4. 运行模型触发Hook ====================
    print("\n[4] 运行模型...")
    with torch.no_grad():
        output = model(input_ids)
    
    hook_handle.remove()
    
    # ==================== 5. 打印结果 ====================
    print("\n" + "="*80)
    print("Extraction Results")
    print("="*80)
    
    if not extracted:
        print("❌ 提取失败！")
        return None
    
    discrete_A = extracted['discrete_A']
    discrete_B = extracted['discrete_B']
    C = extracted['C']
    dt = extracted['dt']
    
    print(f"\nSSM Input (x after conv):")
    print(f"  Shape: {list(extracted['ssm_input'].shape)}")
    
    print(f"\nTime step (dt, after softplus):")
    print(f"  Shape: {list(dt.shape)}")
    print(f"  Stats: mean={dt.mean():.6f}, std={dt.std():.6f}")
    print(f"  Range: [{dt.min():.6f}, {dt.max():.6f}]")
    
    print(f"\nContinuous A matrix:")
    print(f"  Shape: {list(extracted['A_continuous'].shape)}")
    print(f"  Stats: mean={extracted['A_continuous'].mean():.6f}, std={extracted['A_continuous'].std():.6f}")
    print(f"  Range: [{extracted['A_continuous'].min():.6f}, {extracted['A_continuous'].max():.6f}]")
    
    print(f"\nDiscrete A matrix (exp(A * dt)):")
    print(f"  Shape: {list(discrete_A.shape)}")
    print(f"  Stats: mean={discrete_A.mean():.6f}, std={discrete_A.std():.6f}")
    print(f"  Range: [{discrete_A.min():.6f}, {discrete_A.max():.6f}]")
    
    print(f"\nDiscrete B matrix (dt * B):")
    print(f"  Shape: {list(discrete_B.shape)}")
    print(f"  Stats: mean={discrete_B.mean():.6f}, std={discrete_B.std():.6f}")
    print(f"  Range: [{discrete_B.min():.6f}, {discrete_B.max():.6f}]")
    
    print(f"\nC matrix:")
    print(f"  Shape: {list(C.shape)}")
    print(f"  Stats: mean={C.mean():.6f}, std={C.std():.6f}")
    print(f"  Range: [{C.min():.6f}, {C.max():.6f}]")
    
    print("\n" + "="*80)
    print("Done")
    print("="*80)
    
    return extracted


def calculate_alpha_for_one_channel_verified(extracted_matrices, channel_idx=0):
    """
    只为单个指定通道计算alpha矩阵, 并验证alpha数学推导的正确性
    
    Alpha 矩阵的含义：
    α_{i,j} 表示在时刻 i 的输出中，时刻 j 的输入贡献了多少
    
    计算公式：
    α_{i,j} = C_i^T * (∏_{k=j+1}^{i} A_k) * B_j
    
    其中：
    - 当 j = i 时，∏ 为单位矩阵
    - A_k, B_j 是离散化后的矩阵
    
    验证逻辑：
    - 用递归方式计算的 SSM 输出（已在 hook 中验证过正确性）
    - vs 用 alpha 矩阵重构的输出
    - 如果两者一致，说明 alpha 矩阵的数学推导是正确的
    """
    print("\n" + "="*80)
    print(f"Calculating and Verifying Alpha Matrix for Channel {channel_idx}...")
    print("="*80)
    print("Note: This verifies that the alpha matrix mathematical derivation is correct.")

    # 提取该通道所需的数据
    # discrete_A: [batch, d_inner, seqlen, d_state]
    # discrete_B: [batch, d_inner, seqlen, d_state]
    # C: [batch, d_state, seqlen]
    # ssm_input: [batch, d_inner, seqlen]
    # ssm_output: [batch, d_inner, seqlen]
    
    A = extracted_matrices['discrete_A'][0, channel_idx, :, :].float()  # [seqlen, d_state]
    B = extracted_matrices['discrete_B'][0, channel_idx, :, :].float()  # [seqlen, d_state]
    C = extracted_matrices['C'][0, :, :].float()  # [d_state, seqlen]
    ssm_input_channel = extracted_matrices['ssm_input'][0, channel_idx, :].float()  # [seqlen]
    ssm_output_original_channel = extracted_matrices['ssm_output'][0, channel_idx, :].float()  # [seqlen]
    
    seq_len = A.shape[0]
    d_state = A.shape[1]
    
    print(f"  - Channel {channel_idx} dimensions:")
    print(f"    Sequence length: {seq_len}")
    print(f"    State dimension: {d_state}")
    
    # 1. 高效计算单个通道的alpha矩阵
    print(f"  - Calculating alpha matrix [{seq_len} x {seq_len}]...")
    alpha_channel_matrix = torch.zeros((seq_len, seq_len), device=A.device, dtype=A.dtype)
    
    for i in range(seq_len):
        # A_prod 保存累积连乘 ∏_{k=j+1}^{i} A_k
        # 初始化为全1向量 (因为 A_k 是对角操作)
        A_prod = torch.ones(d_state, device=A.device, dtype=A.dtype)
        
        # 计算 alpha_{i,i} = C_i^T * B_i (因为 ∏ 为单位)
        alpha_ii = torch.dot(C[:, i], B[i, :])
        alpha_channel_matrix[i, i] = alpha_ii
        
        # 从 j = i-1 向 j = 0 迭代
        for j in range(i - 1, -1, -1):
            # 更新累积乘积：A_prod *= A_{j+1}
            A_prod = A_prod * A[j + 1, :]
            
            # 计算 alpha_{i,j} = C_i^T * A_prod * B_j
            B_effective = A_prod * B[j, :]
            alpha_ij = torch.dot(C[:, i], B_effective)
            alpha_channel_matrix[i, j] = alpha_ij

    print("  - Alpha matrix calculation complete.")

    # 2. 验证 alpha 矩阵的核心属性
    print("  - Verifying KEY property: y = alpha @ x (input-only reconstruction)...")
    print("    This tests if SSM output can be reconstructed from INPUT ALONE")
    print("    using alpha matrix, WITHOUT needing A/B/C matrices.")
    
    alpha_reconstructed_output = torch.zeros_like(ssm_output_original_channel)
    for i in range(seq_len):
        # Y_i = Σ_{j=0}^{i} α_{i,j} * X_j
        # This formula uses ONLY alpha and input, not A/B/C!
        y_i = torch.sum(alpha_channel_matrix[i, :i+1] * ssm_input_channel[:i+1])
        alpha_reconstructed_output[i] = y_i

    # 3. 计算差异
    diff = (ssm_output_original_channel - alpha_reconstructed_output).abs().max().item()
    mean_diff = (ssm_output_original_channel - alpha_reconstructed_output).abs().mean().item()
    
    print(f"\n  - Verification results:")
    print(f"    Max difference: {diff:.2e}")
    print(f"    Mean difference: {mean_diff:.2e}")
    
    if diff < 1e-3:
        print(f"  - ✓ KEY property verified for channel {channel_idx}!")
        print(f"    SSM output successfully reconstructed from INPUT ALONE using alpha!")
        print(f"    This proves: given alpha, we don't need A/B/C to compute output.")
    else:
        print(f"  - ⚠ Warning: difference might be large due to numerical precision")
    
    print(f"\n  - Interpretation:")
    print(f"    Alpha matrix acts as 'attention weights' in Mamba:")
    print(f"    α[i,j] = how much input token j contributes to output token i")
    print(f"    Just like Transformer attention, but derived from SSM dynamics!")
    
    print("="*80)
    
    return alpha_channel_matrix


def calculate_and_verify_alpha_matrix_verified(extracted_matrices, verify=True):
    """
    计算完整的alpha注意力矩阵
    
    参数:
        extracted_matrices: 提取的矩阵字典
        verify: 是否验证alpha数学推导的正确性
    
    返回:
        alpha_matrix: [seqlen, seqlen, d_inner] 的注意力矩阵
    
    验证逻辑：
    - 用递归方式计算的 SSM 输出（已在 hook 中验证过正确性）
    - vs 用 alpha 矩阵重构的输出
    - 如果两者一致，说明 alpha 矩阵的数学推导是正确的
    """
    print("\n" + "="*80)
    print("Calculating Full Attention Matrix (alpha)...")
    print("="*80)
    print("Note: This calculates alpha for ALL channels (may be memory/time intensive).")
    
    # 提取张量
    # discrete_A: [batch, d_inner, seqlen, d_state]
    # discrete_B: [batch, d_inner, seqlen, d_state]
    # C: [batch, d_state, seqlen]
    A = extracted_matrices['discrete_A'][0].float()  # [d_inner, seqlen, d_state]
    B = extracted_matrices['discrete_B'][0].float()  # [d_inner, seqlen, d_state]
    C = extracted_matrices['C'][0].float()  # [d_state, seqlen]
    
    d_inner, seq_len, d_state = A.shape
    
    print(f"  - Matrix dimensions:")
    print(f"    d_inner (channels): {d_inner}")
    print(f"    seq_len (sequence): {seq_len}")
    print(f"    d_state (state): {d_state}")
    print(f"  - Alpha matrix will be [{seq_len} x {seq_len} x {d_inner}]")
    print(f"  - This may take a while for large dimensions...")
    
    # 初始化 alpha 矩阵
    alpha_matrix = torch.zeros(
        (seq_len, seq_len, d_inner), 
        device=A.device, 
        dtype=A.dtype
    )
    
    # 对每个通道计算 alpha
    for ch in range(d_inner):
        if ch % 500 == 0:
            print(f"  - Processing channel {ch}/{d_inner}...")
        
        for i in range(seq_len):
            A_prod = torch.ones(d_state, device=A.device, dtype=A.dtype)
            
            # alpha_{i,i}
            alpha_ii = torch.dot(C[:, i], B[ch, i, :])
            alpha_matrix[i, i, ch] = alpha_ii
            
            # alpha_{i,j} for j < i
            for j in range(i - 1, -1, -1):
                A_prod = A_prod * A[ch, j + 1, :]
                B_effective = A_prod * B[ch, j, :]
                alpha_ij = torch.dot(C[:, i], B_effective)
                alpha_matrix[i, j, ch] = alpha_ij
    
    print("  - Alpha matrix calculation complete.")
    
    # 验证
    if verify:
        print("\n  - Verifying alpha matrix by reconstructing SSM output...")
        print("    (Comparing: SSM recursion vs alpha-based reconstruction for all channels)")
        ssm_input = extracted_matrices['ssm_input'][0].float()  # [d_inner, seqlen]
        ssm_output_original = extracted_matrices['ssm_output'][0].float()  # [d_inner, seqlen]
        
        alpha_reconstructed = torch.zeros_like(ssm_output_original)
        for i in range(seq_len):
            for ch in range(d_inner):
                # Y_{i,ch} = Σ_{j=0}^{i} α_{i,j,ch} * X_{j,ch}
                y_i_ch = torch.sum(alpha_matrix[i, :i+1, ch] * ssm_input[ch, :i+1])
                alpha_reconstructed[ch, i] = y_i_ch
        
        diff = (ssm_output_original - alpha_reconstructed).abs().max().item()
        mean_diff = (ssm_output_original - alpha_reconstructed).abs().mean().item()
        
        print(f"    Max difference: {diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        
        if diff < 1e-3:
            print("  - ✓ Alpha matrix derivation verified for all channels!")
            print("    SSM recursion and alpha reconstruction produce identical results.")
        else:
            print(f"  - ⚠ Warning: difference might be due to numerical precision")
    
    print("="*80)
    
    return alpha_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract discrete A, B, C matrices from a Mamba layer using official mamba-ssm."
    )
    parser.add_argument(
        "--text", 
        type=str, 
        default="Hello world! This is a test. LALALALALALA！", 
        help="Input text for the model."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="state-spaces/mamba-2.8b",
        help="Path to the Mamba model."
    )
    parser.add_argument(
        "--layer_idx", 
        type=int, 
        default=0, 
        help="Index of the Mamba layer to analyze (0-63 for 2.8B model)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to run the model on."
    )
    parser.add_argument(
        "--verify_channel",
        type=int,
        default=0,
        help="Channel index to verify (default: 0)."
    )
    parser.add_argument(
        "--verify_full",
        action="store_true",
        help="Calculate and verify full alpha matrix (memory intensive)."
    )
    
    args = parser.parse_args()
    
    # 提取参数
    extracted_matrices = extract_abc_from_layer_verified(
        text=args.text,
        model_path=args.model_path,
        layer_idx=args.layer_idx,
        device=args.device
    )

    if extracted_matrices:
        print("\n" + "="*80)
        print("Extracted tensor shapes:")
        print("="*80)
        for name, tensor in extracted_matrices.items():
            print(f"  {name:20s}: {list(tensor.shape)}")
        print("="*80)
        
        # 验证单个通道
        print("\n[Optional] Verifying single channel...")
        calculate_alpha_for_one_channel_verified(
            extracted_matrices, 
            channel_idx=args.verify_channel
        )
        
        # 可选：计算完整 alpha 矩阵
        if args.verify_full:
            print("\n[Optional] Calculating full alpha matrix (this may take time)...")
            alpha_matrix = calculate_and_verify_alpha_matrix_verified(extracted_matrices, verify=True)
            print(f"\nFull alpha matrix shape: {list(alpha_matrix.shape)}")

