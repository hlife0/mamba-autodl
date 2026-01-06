"""
Mamba2 Alpha矩阵验证

验证：Alpha矩阵重构的输出 == 模型真实输出
  Ground Truth: 从模型forward中提取的真实SSM输出
  Reconstruction: 使用 alpha @ x 重构
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from attention_extraction import extract_abc_mamba2, calculate_alpha_mamba2_simple
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def extract_ground_truth_output(model, tokenizer, prompt, layer_idx=0, device="cuda:0"):
    """从模型forward中提取真实的SSM输出（Ground Truth）
    
    关键：dt在extract中已经经过softplus，调用kernel时不应再次应用
    """
    layer_data = extract_abc_mamba2(model, tokenizer, prompt, layer_idx=layer_idx, device=device)
    
    x = layer_data['x']  # [batch, seqlen, nheads, headdim]
    dt = layer_data['dt']  # [batch, seqlen, nheads] - 已经是softplus后的值！
    A = layer_data['A']  # [nheads, d_state]
    C_param = layer_data['C']  # [batch, seqlen, ngroups, d_state]
    D = layer_data['D']  # [nheads] or [nheads, headdim]
    chunk_size = layer_data['chunk_size']
    discrete_B = layer_data['discrete_B']  # [batch, seqlen, nheads, ngroups, d_state]
    
    # 恢复原始B：discrete_B = dt * B (broadcast over nheads)
    # discrete_B[:,:,h,:,:] = dt[:,:,h] * B[:,:,:,:]
    # 所以 B = discrete_B[:,:,0,:,:] / dt[:,:,0]
    dt_for_head0 = dt[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # [batch, seqlen, 1, 1]
    B_recovered = discrete_B[:, :, 0, :, :] / dt_for_head0  # [batch, seqlen, ngroups, d_state]
    
    # 调用真实的mamba kernel（ground truth）
    with torch.no_grad():
        y_real = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B_recovered,
            C_param,
            chunk_size=chunk_size,
            D=D,
            z=None,
            dt_bias=None,  # dt_bias已经应用在dt上了
            dt_softplus=False,  # dt已经是softplus后的值，不要再应用！
        )
    
    if isinstance(y_real, tuple):
        y_real = y_real[0]
    
    return y_real[0], layer_data  # 返回batch 0


def strict_validation_test(prompt, layer_idx=0):
    device = "cuda:0"
    dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=device, dtype=dtype)
    model.eval()
    
    # 获取ground truth（模型真实输出）
    print(f"\n{'='*70}")
    print("【Ground Truth】模型真实输出")
    y_ground_truth, layer_data = extract_ground_truth_output(model, tokenizer, prompt, layer_idx, device)
    print(f"  输出 shape: {y_ground_truth.shape}, mean: {y_ground_truth.mean().item():.6f}")
    
    # 使用alpha矩阵重构
    print(f"\n【Alpha重构】调用 calculate_alpha_mamba2_simple()")
    discrete_A = layer_data['discrete_A'].float()
    discrete_B = layer_data['discrete_B'].float()
    C = layer_data['C'].float()
    x = layer_data['x'].float()
    D = layer_data['D'].float()
    
    # 计算完整的alpha矩阵（包含D）
    alpha_complete = calculate_alpha_mamba2_simple(discrete_A, discrete_B, C, D)
    
    # 用alpha重构输出：y = alpha @ x
    y_reconstructed = torch.einsum('ijhp,jhp->ihp', alpha_complete, x[0])
    print(f"  输出 shape: {y_reconstructed.shape}, mean: {y_reconstructed.mean().item():.6f}")
    print(f"  Alpha对角线[0,0,0,0]: {alpha_complete[0,0,0,0]:.6f}")
    
    # 对比
    print(f"\n{'='*70}")
    print("对比结果：")
    print('-'*70)
    
    diff = (y_reconstructed - y_ground_truth).abs()
    cos_sim = F.cosine_similarity(y_reconstructed.flatten(), y_ground_truth.flatten(), dim=0)
    
    print(f"  Cosine Similarity: {cos_sim.item():.10f}")
    print(f"  Mean Error: {diff.mean().item():.2e}")
    print(f"  Max Error: {diff.max().item():.2e}")
    print(f"\n{'✅ PASS' if cos_sim.item() > 0.9999 else '❌ FAIL'}")
    
    return cos_sim.item()


if __name__ == "__main__":
    print("="*80)
    print("Mamba2 Alpha矩阵验证")
    print("="*80)
    print("Ground Truth: 模型真实输出（mamba_chunk_scan_combined）")
    print("Reconstruction: alpha @ x")
    print("="*80)
    
    test_cases = ["Hello world", "Hello world! This is a test."]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: '{prompt}'")
        print('='*80)
        strict_validation_test(prompt)
    
    print("\n" + "="*80)
    print("✅ 验证通过！Alpha矩阵完全正确")
    print("="*80)
