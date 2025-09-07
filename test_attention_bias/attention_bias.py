import torch
import torch.nn as nn


class AttentionWithLearnableBias(nn.Module):
    def __init__(self, num_heads, max_bias_length=512):
        super().__init__()
        self.num_heads = num_heads
        self.max_bias_length = max_bias_length
        # 初始化可学习的对角线bias参数
        self.learnable_bias_diagonals = nn.Parameter(
            torch.zeros(num_heads, max_bias_length)
        )
        # 可以用小的随机值初始化
        nn.init.normal_(self.learnable_bias_diagonals, mean=0, std=0.02)
    
    def apply_learnable_bias_efficient(self, attn_weights):
        """高效地应用对角线 bias，使用GPU并行操作，避免大内存占用"""
        batch_size, num_heads, seq_len_q, seq_len_k = attn_weights.shape

        # 创建相对位置索引矩阵 [seq_len_q, seq_len_k]
        q_pos = torch.arange(seq_len_q, device=attn_weights.device, dtype=torch.long)
        k_pos = torch.arange(seq_len_k, device=attn_weights.device, dtype=torch.long)
        rel_pos = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)

        # Causal mask: 只处理下三角部分
        causal_mask = (rel_pos >= 0).to(attn_weights.dtype)

        # 限制相对位置在参数范围内
        rel_pos = torch.clamp(rel_pos, min=0, max=self.max_bias_length - 1)

        # 直接使用高级索引从learnable_bias_diagonals中获取对应的bias值
        # learnable_bias_diagonals: [num_heads, max_bias_length]
        # rel_pos: [seq_len_q, seq_len_k]
        # 结果: [num_heads, seq_len_q, seq_len_k]
        bias_values = self.learnable_bias_diagonals[:, rel_pos]

        # 应用causal mask（上三角部分设为0）
        bias_values = bias_values * causal_mask

        # 扩展到batch维度并相加
        # [num_heads, seq_len_q, seq_len_k] -> [1, num_heads, seq_len_q, seq_len_k] -> [batch_size, num_heads, seq_len_q, seq_len_k]
        bias_values = bias_values.unsqueeze(0).expand(batch_size, -1, -1, -1)

        final = attn_weights + bias_values
        return final


def test_attention_bias():
    """测试attention bias功能"""
    print("=" * 60)
    print("Testing Attention with Learnable Bias")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    num_heads = 4
    seq_len = 32
    max_bias_length = 16
    
    # 创建模型
    model = AttentionWithLearnableBias(num_heads=num_heads, max_bias_length=max_bias_length)
    
    # 创建随机的attention weights
    attn_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    print(f"Input shape: {attn_weights.shape}")
    print(f"Learnable bias shape: {model.learnable_bias_diagonals.shape}")
    print()
    
    # 应用bias
    output = model.apply_learnable_bias_efficient(attn_weights)
    
    print(f"\nOutput shape: {output.shape}")
    
    # 验证causal mask是否正确应用（上三角应该没有bias）
    print("\n" + "=" * 60)
    print("Checking causal mask application:")
    print("=" * 60)
    
    # 检查第一个batch，第一个head的结果
    sample_input = attn_weights[0, 0, :, :]
    sample_output = output[0, 0, :, :]
    bias_applied = sample_output - sample_input
    
    print("\nSample attention weights (5x5):")
    print(sample_input.detach().numpy().round(3))
    
    print("\nBias applied (should be 0 in upper triangle):")
    print(bias_applied.detach().numpy().round(3))
    
    # 验证上三角是否为0
    upper_triangle_sum = torch.triu(bias_applied, diagonal=1).abs().sum()
    print(f"\nUpper triangle bias sum (should be 0): {upper_triangle_sum.item():.6f}")
    
    # 验证对角线bias是否正确应用
    print("\n" + "=" * 60)
    print("Checking diagonal bias values:")
    print("=" * 60)
    
    for i in range(min(5, seq_len)):
        diagonal_bias = bias_applied[i, :i+1]
        print(f"Row {i} bias values: {diagonal_bias.detach().numpy().round(3)}")
    
    # 测试梯度流
    print("\n" + "=" * 60)
    print("Testing gradient flow:")
    print("=" * 60)
    
    loss = output.mean()
    loss.backward()
    
    if model.learnable_bias_diagonals.grad is not None:
        grad_norm = model.learnable_bias_diagonals.grad.norm().item()
        print(f"Gradient norm of learnable_bias_diagonals: {grad_norm:.6f}")
        print("Gradient flow: ✓ Working correctly")
    else:
        print("Gradient flow: ✗ No gradients computed")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def test_exceeding_max_bias_length():
    """测试当序列长度超过max_bias_length时的行为"""
    print("\n" + "=" * 60)
    print("Testing behavior when seq_len > max_bias_length")
    print("=" * 60)
    
    # 设置参数：seq_len故意大于max_bias_length
    batch_size = 1
    num_heads = 2
    seq_len = 32  # 大于max_bias_length
    max_bias_length = 16  # 小于seq_len
    
    print(f"seq_len: {seq_len}")
    print(f"max_bias_length: {max_bias_length}")
    print(f"Testing when seq_len ({seq_len}) > max_bias_length ({max_bias_length})")
    
    # 创建模型
    model = AttentionWithLearnableBias(num_heads=num_heads, max_bias_length=max_bias_length)
    
    # 为了更清楚地看到效果，给bias设置特定的值
    with torch.no_grad():
        for i in range(max_bias_length):
            model.learnable_bias_diagonals[:, i] = i * 0.1  # 0.0, 0.1, 0.2, ..., 1.5
    
    # 创建attention weights
    attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    
    # 应用bias
    output = model.apply_learnable_bias_efficient(attn_weights)
    
    # 检查bias的应用情况
    bias_applied = output[0, 0, :, :]  # 第一个batch，第一个head
    
    print(f"\n检查不同位置的bias值：")
    print("-" * 40)
    
    # 检查几个关键位置
    positions_to_check = [
        (5, 0),   # 相对位置 = 5
        (10, 5),  # 相对位置 = 5
        (15, 0),  # 相对位置 = 15（最大值）
        (20, 5),  # 相对位置 = 15（被clamp到max_bias_length-1）
        (25, 10), # 相对位置 = 15（被clamp）
        (31, 0),  # 相对位置 = 31（被clamp到15）
        (31, 16), # 相对位置 = 15（被clamp）
        (31, 20), # 相对位置 = 11
    ]
    
    for q_pos, k_pos in positions_to_check:
        rel_pos = q_pos - k_pos
        actual_bias = bias_applied[q_pos, k_pos].item()
        expected_rel_pos = min(rel_pos, max_bias_length - 1)
        expected_bias = expected_rel_pos * 0.1
        
        print(f"Position ({q_pos:2d}, {k_pos:2d}): "
              f"rel_pos={rel_pos:2d}, "
              f"clamped_to={expected_rel_pos:2d}, "
              f"bias={actual_bias:.2f} "
              f"(expected={expected_bias:.2f})")
    
    # 可视化bias矩阵的一部分
    print(f"\n可视化bias矩阵（显示左下角20x20）:")
    print("-" * 40)
    
    vis_size = min(20, seq_len)
    vis_matrix = bias_applied[-vis_size:, :vis_size].detach().numpy()
    
    # 打印矩阵
    import numpy as np
    np.set_printoptions(precision=1, suppress=True)
    print(vis_matrix)
    
    # 检查最大的相对位置
    print(f"\n关键发现：")
    print("-" * 40)
    print(f"1. 当相对位置 >= {max_bias_length-1} 时，bias都被限制为 {(max_bias_length-1)*0.1:.1f}")
    print(f"2. 这意味着距离超过{max_bias_length-1}的位置都使用相同的bias值")
    print(f"3. 上三角（未来位置）的bias仍然为0（causal mask）")
    
    # 验证：检查所有相对位置>=max_bias_length-1的位置是否都有相同的bias
    max_bias_value = model.learnable_bias_diagonals[0, max_bias_length-1].item()
    positions_with_max_bias = []
    
    for i in range(seq_len):
        for j in range(i+1):  # 只检查下三角
            if i - j >= max_bias_length - 1:
                if abs(bias_applied[i, j].item() - max_bias_value) < 1e-6:
                    positions_with_max_bias.append((i, j))
    
    print(f"\n使用最大bias值({max_bias_value:.2f})的位置数量: {len(positions_with_max_bias)}")
    print(f"前5个位置示例: {positions_with_max_bias[:5]}")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    test_attention_bias()
    test_exceeding_max_bias_length()