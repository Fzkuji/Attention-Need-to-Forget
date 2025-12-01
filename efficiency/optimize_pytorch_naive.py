"""
测试优化 PyTorch naive Lazy Attention 的几种方法
对比: 原始实现 vs nn.Embedding vs Triton vs Flash Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

B, H, L, D = 4, 16, 4096, 64
max_bias_length = 4096
device = 'cuda'
dtype = torch.bfloat16

q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
bias = torch.randn(H, max_bias_length, device=device, dtype=dtype, requires_grad=True)
tau = torch.full((H,), -1.0, device=device, dtype=dtype, requires_grad=True)

scale = D ** -0.5

# ============================================================
# 方法 1: 原始实现 (baseline) - 使用 bias[:, indices]
# ============================================================
def lazy_attention_original(q, k, v, bias, tau):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    L = q.shape[-2]
    rel_pos = torch.arange(L, device=q.device)[:, None] - torch.arange(L, device=q.device)[None, :]
    causal_mask = rel_pos >= 0
    bias_mask = causal_mask & (rel_pos < max_bias_length)
    indices = rel_pos.clamp(0, max_bias_length - 1)
    bias_matrix = bias[:, indices] * bias_mask.to(scores.dtype)
    scores = scores + bias_matrix.unsqueeze(0)

    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    tau_view = tau.view(1, -1, 1, 1)
    i_pos = torch.arange(1, L + 1, device=q.device, dtype=attn.dtype).view(1, 1, -1, 1)
    attn = F.relu(attn + tau_view / i_pos)

    return torch.matmul(attn, v)

# ============================================================
# 方法 2: 使用 nn.Embedding 替代直接索引 (优化后)
# ============================================================
class LazyAttentionEmbedding(nn.Module):
    def __init__(self, num_heads, max_bias_length, dtype):
        super().__init__()
        self.num_heads = num_heads
        self.max_bias_length = max_bias_length
        self.bias_embed = nn.Embedding(max_bias_length, num_heads)
        self.bias_embed.weight.data = self.bias_embed.weight.data.to(dtype)

    def forward(self, q, k, v, tau):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        L = q.shape[-2]

        rel_pos = torch.arange(L, device=q.device)[:, None] - torch.arange(L, device=q.device)[None, :]
        causal_mask = rel_pos >= 0
        bias_mask = causal_mask & (rel_pos < self.max_bias_length)
        indices = rel_pos.clamp(0, self.max_bias_length - 1)

        # Embedding: [L, L] -> [L, L, H] -> [H, L, L]
        bias_values = self.bias_embed(indices).permute(2, 0, 1)
        bias_matrix = bias_values * bias_mask.to(scores.dtype)
        scores = scores + bias_matrix.unsqueeze(0)

        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        tau_view = tau.view(1, -1, 1, 1)
        i_pos = torch.arange(1, L + 1, device=q.device, dtype=attn.dtype).view(1, 1, -1, 1)
        attn = F.relu(attn + tau_view / i_pos)

        return torch.matmul(attn, v)


# ============================================================
# Benchmark
# ============================================================
def benchmark(name, func, *args, warmup=5, repeat=20):
    # Warmup
    for _ in range(warmup):
        out = func(*args)
        out.sum().backward()
        for a in args:
            if hasattr(a, 'grad') and a.grad is not None:
                a.grad = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        out = func(*args)
        out.sum().backward()
        for a in args:
            if hasattr(a, 'grad') and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()

    avg_time = (time.perf_counter() - start) / repeat * 1000
    print(f'{name}: {avg_time:.2f}ms')
    return avg_time


print("=" * 70)
print("PyTorch Naive Lazy Attention 优化测试")
print(f"B={B}, H={H}, L={L}, D={D}, max_bias_length={max_bias_length}")
print("=" * 70)

# 基准：标准 softmax attention (PyTorch naive)
def softmax_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    L = q.shape[-2]
    causal_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)

print("\n--- Baselines ---")
softmax_time = benchmark("Softmax PyTorch", softmax_attention, q, k, v)

# Flash Attention baseline
try:
    from flash_attn import flash_attn_func
    q_flash = q.transpose(1, 2).contiguous()
    k_flash = k.transpose(1, 2).contiguous()
    v_flash = v.transpose(1, 2).contiguous()

    def flash_attn_wrapper(q, k, v):
        return flash_attn_func(q, k, v, causal=True)

    flash_time = benchmark("Flash Attention", flash_attn_wrapper, q_flash, k_flash, v_flash)
except ImportError:
    print("Flash Attention: Not available")
    flash_time = None

print("\n--- Lazy Attention Implementations ---")

# 原始实现
original_time = benchmark("1. Original (bias[:, indices])", lazy_attention_original, q, k, v, bias, tau)

# Embedding 实现
embed_model = LazyAttentionEmbedding(H, max_bias_length, dtype).to(device)
embed_model.bias_embed.weight.data = bias.t().detach().clone()
embed_time = benchmark("2. Embedding (optimized)", embed_model, q, k, v, tau)

# Triton kernel
try:
    from lazy_attention_triton import lazy_attention_triton
    triton_time = benchmark("3. Triton Kernel", lazy_attention_triton, q, k, v, bias.to(q.dtype), tau.to(q.dtype))
except ImportError as e:
    print(f"3. Triton Kernel: Not available - {e}")
    triton_time = None

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Method':<35} {'Time (ms)':<12} {'vs Softmax':<12} {'vs Flash':<12}")
print("-" * 70)

def format_row(name, time_ms, softmax_time, flash_time):
    vs_softmax = f"{time_ms/softmax_time*100:.1f}%" if softmax_time else "N/A"
    vs_flash = f"{time_ms/flash_time:.1f}x" if flash_time else "N/A"
    print(f"{name:<35} {time_ms:<12.2f} {vs_softmax:<12} {vs_flash:<12}")

format_row("Softmax PyTorch (baseline)", softmax_time, softmax_time, flash_time)
if flash_time:
    format_row("Flash Attention (baseline)", flash_time, softmax_time, flash_time)
print("-" * 70)
format_row("1. Original (bias[:, indices])", original_time, softmax_time, flash_time)
format_row("2. Embedding (optimized)", embed_time, softmax_time, flash_time)
if triton_time:
    format_row("3. Triton Kernel", triton_time, softmax_time, flash_time)

print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print(f"  - Embedding vs Original: {original_time/embed_time:.1f}x speedup")
if triton_time:
    print(f"  - Triton vs Original: {original_time/triton_time:.1f}x speedup")
    print(f"  - Embedding vs Triton: {embed_time/triton_time:.1f}x (Embedding is {'slower' if embed_time > triton_time else 'faster'})")
print("=" * 70)
