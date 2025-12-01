"""
测试优化 PyTorch naive Lazy Attention 的几种方法
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
# 方法 1: 原始实现 (baseline)
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
# 方法 2: 使用 nn.Embedding 替代直接索引
# ============================================================
class LazyAttentionEmbedding(nn.Module):
    def __init__(self, num_heads, max_bias_length, dtype):
        super().__init__()
        self.num_heads = num_heads
        self.max_bias_length = max_bias_length
        # 使用 Embedding 层
        self.bias_embed = nn.Embedding(max_bias_length, num_heads)
        self.bias_embed.weight.data = self.bias_embed.weight.data.to(dtype)

    def forward(self, q, k, v, tau):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        L = q.shape[-2]

        # 使用 embedding 索引
        rel_pos = torch.arange(L, device=q.device)[:, None] - torch.arange(L, device=q.device)[None, :]
        causal_mask = rel_pos >= 0
        bias_mask = causal_mask & (rel_pos < self.max_bias_length)
        indices = rel_pos.clamp(0, self.max_bias_length - 1)

        # Embedding: [L, L] -> [L, L, H]
        bias_values = self.bias_embed(indices)  # [L, L, H]
        bias_values = bias_values.permute(2, 0, 1)  # [H, L, L]
        bias_matrix = bias_values * bias_mask.to(scores.dtype)
        scores = scores + bias_matrix.unsqueeze(0)

        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        tau_view = tau.view(1, -1, 1, 1)
        i_pos = torch.arange(1, L + 1, device=q.device, dtype=attn.dtype).view(1, 1, -1, 1)
        attn = F.relu(attn + tau_view / i_pos)

        return torch.matmul(attn, v)

# ============================================================
# 方法 3: 预计算完整 Toeplitz 矩阵
# ============================================================
def precompute_bias_matrix(bias, L, max_bias_length, device, dtype):
    """预计算完整的 bias 矩阵 [H, L, L]"""
    H = bias.shape[0]
    rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
    causal_mask = rel_pos >= 0
    bias_mask = causal_mask & (rel_pos < max_bias_length)
    indices = rel_pos.clamp(0, max_bias_length - 1)

    # 预计算时不需要梯度
    with torch.no_grad():
        bias_matrix = bias[:, indices] * bias_mask.to(dtype)

    return bias_matrix, causal_mask

def lazy_attention_precomputed(q, k, v, bias_matrix, causal_mask, tau):
    """使用预计算的 bias 矩阵"""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # 直接加上预计算的 bias（但这样 bias 没有梯度...）
    scores = scores + bias_matrix.unsqueeze(0)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    L = q.shape[-2]
    tau_view = tau.view(1, -1, 1, 1)
    i_pos = torch.arange(1, L + 1, device=q.device, dtype=attn.dtype).view(1, 1, -1, 1)
    attn = F.relu(attn + tau_view / i_pos)

    return torch.matmul(attn, v)

# ============================================================
# 方法 4: 使用 torch.compile
# ============================================================
lazy_attention_compiled = torch.compile(lazy_attention_original, mode='reduce-overhead')

# ============================================================
# 方法 5: 手动优化 - 避免每次重新创建 rel_pos
# ============================================================
# 预计算静态张量
_rel_pos_cache = {}
_causal_mask_cache = {}
_i_pos_cache = {}

def get_cached_tensors(L, device, dtype):
    key = (L, device)
    if key not in _rel_pos_cache:
        rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
        _rel_pos_cache[key] = rel_pos
        _causal_mask_cache[key] = rel_pos >= 0
        _i_pos_cache[key] = torch.arange(1, L + 1, device=device, dtype=dtype).view(1, 1, -1, 1)
    return _rel_pos_cache[key], _causal_mask_cache[key], _i_pos_cache[key]

def lazy_attention_cached(q, k, v, bias, tau):
    L = q.shape[-2]
    rel_pos, causal_mask, i_pos = get_cached_tensors(L, q.device, q.dtype)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    bias_mask = causal_mask & (rel_pos < max_bias_length)
    indices = rel_pos.clamp(0, max_bias_length - 1)
    bias_matrix = bias[:, indices] * bias_mask.to(scores.dtype)
    scores = scores + bias_matrix.unsqueeze(0)

    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    tau_view = tau.view(1, -1, 1, 1)
    attn = F.relu(attn + tau_view / i_pos.to(attn.dtype))

    return torch.matmul(attn, v)

# ============================================================
# 方法 6: 使用 unfold 构建 Toeplitz 矩阵（实验性）
# ============================================================
def lazy_attention_unfold(q, k, v, bias, tau):
    """尝试用 as_strided 避免 gather"""
    L = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    H = bias.shape[0]
    # 构建 Toeplitz 矩阵：使用 as_strided
    # bias: [H, max_bias_length]
    # 需要的: [H, L, L] 其中 out[h, i, j] = bias[h, i-j] if 0 <= i-j < max_bias_length else 0

    # Pad bias 以支持 as_strided
    padded_bias = torch.cat([torch.zeros(H, L-1, device=bias.device, dtype=bias.dtype), bias], dim=1)
    # padded_bias: [H, L-1 + max_bias_length]

    # 使用 unfold 或 as_strided
    # 实际上对于 Toeplitz 矩阵，这个方法可能更复杂
    # 先用简单方法
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


print("=" * 60)
print("PyTorch Naive Lazy Attention 优化测试")
print(f"B={B}, H={H}, L={L}, D={D}, max_bias_length={max_bias_length}")
print("=" * 60)

# 基准：标准 softmax attention
def softmax_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    L = q.shape[-2]
    causal_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)

print("\n--- Baseline ---")
softmax_time = benchmark("Softmax Attention", softmax_attention, q, k, v)

print("\n--- Lazy Attention Variants ---")
original_time = benchmark("1. Original", lazy_attention_original, q, k, v, bias, tau)

# 方法 2: Embedding
embed_model = LazyAttentionEmbedding(H, max_bias_length, dtype).to(device)
embed_model.bias_embed.weight.data = bias.t().detach().clone()  # 复制 bias 参数
embed_time = benchmark("2. Embedding", embed_model, q, k, v, tau)

# 方法 4: torch.compile (需要预热更多)
print("3. Compiling with torch.compile (this may take a while)...")
try:
    # 额外预热 compile
    for _ in range(3):
        out = lazy_attention_compiled(q, k, v, bias, tau)
        out.sum().backward()
        for t in [q, k, v, bias, tau]:
            if t.grad is not None:
                t.grad = None
    compile_time = benchmark("3. torch.compile", lazy_attention_compiled, q, k, v, bias, tau, warmup=3)
except Exception as e:
    print(f"3. torch.compile: Failed - {e}")
    compile_time = None

# 方法 5: Cached tensors
cached_time = benchmark("4. Cached tensors", lazy_attention_cached, q, k, v, bias, tau)

print("\n--- Summary ---")
print(f"Softmax Attention: {softmax_time:.2f}ms (baseline)")
print(f"1. Original:       {original_time:.2f}ms ({original_time/softmax_time*100:.1f}%)")
print(f"2. Embedding:      {embed_time:.2f}ms ({embed_time/softmax_time*100:.1f}%)")
if compile_time:
    print(f"3. torch.compile:  {compile_time:.2f}ms ({compile_time/softmax_time*100:.1f}%)")
print(f"4. Cached tensors: {cached_time:.2f}ms ({cached_time/softmax_time*100:.1f}%)")

# 对比 Triton
print("\n--- Triton Kernel Comparison ---")
try:
    from lazy_attention_triton import lazy_attention_triton
    triton_time = benchmark("Lazy Triton", lazy_attention_triton, q, k, v, bias.to(q.dtype), tau.to(q.dtype))
    print(f"\nTriton vs PyTorch Original: {triton_time/original_time*100:.1f}%")
except ImportError as e:
    print(f"Triton not available: {e}")
