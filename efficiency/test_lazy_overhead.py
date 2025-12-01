"""
测试 Lazy Attention 相比标准 Softmax Attention 的额外开销
"""
import torch
import torch.nn.functional as F
import time

B, H, L, D = 4, 16, 4096, 64
device = 'cuda'
dtype = torch.bfloat16

q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)

# Warmup
for _ in range(5):
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    out.sum().backward()

torch.cuda.synchronize()

# 1. Standard softmax attention (PyTorch naive)
start = time.perf_counter()
for _ in range(20):
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
    causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
    out = torch.matmul(attn, v)
    out.sum().backward()
torch.cuda.synchronize()
softmax_time = (time.perf_counter() - start) / 20 * 1000
print(f'Standard Softmax PyTorch: {softmax_time:.2f}ms')

# 2. Lazy attention (with bias + relu)
max_bias_length = 4096
bias = torch.randn(H, max_bias_length, device=device, dtype=dtype, requires_grad=True)
tau = torch.full((H,), -1.0, device=device, dtype=dtype, requires_grad=True)

start = time.perf_counter()
for _ in range(20):
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
    rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
    causal_mask = rel_pos >= 0
    bias_mask = causal_mask & (rel_pos < max_bias_length)
    indices = rel_pos.clamp(0, max_bias_length - 1)
    bias_matrix = bias[:, indices] * bias_mask.to(scores.dtype)
    scores = scores + bias_matrix.unsqueeze(0)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
    tau_view = tau.view(1, -1, 1, 1)
    i_pos = torch.arange(1, L + 1, device=device, dtype=attn.dtype).view(1, 1, -1, 1)
    attn = F.relu(attn + tau_view / i_pos)
    out = torch.matmul(attn, v)
    out.sum().backward()
torch.cuda.synchronize()
lazy_time = (time.perf_counter() - start) / 20 * 1000
print(f'Lazy Attention PyTorch: {lazy_time:.2f}ms')
print(f'Overhead: {lazy_time / softmax_time * 100:.1f}%')

# 3. Test Triton kernel
print("\n--- Triton Kernel ---")
try:
    from lazy_attention_triton import lazy_attention_triton

    # Warmup
    for _ in range(5):
        out = lazy_attention_triton(q, k, v, bias.to(q.dtype), tau.to(q.dtype))
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        out = lazy_attention_triton(q, k, v, bias.to(q.dtype), tau.to(q.dtype))
        out.sum().backward()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 20 * 1000
    print(f'Lazy Attention Triton: {triton_time:.2f}ms')
    print(f'vs Softmax PyTorch: {triton_time / softmax_time * 100:.1f}%')
except ImportError as e:
    print(f"Triton kernel not available: {e}")

# 4. Test Flash Attention baseline
print("\n--- Flash Attention Baseline ---")
try:
    from flash_attn import flash_attn_func

    q_flash = q.transpose(1, 2).contiguous()  # B, L, H, D
    k_flash = k.transpose(1, 2).contiguous()
    v_flash = v.transpose(1, 2).contiguous()

    # Warmup
    for _ in range(5):
        out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
        out.sum().backward()
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - start) / 20 * 1000
    print(f'Flash Attention: {flash_time:.2f}ms')
    print(f'Lazy Triton vs Flash: {triton_time / flash_time * 100:.1f}%' if 'triton_time' in dir() else '')
except ImportError as e:
    print(f"Flash Attention not available: {e}")
