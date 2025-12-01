# -*- coding: utf-8 -*-
"""
Full Model Efficiency Benchmark

Compares ENTIRE Transformer model (not just attention):
- Attention + FFN + LayerNorm + Residual connections

Shows percentage breakdown:
- Total model = 100%
- Attention = X%
- FFN = Y%
- Other (LayerNorm, Residual, Projections) = Z%

For each attention type (Softmax, Entmax, SWAT), compares:
- Standard PyTorch implementation (naive)
- Flash/Triton optimized implementation

Model config: 340M parameters
- hidden_size: 1024
- num_heads: 16
- num_layers: 24
- head_dim: 64
- intermediate_size: 4096 (4x hidden_size)
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, '/home/fzkuji/PycharmProjects/flash-lazy-attention')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Callable
import time


@dataclass
class ModelConfig:
    """340M model configuration"""
    batch_size: int = 4
    seq_len: int = 4096
    hidden_size: int = 1024
    num_heads: int = 16
    head_dim: int = 64
    intermediate_size: int = 4096  # 4x hidden_size for FFN
    max_bias_length: int = 4096  # for SWAT bias


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model Components
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm as used in modern LLMs"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class FFN(nn.Module):
    """Feed-Forward Network with SwiGLU activation (as in LLaMA)"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================
# Attention Implementations
# ============================================================

class SoftmaxAttention(nn.Module):
    """Standard softmax attention"""
    def __init__(self, config: ModelConfig, use_flash: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.use_flash = use_flash

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        if self.use_flash:
            # Use Flash Attention
            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, causal=True)
        else:
            # PyTorch naive
            q = q.transpose(1, 2)  # B, H, L, D
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2)

        out = out.reshape(B, L, -1)
        return self.o_proj(out)


class EntmaxAttention(nn.Module):
    """Entmax (α=1.5) attention using AdaSplash"""
    def __init__(self, config: ModelConfig, use_flash: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.use_flash = use_flash

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _entmax_naive(self, scores, alpha=1.5):
        """Sparsemax approximation (entmax with α≈1.5)"""
        # Convert to float32 for numerical stability
        original_dtype = scores.dtype
        scores = scores.float()

        B, H, M, N = scores.shape
        sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=-1)
        k = torch.arange(1, N + 1, device=scores.device, dtype=scores.dtype)
        tau = (cumsum - 1) / k
        support = sorted_scores > tau
        k_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
        tau_star = (scores.sum(dim=-1, keepdim=True) - 1) / k_support.float()
        output = torch.clamp(scores - tau_star, min=0)
        output = output / (output.sum(dim=-1, keepdim=True) + 1e-10)

        return output.to(original_dtype)

    def forward(self, x):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        if self.use_flash:
            # AdaSplash Triton kernel
            from adasplash.adasplash_no_block_mask import sparse_attn
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out = sparse_attn(q, k, v, alpha=1.5, is_causal=True)
            out = out.transpose(1, 2)
        else:
            # PyTorch naive
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

            attn = self._entmax_naive(scores)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2)

        out = out.reshape(B, L, -1)
        return self.o_proj(out)


class SWATAttention(nn.Module):
    """SWAT/Lazy attention with learnable bias and elastic softmax"""
    def __init__(self, config: ModelConfig, use_flash: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_bias_length = config.max_bias_length
        self.use_flash = use_flash

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Learnable bias and tau
        self.bias = nn.Parameter(torch.zeros(config.num_heads, config.max_bias_length))
        self.tau = nn.Parameter(torch.full((config.num_heads,), -1.0))
        nn.init.normal_(self.bias, mean=0.0, std=1e-3)

    def forward(self, x):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        q = q.transpose(1, 2).contiguous()  # B, H, L, D
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if self.use_flash:
            # Lazy Attention Triton kernel
            from adasplash.lazy_attention_triton import lazy_attention_triton
            out = lazy_attention_triton(q, k, v, self.bias.to(q.dtype), self.tau.to(q.dtype))
        else:
            # PyTorch naive
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Add distance-based bias
            positions = torch.arange(L, device=x.device)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)

            dist_clamped = dist.clamp(0, self.max_bias_length - 1)
            in_window = (dist >= 0) & (dist < self.max_bias_length)

            bias_matrix = self.bias[:, dist_clamped]
            bias_matrix = bias_matrix * in_window.float()

            scores = scores + bias_matrix.unsqueeze(0)

            # Causal mask
            causal_mask = dist >= 0
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Softmax
            attn = F.softmax(scores, dim=-1)

            # Elastic softmax: ReLU(attn + tau/i)
            idx_i = torch.arange(1, L + 1, device=x.device, dtype=attn.dtype)
            tau_term = self.tau.to(attn.dtype).view(1, -1, 1, 1) / idx_i.view(1, 1, L, 1)
            attn = torch.relu(attn + tau_term)

            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


# ============================================================
# Full Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """Complete Transformer block with attention + FFN"""
    def __init__(self, config: ModelConfig, attention_module: nn.Module):
        super().__init__()
        self.attention = attention_module
        self.ffn = FFN(config.hidden_size, config.intermediate_size)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x):
        # Pre-norm architecture (as in LLaMA)
        h = x + self.attention(self.norm1(x))
        out = h + self.ffn(self.norm2(h))
        return out


# ============================================================
# Benchmark Functions
# ============================================================

def benchmark_component(
    func: Callable,
    inputs: tuple,
    n_warmup: int = 5,
    n_iters: int = 20,
    backward: bool = False
) -> float:
    """Benchmark a function/module"""
    device = next(iter(inputs[0].parameters())).device if isinstance(inputs[0], nn.Module) else inputs[0].device

    # Warmup
    for _ in range(n_warmup):
        if backward:
            x = inputs[1].detach().clone().requires_grad_(True)
            out = func(x)
            grad_out = torch.randn_like(out)
            out.backward(grad_out)
        else:
            with torch.no_grad():
                out = func(inputs[1])
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(n_iters):
        if backward:
            x = inputs[1].detach().clone().requires_grad_(True)
            out = func(x)
            grad_out = torch.randn_like(out)
            out.backward(grad_out)
        else:
            with torch.no_grad():
                out = func(inputs[1])

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / n_iters


def benchmark_with_breakdown(
    block: TransformerBlock,
    x: torch.Tensor,
    n_warmup: int = 5,
    n_iters: int = 20,
    backward: bool = False
) -> Dict[str, float]:
    """Benchmark with component breakdown"""
    device = x.device
    results = {}

    # 1. Total block time
    def full_forward(inp):
        return block(inp)

    total_time = benchmark_component(full_forward, (block, x), n_warmup, n_iters, backward)
    results['total'] = total_time

    # 2. Attention only (norm1 + attention + residual)
    def attention_forward(inp):
        h = inp + block.attention(block.norm1(inp))
        return h

    attn_time = benchmark_component(attention_forward, (block, x), n_warmup, n_iters, backward)
    results['attention'] = attn_time

    # 3. FFN only (norm2 + FFN + residual) - use output from attention
    h_for_ffn = x + block.attention(block.norm1(x))
    h_for_ffn = h_for_ffn.detach()

    def ffn_forward(inp):
        return inp + block.ffn(block.norm2(inp))

    ffn_time = benchmark_component(ffn_forward, (block, h_for_ffn), n_warmup, n_iters, backward)
    results['ffn'] = ffn_time

    # Compute percentages
    results['attention_pct'] = attn_time / total_time * 100
    results['ffn_pct'] = ffn_time / total_time * 100

    return results


def run_full_benchmark(config: ModelConfig, mode: str = "training") -> Dict[str, Dict]:
    """Run benchmarks for all attention types"""
    device = get_device()
    dtype = torch.bfloat16
    backward = (mode == "training")

    print(f"\n{'='*70}")
    print(f"Running {mode.upper()} benchmarks (Full Model)")
    print(f"Config: B={config.batch_size}, L={config.seq_len}, H={config.hidden_size}, "
          f"Heads={config.num_heads}, FFN={config.intermediate_size}")
    print(f"{'='*70}")

    # Create input
    x = torch.randn(config.batch_size, config.seq_len, config.hidden_size,
                    device=device, dtype=dtype)

    results = {}

    attention_types = [
        ("Softmax", SoftmaxAttention),
        ("Entmax", EntmaxAttention),
        ("SWAT", SWATAttention),
    ]

    implementations = [
        ("PyTorch", False),
        ("Flash", True),
    ]

    for attn_name, attn_class in attention_types:
        for impl_name, use_flash in implementations:
            key = f"{attn_name}-{impl_name}"
            print(f"\n[{key}]")

            try:
                # Create attention module
                attention = attn_class(config, use_flash=use_flash).to(device).to(dtype)

                # Create full block
                block = TransformerBlock(config, attention).to(device).to(dtype)

                # Benchmark
                breakdown = benchmark_with_breakdown(block, x, n_warmup=5, n_iters=20, backward=backward)
                results[key] = breakdown

                print(f"  Total:     {breakdown['total']:.2f} ms")
                print(f"  Attention: {breakdown['attention']:.2f} ms ({breakdown['attention_pct']:.1f}%)")
                print(f"  FFN:       {breakdown['ffn']:.2f} ms ({breakdown['ffn_pct']:.1f}%)")

            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = {
                    'total': float('nan'),
                    'attention': float('nan'),
                    'ffn': float('nan'),
                    'attention_pct': float('nan'),
                    'ffn_pct': float('nan'),
                }

    return results


def plot_model_comparison(
    results_training: Dict,
    results_inference: Dict,
    config: ModelConfig,
    save_path: str
):
    """Create comprehensive comparison chart"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    attention_types = ['Softmax', 'Entmax', 'SWAT']
    implementations = ['PyTorch', 'Flash']

    # Get baseline (Softmax-Flash total)
    baseline_train = results_training.get('Softmax-Flash', {}).get('total', 1.0)
    baseline_infer = results_inference.get('Softmax-Flash', {}).get('total', 1.0)

    for row, (results, baseline, mode_title) in enumerate([
        (results_training, baseline_train, 'Training'),
        (results_inference, baseline_infer, 'Inference')
    ]):
        # Left: Stacked bar (Attention + FFN breakdown)
        ax_stack = axes[row, 0]

        x_labels = []
        attn_vals = []
        ffn_vals = []

        for attn_name in attention_types:
            for impl_name in implementations:
                key = f"{attn_name}-{impl_name}"
                data = results.get(key, {})

                total = data.get('total', float('nan'))
                attn_time = data.get('attention', float('nan'))
                ffn_time = data.get('ffn', float('nan'))

                if np.isnan(total):
                    attn_vals.append(0)
                    ffn_vals.append(0)
                else:
                    attn_vals.append(attn_time)
                    ffn_vals.append(ffn_time)

                x_labels.append(f"{attn_name}\n{impl_name}")

        x = np.arange(len(x_labels))
        width = 0.6

        bars_attn = ax_stack.bar(x, attn_vals, width, label='Attention', color='#4472C4')
        bars_ffn = ax_stack.bar(x, ffn_vals, width, bottom=attn_vals, label='FFN', color='#ED7D31')

        # Add total time labels
        for i, (attn, ffn) in enumerate(zip(attn_vals, ffn_vals)):
            total = attn + ffn
            if total > 0:
                ax_stack.annotate(f'{total:.1f}ms',
                                 xy=(i, total),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_stack.set_ylabel('Time (ms)', fontsize=11)
        ax_stack.set_title(f'{mode_title}: Component Breakdown', fontsize=13, fontweight='bold')
        ax_stack.set_xticks(x)
        ax_stack.set_xticklabels(x_labels, fontsize=9)
        ax_stack.legend(loc='upper right')
        ax_stack.grid(axis='y', alpha=0.3)

        # Right: Percentage comparison (relative to Softmax-Flash)
        ax_pct = axes[row, 1]

        totals_pct = []
        colors = []
        labels = []

        color_map = {
            'Softmax-PyTorch': '#A5C8E1',
            'Softmax-Flash': '#4472C4',
            'Entmax-PyTorch': '#C5E0B4',
            'Entmax-Flash': '#70AD47',
            'SWAT-PyTorch': '#F8CBAD',
            'SWAT-Flash': '#ED7D31',
        }

        for attn_name in attention_types:
            for impl_name in implementations:
                key = f"{attn_name}-{impl_name}"
                data = results.get(key, {})
                total = data.get('total', float('nan'))

                if np.isnan(total) or baseline == 0:
                    totals_pct.append(0)
                else:
                    totals_pct.append(total / baseline * 100)

                colors.append(color_map.get(key, '#808080'))
                labels.append(f"{attn_name}\n{impl_name}")

        x = np.arange(len(labels))
        bars = ax_pct.bar(x, totals_pct, color=colors, edgecolor='black', linewidth=0.8)

        # Add percentage labels
        for bar, pct in zip(bars, totals_pct):
            if pct > 0:
                height = bar.get_height()
                ax_pct.annotate(f'{pct:.0f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Baseline reference
        ax_pct.axhline(y=100, color='red', linestyle='--', linewidth=2,
                       label='Baseline (Softmax-Flash)')

        ax_pct.set_ylabel('Relative Time (%)', fontsize=11)
        ax_pct.set_title(f'{mode_title}: Relative to Softmax-Flash', fontsize=13, fontweight='bold')
        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(labels, fontsize=9)
        ax_pct.legend(loc='upper right')
        ax_pct.grid(axis='y', alpha=0.3)
        ax_pct.set_ylim(0, max(totals_pct) * 1.15 if max(totals_pct) > 0 else 150)

    fig.suptitle(f'Full Model Efficiency Comparison (340M Config)\n'
                 f'B={config.batch_size}, L={config.seq_len}, H={config.hidden_size}, FFN={config.intermediate_size}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.close()


def plot_flash_comparison(
    results_training: Dict,
    results_inference: Dict,
    config: ModelConfig,
    save_path: str
):
    """Create clean chart comparing only Flash implementations (for paper)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    attention_types = ['Softmax', 'Entmax', 'SWAT']
    impl_labels = ['Softmax\n(FlashAttn)', 'Entmax\n(AdaSplash)', 'SWAT\n(Ours)']
    colors = ['#4472C4', '#70AD47', '#ED7D31']

    for idx, (ax, results, title) in enumerate([
        (axes[0], results_training, 'Training'),
        (axes[1], results_inference, 'Inference')
    ]):
        # Get baseline
        baseline = results.get('Softmax-Flash', {}).get('total', 1.0)

        # Collect data for Flash implementations
        totals = []
        attn_pcts = []
        ffn_pcts = []

        for attn_name in attention_types:
            key = f"{attn_name}-Flash"
            data = results.get(key, {})

            total = data.get('total', float('nan'))
            attn_pct = data.get('attention_pct', float('nan'))
            ffn_pct = data.get('ffn_pct', float('nan'))

            totals.append(total if not np.isnan(total) else 0)
            attn_pcts.append(attn_pct if not np.isnan(attn_pct) else 0)
            ffn_pcts.append(ffn_pct if not np.isnan(ffn_pct) else 0)

        # Convert to percentage relative to baseline
        pcts = [(t / baseline * 100) if baseline > 0 else 0 for t in totals]

        x = np.arange(len(impl_labels))
        bars = ax.bar(x, pcts, color=colors, edgecolor='black', linewidth=1.2)

        # Add labels with time and breakdown
        for i, (bar, pct, total, attn_p, ffn_p) in enumerate(zip(bars, pcts, totals, attn_pcts, ffn_pcts)):
            if pct > 0:
                height = bar.get_height()
                ax.annotate(f'{pct:.0f}%\n({total:.1f}ms)\n[Attn:{attn_p:.0f}%]',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.axhline(y=100, color='#C00000', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_ylabel('Relative Time (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(impl_labels, fontsize=11)
        ax.set_ylim(0, max(pcts) * 1.25 if max(pcts) > 0 else 150)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Full Model Efficiency (Flash/Triton Implementations Only)\n'
                 f'340M Config: B={config.batch_size}, L={config.seq_len}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {save_path}")
    plt.close()


def print_summary(results_training: Dict, results_inference: Dict):
    """Print summary table"""
    print("\n" + "="*70)
    print("SUMMARY - Full Model Benchmark")
    print("="*70)

    baseline_train = results_training.get('Softmax-Flash', {}).get('total', 1.0)
    baseline_infer = results_inference.get('Softmax-Flash', {}).get('total', 1.0)

    print(f"\n{'Method':<20} {'Train (ms)':<12} {'Train %':<10} {'Infer (ms)':<12} {'Infer %':<10}")
    print("-"*70)

    for key in ['Softmax-PyTorch', 'Softmax-Flash', 'Entmax-PyTorch', 'Entmax-Flash',
                'SWAT-PyTorch', 'SWAT-Flash']:
        train_data = results_training.get(key, {})
        infer_data = results_inference.get(key, {})

        train_total = train_data.get('total', float('nan'))
        infer_total = infer_data.get('total', float('nan'))

        train_pct = (train_total / baseline_train * 100) if not np.isnan(train_total) else float('nan')
        infer_pct = (infer_total / baseline_infer * 100) if not np.isnan(infer_total) else float('nan')

        train_str = f"{train_total:.2f}" if not np.isnan(train_total) else "N/A"
        infer_str = f"{infer_total:.2f}" if not np.isnan(infer_total) else "N/A"
        train_pct_str = f"{train_pct:.0f}%" if not np.isnan(train_pct) else "N/A"
        infer_pct_str = f"{infer_pct:.0f}%" if not np.isnan(infer_pct) else "N/A"

        print(f"{key:<20} {train_str:<12} {train_pct_str:<10} {infer_str:<12} {infer_pct_str:<10}")

    print("\n" + "-"*70)
    print("Component Breakdown (Flash implementations):")
    print("-"*70)
    print(f"{'Method':<20} {'Attn %':<15} {'FFN %':<15}")
    print("-"*70)

    for attn_name in ['Softmax', 'Entmax', 'SWAT']:
        key = f"{attn_name}-Flash"
        train_data = results_training.get(key, {})

        attn_pct = train_data.get('attention_pct', float('nan'))
        ffn_pct = train_data.get('ffn_pct', float('nan'))

        attn_str = f"{attn_pct:.1f}%" if not np.isnan(attn_pct) else "N/A"
        ffn_str = f"{ffn_pct:.1f}%" if not np.isnan(ffn_pct) else "N/A"

        print(f"{key:<20} {attn_str:<15} {ffn_str:<15}")


def main():
    config = ModelConfig(
        batch_size=4,
        seq_len=4096,
        hidden_size=1024,
        num_heads=16,
        head_dim=64,
        intermediate_size=4096,
        max_bias_length=4096,
    )

    print("="*70)
    print("FULL MODEL EFFICIENCY BENCHMARK")
    print("="*70)
    print(f"Model: 340M (hidden={config.hidden_size}, heads={config.num_heads}, "
          f"ffn={config.intermediate_size})")
    print(f"Sequence length: {config.seq_len}")
    print(f"Batch size: {config.batch_size}")
    print("="*70)

    # Run benchmarks
    results_training = run_full_benchmark(config, mode="training")
    results_inference = run_full_benchmark(config, mode="inference")

    # Print summary
    print_summary(results_training, results_inference)

    # Generate plots
    output_dir = os.path.dirname(os.path.abspath(__file__))

    plot_model_comparison(
        results_training, results_inference, config,
        os.path.join(output_dir, 'model_efficiency_comparison.png')
    )

    plot_flash_comparison(
        results_training, results_inference, config,
        os.path.join(output_dir, 'model_efficiency_flash_only.png')
    )

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
