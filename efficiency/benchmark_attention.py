# -*- coding: utf-8 -*-
"""
Attention Efficiency Benchmark

Compares:
1. Standard Transformer (Softmax Attention)
2. Entmax Attention (α=1.5)
3. SWAT / Lazy Attention (ReLU + learnable bias)

For each method, benchmarks both:
- PyTorch naive implementation
- Flash/Triton optimized implementation

Model config: 340M parameters
- hidden_size: 1024
- num_heads: 16
- num_layers: 24
- head_dim: 64
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, '/home/fzkuji/PycharmProjects/flash-lazy-attention')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time


@dataclass
class ModelConfig:
    """340M model configuration"""
    batch_size: int = 4
    num_heads: int = 16
    seq_len: int = 4096
    head_dim: int = 64
    window_size: int = 4096  # for SWAT bias - use full seq_len for fair comparison


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PyTorch Naive Implementations
# ============================================================

def softmax_attention_pytorch(q, k, v):
    """Standard softmax attention (PyTorch naive)"""
    B, H, L, D = q.shape
    scale = D ** -0.5

    # QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # Output
    out = torch.matmul(attn, v)
    return out


def entmax_pytorch(scores, alpha=1.5):
    """Entmax activation (PyTorch naive implementation)"""
    # Simple bisection-based entmax
    # This is slower but correct for benchmarking
    B, H, M, N = scores.shape

    # For simplicity, use sparsemax approximation when alpha close to 1.5
    # True entmax would require iterative solver
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_scores, dim=-1)
    k = torch.arange(1, N + 1, device=scores.device, dtype=scores.dtype)
    tau = (cumsum - 1) / k

    # Find support
    support = sorted_scores > tau
    k_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau_star = (scores.sum(dim=-1, keepdim=True) - 1) / k_support.float()

    # Compute output
    output = torch.clamp(scores - tau_star, min=0)
    output = output / (output.sum(dim=-1, keepdim=True) + 1e-10)

    return output


def entmax_attention_pytorch(q, k, v, alpha=1.5):
    """Entmax attention (PyTorch naive)"""
    B, H, L, D = q.shape
    scale = D ** -0.5

    # QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))

    # Entmax
    attn = entmax_pytorch(scores, alpha)

    # Output
    out = torch.matmul(attn, v)
    return out


def swat_attention_pytorch(q, k, v, bias, tau, window_size):
    """SWAT/Lazy attention (PyTorch naive)"""
    B, H, L, D = q.shape
    scale = D ** -0.5

    # QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Add distance-based bias
    positions = torch.arange(L, device=q.device)
    dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # [L, L]

    # Clamp distance and get bias
    dist_clamped = dist.clamp(0, window_size - 1)
    in_window = (dist >= 0) & (dist < window_size)

    bias_matrix = bias[:, dist_clamped]  # [H, L, L]
    bias_matrix = bias_matrix * in_window.float()

    scores = scores + bias_matrix.unsqueeze(0)

    # Causal mask
    causal_mask = dist >= 0
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # ReLU(attn + tau/i)
    idx_i = torch.arange(1, L + 1, device=q.device, dtype=torch.float32)
    tau_term = tau.view(1, H, 1, 1) / idx_i.view(1, 1, L, 1)
    attn_elastic = torch.relu(attn + tau_term)

    # Output
    out = torch.matmul(attn_elastic, v)
    return out


# ============================================================
# Flash/Triton Implementations
# ============================================================

def softmax_attention_flash(q, k, v):
    """Standard attention using Flash Attention"""
    try:
        from flash_attn import flash_attn_func
        # flash_attn expects (B, L, H, D) format
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q_t, k_t, v_t, causal=True)
        return out.transpose(1, 2)
    except ImportError:
        # Fallback to PyTorch SDPA
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out


def entmax_attention_flash(q, k, v, alpha):
    """Entmax attention using AdaSplash Triton kernel"""
    from adasplash.adasplash_no_block_mask import sparse_attn
    return sparse_attn(q, k, v, alpha=alpha, is_causal=True)


def swat_attention_flash(q, k, v, bias, tau, window_size):
    """SWAT attention using optimized Triton kernel"""
    from adasplash.lazy_attention_triton import lazy_attention_triton
    return lazy_attention_triton(q, k, v, bias, tau, window_size)


# ============================================================
# Benchmark Functions
# ============================================================

def clone_for_backward(inputs):
    """Clone inputs for backward pass, enabling grad for tensors"""
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            if x.dtype.is_floating_point and x.dim() > 0:
                result.append(x.detach().clone().requires_grad_(True))
            else:
                result.append(x)
        else:
            result.append(x)
    return result


def benchmark_function(func, inputs, n_warmup=5, n_iters=20, backward=False):
    """Benchmark a function with optional backward pass"""
    device = inputs[0].device if isinstance(inputs[0], torch.Tensor) else torch.device('cuda')

    # Warmup
    for _ in range(n_warmup):
        if backward:
            grad_inputs = clone_for_backward(inputs)
            out = func(*grad_inputs)
            grad_out = torch.randn_like(out)
            out.backward(grad_out)
        else:
            out = func(*inputs)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(n_iters):
        if backward:
            grad_inputs = clone_for_backward(inputs)
            out = func(*grad_inputs)
            grad_out = torch.randn_like(out)
            out.backward(grad_out)
        else:
            out = func(*inputs)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / n_iters
    return elapsed_ms


def run_benchmarks(config: ModelConfig, mode: str = "training") -> Dict[str, Dict[str, float]]:
    """
    Run all benchmarks

    mode: "training" (forward + backward) or "inference" (forward only)
    """
    device = get_device()
    dtype = torch.bfloat16
    backward = (mode == "training")

    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} benchmarks")
    print(f"Config: B={config.batch_size}, H={config.num_heads}, L={config.seq_len}, D={config.head_dim}")
    print(f"{'='*60}")

    # Create inputs
    q = torch.randn(config.batch_size, config.num_heads, config.seq_len, config.head_dim,
                    device=device, dtype=dtype, requires_grad=backward)
    k = torch.randn_like(q, requires_grad=backward)
    v = torch.randn_like(q, requires_grad=backward)

    # SWAT-specific inputs
    bias = torch.randn(config.num_heads, config.window_size, device=device, dtype=dtype, requires_grad=backward)
    tau = torch.full((config.num_heads,), -1.0, device=device, dtype=dtype, requires_grad=backward)

    results = {}

    # 1. Standard Transformer
    print("\n[Softmax Attention]")

    # PyTorch
    try:
        time_pytorch = benchmark_function(
            softmax_attention_pytorch, [q, k, v], backward=backward
        )
        print(f"  PyTorch: {time_pytorch:.2f} ms")
        results['Softmax-PyTorch'] = time_pytorch
    except Exception as e:
        print(f"  PyTorch: FAILED ({e})")
        results['Softmax-PyTorch'] = float('nan')

    # Flash
    try:
        time_flash = benchmark_function(
            softmax_attention_flash, [q, k, v], backward=backward
        )
        print(f"  Flash:   {time_flash:.2f} ms")
        results['Softmax-Flash'] = time_flash
    except Exception as e:
        print(f"  Flash:   FAILED ({e})")
        results['Softmax-Flash'] = float('nan')

    # 2. Entmax Attention
    print("\n[Entmax Attention (α=1.5)]")

    # PyTorch (only forward for now - backward is complex)
    if not backward:
        try:
            q_fp32 = q.float()
            k_fp32 = k.float()
            v_fp32 = v.float()
            time_pytorch = benchmark_function(
                entmax_attention_pytorch, [q_fp32, k_fp32, v_fp32, 1.5], backward=False
            )
            print(f"  PyTorch: {time_pytorch:.2f} ms")
            results['Entmax-PyTorch'] = time_pytorch
        except Exception as e:
            print(f"  PyTorch: FAILED ({e})")
            results['Entmax-PyTorch'] = float('nan')
    else:
        results['Entmax-PyTorch'] = float('nan')
        print(f"  PyTorch: N/A (no backward impl)")

    # Flash (AdaSplash) - alpha must be a float, not a tensor
    try:
        alpha_val = 1.5  # plain float
        def entmax_wrapper(q, k, v):
            return entmax_attention_flash(q, k, v, alpha_val)
        time_flash = benchmark_function(
            entmax_wrapper, [q, k, v], backward=backward
        )
        print(f"  Flash:   {time_flash:.2f} ms")
        results['Entmax-Flash'] = time_flash
    except Exception as e:
        print(f"  Flash:   FAILED ({e})")
        results['Entmax-Flash'] = float('nan')

    # 3. SWAT / Lazy Attention
    print("\n[SWAT Attention (ReLU + bias)]")

    # PyTorch (only inference - too slow for training benchmark)
    if not backward and config.seq_len <= 2048:
        try:
            time_pytorch = benchmark_function(
                swat_attention_pytorch, [q, k, v, bias, tau, config.window_size], backward=False
            )
            print(f"  PyTorch: {time_pytorch:.2f} ms")
            results['SWAT-PyTorch'] = time_pytorch
        except Exception as e:
            print(f"  PyTorch: FAILED ({e})")
            results['SWAT-PyTorch'] = float('nan')
    else:
        results['SWAT-PyTorch'] = float('nan')
        print(f"  PyTorch: N/A (too slow / no backward)")

    # Flash (Triton optimized) - window_size must be plain int
    try:
        ws = config.window_size  # plain int
        def swat_wrapper(q, k, v, bias, tau):
            return swat_attention_flash(q, k, v, bias, tau, ws)
        time_flash = benchmark_function(
            swat_wrapper, [q, k, v, bias, tau], backward=backward
        )
        print(f"  Flash:   {time_flash:.2f} ms")
        results['SWAT-Flash'] = time_flash
    except Exception as e:
        print(f"  Flash:   FAILED ({e})")
        results['SWAT-Flash'] = float('nan')

    return results


def plot_results(results_training: Dict, results_inference: Dict, config: ModelConfig, save_path: str):
    """
    Create stacked bar chart comparing attention methods

    Softmax-Flash is baseline (100%), other methods shown relative to it
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (ax, results, title) in enumerate([
        (axes[0], results_training, 'Training (Forward + Backward)'),
        (axes[1], results_inference, 'Inference (Forward Only)')
    ]):
        # Get baseline (Softmax-Flash)
        baseline = results.get('Softmax-Flash', 1.0)
        if np.isnan(baseline) or baseline == 0:
            baseline = results.get('Softmax-PyTorch', 1.0)

        # Methods to plot
        methods = ['Softmax', 'Entmax', 'SWAT']
        implementations = ['PyTorch', 'Flash']

        x = np.arange(len(methods))
        width = 0.35

        colors_pytorch = ['#ff9999', '#99ff99', '#9999ff']
        colors_flash = ['#cc0000', '#00cc00', '#0000cc']

        pytorch_times = []
        flash_times = []

        for method in methods:
            pt_key = f'{method}-PyTorch'
            fl_key = f'{method}-Flash'

            pt_val = results.get(pt_key, float('nan'))
            fl_val = results.get(fl_key, float('nan'))

            # Convert to percentage relative to baseline
            pt_pct = (pt_val / baseline * 100) if not np.isnan(pt_val) else 0
            fl_pct = (fl_val / baseline * 100) if not np.isnan(fl_val) else 0

            pytorch_times.append(pt_pct)
            flash_times.append(fl_pct)

        # Plot bars
        bars1 = ax.bar(x - width/2, pytorch_times, width, label='PyTorch (naive)',
                       color=['#ffcccc', '#ccffcc', '#ccccff'], edgecolor='black')
        bars2 = ax.bar(x + width/2, flash_times, width, label='Flash/Triton',
                       color=['#ff6666', '#66ff66', '#6666ff'], edgecolor='black')

        # Add value labels
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.annotate(f'{val:.0f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=10, fontweight='bold')

        add_labels(bars1, pytorch_times)
        add_labels(bars2, flash_times)

        # Add baseline reference line
        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Baseline (Softmax-Flash = 100%)')

        ax.set_ylabel('Relative Time (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(max(pytorch_times), max(flash_times)) * 1.2)
        ax.grid(axis='y', alpha=0.3)

    # Add overall title
    fig.suptitle(f'Attention Efficiency Comparison (340M Config: B={config.batch_size}, H={config.num_heads}, L={config.seq_len}, D={config.head_dim})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.close()


def plot_flash_only(results_training: Dict, results_inference: Dict, config: ModelConfig, save_path: str):
    """
    Create a clean bar chart comparing only Flash implementations (for paper)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (ax, results, title) in enumerate([
        (axes[0], results_training, 'Training'),
        (axes[1], results_inference, 'Inference')
    ]):
        # Get baseline (Softmax-Flash)
        baseline = results.get('Softmax-Flash', 1.0)

        # Only Flash implementations
        methods = ['Softmax\n(FlashAttn)', 'Entmax\n(AdaSplash)', 'SWAT\n(Ours)']
        keys = ['Softmax-Flash', 'Entmax-Flash', 'SWAT-Flash']
        colors = ['#4472C4', '#70AD47', '#ED7D31']  # Professional colors

        times = []
        percentages = []
        for key in keys:
            val = results.get(key, float('nan'))
            times.append(val if not np.isnan(val) else 0)
            pct = (val / baseline * 100) if not np.isnan(val) and baseline > 0 else 0
            percentages.append(pct)

        x = np.arange(len(methods))
        bars = ax.bar(x, percentages, color=colors, edgecolor='black', linewidth=1.2)

        # Add value labels with time in ms
        for bar, pct, t in zip(bars, percentages, times):
            if pct > 0:
                height = bar.get_height()
                # Show percentage and actual time
                ax.annotate(f'{pct:.0f}%\n({t:.1f}ms)',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add baseline reference line
        ax.axhline(y=100, color='#C00000', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_ylabel('Relative Time (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylim(0, max(percentages) * 1.25)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Attention Efficiency (Optimized Implementations)\nModel: 340M, Seq: {config.seq_len}, Batch: {config.batch_size}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {save_path}")
    plt.close()


def plot_detailed_results(results_training: Dict, results_inference: Dict, config: ModelConfig, save_path: str):
    """
    Create detailed bar chart with actual timing values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (ax, results, title) in enumerate([
        (axes[0], results_training, 'Training (Forward + Backward)'),
        (axes[1], results_inference, 'Inference (Forward Only)')
    ]):
        # Methods
        methods = []
        times = []
        colors = []

        color_map = {
            'Softmax-PyTorch': '#ffcccc',
            'Softmax-Flash': '#ff6666',
            'Entmax-PyTorch': '#ccffcc',
            'Entmax-Flash': '#66ff66',
            'SWAT-PyTorch': '#ccccff',
            'SWAT-Flash': '#6666ff',
        }

        for key in ['Softmax-PyTorch', 'Softmax-Flash', 'Entmax-PyTorch', 'Entmax-Flash', 'SWAT-PyTorch', 'SWAT-Flash']:
            val = results.get(key, float('nan'))
            if not np.isnan(val):
                methods.append(key)
                times.append(val)
                colors.append(color_map[key])

        x = np.arange(len(methods))
        bars = ax.bar(x, times, color=colors, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, times):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}ms',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Attention Timing Comparison (340M Config: B={config.batch_size}, H={config.num_heads}, L={config.seq_len}, D={config.head_dim})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")
    plt.close()


def main():
    # Configuration matching 340M model
    seq_len = 4096

    print("="*60)
    print("ATTENTION EFFICIENCY BENCHMARK")
    print("="*60)
    print(f"Model: 340M (hidden=1024, heads=16, layers=24)")
    print("="*60)

    # Test two scenarios:
    # 1. Fair comparison: window_size = seq_len (full causal attention)
    # 2. Practical use: window_size = 512 (windowed attention)

    all_results = {}

    for window_size, scenario in [(seq_len, "Full Causal (Fair Comparison)"), (512, "Windowed (Practical)")]:
        config = ModelConfig(
            batch_size=4,
            num_heads=16,
            seq_len=seq_len,
            head_dim=64,
            window_size=window_size
        )

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"Window size: {config.window_size}")
        print("="*60)

        results_training = run_benchmarks(config, mode="training")
        results_inference = run_benchmarks(config, mode="inference")

        all_results[scenario] = {
            'training': results_training,
            'inference': results_inference,
            'config': config
        }

    # Use full causal for main plots (fair comparison)
    fair_results = all_results["Full Causal (Fair Comparison)"]
    results_training = fair_results['training']
    results_inference = fair_results['inference']
    config = fair_results['config']

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_train = results_training.get('Softmax-Flash', 1.0)
    baseline_infer = results_inference.get('Softmax-Flash', 1.0)

    print("\nTraining (relative to Softmax-Flash):")
    for key, val in results_training.items():
        if not np.isnan(val):
            pct = val / baseline_train * 100
            print(f"  {key}: {val:.2f} ms ({pct:.0f}%)")

    print("\nInference (relative to Softmax-Flash):")
    for key, val in results_inference.items():
        if not np.isnan(val):
            pct = val / baseline_infer * 100
            print(f"  {key}: {val:.2f} ms ({pct:.0f}%)")

    # Generate plots
    output_dir = os.path.dirname(os.path.abspath(__file__))

    plot_results(results_training, results_inference, config,
                 os.path.join(output_dir, 'attention_efficiency_relative.png'))

    plot_detailed_results(results_training, results_inference, config,
                          os.path.join(output_dir, 'attention_efficiency_absolute.png'))

    plot_flash_only(results_training, results_inference, config,
                    os.path.join(output_dir, 'attention_efficiency_flash_only.png'))

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
