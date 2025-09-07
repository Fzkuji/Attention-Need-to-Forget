#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib parameters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def standard_dog(x, A1=1.0, s1=5.0, c1=3, A2=0.6, s2=25.0, c2=1, offset=0):
    """Standard Difference of Gaussians (DoG) function
    
    Key insight: When c2=1 (far Gaussian centered at start), 
    the far Gaussian has maximum value at x=1, creating negative start.
    """
    near = A1 * np.exp(-((x-c1)/s1)**2)  # Positive contribution (peak)
    far = A2 * np.exp(-((x-c2)/s2)**2)   # Negative contribution (suppression)
    return near - far + offset

# Create data
x = np.linspace(1, 150, 1000)

# Test different parameter combinations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Standard DoG with Different Parameters', fontsize=16, fontweight='bold')

# Different parameter sets to explore - focusing on wider near Gaussian
param_sets = [
    {'A1': 1.0, 's1': 5.0, 'c1': 3, 'A2': 0.6, 's2': 25.0, 'c2': 1, 'offset': 0,
     'title': 'Original (s1=5)'},
    {'A1': 0.01, 's1': 10.0, 'c1': 5, 'A2': 0.01, 's2': 25.0, 'c2': 10, 'offset': 0,
     'title': 'Wider Near (s1=10)'},
    {'A1': 1.0, 's1': 15.0, 'c1': 3, 'A2': 0.6, 's2': 25.0, 'c2': 1, 'offset': 0,
     'title': 'Much Wider Near (s1=15)'},
    {'A1': 1.0, 's1': 20.0, 'c1': 3, 'A2': 0.6, 's2': 25.0, 'c2': 1, 'offset': 0,
     'title': 'Very Wide Near (s1=20)'},
    {'A1': 1.0, 's1': 30.0, 'c1': 3, 'A2': 0.6, 's2': 25.0, 'c2': 1, 'offset': 0,
     'title': 'Extra Wide Near (s1=30)'},
    {'A1': 1.0, 's1': 40.0, 'c1': 3, 'A2': 0.6, 's2': 25.0, 'c2': 1, 'offset': 0,
     'title': 'Super Wide Near (s1=40)'},
]

for idx, params in enumerate(param_sets):
    ax = axes[idx // 3, idx % 3]
    
    # Extract title separately
    title = params.pop('title')
    
    # Calculate DoG
    y = standard_dog(x, **params)
    params['title'] = title  # Add title back for later use
    
    # Plot
    ax.plot(x, y, 'b-', linewidth=2.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(x, y, 0, where=(y > 0), alpha=0.3, color='green')
    ax.fill_between(x, y, 0, where=(y <= 0), alpha=0.3, color='red')
    
    # Mark key points (remove title from params for function call)
    params_no_title = {k: v for k, v in params.items() if k != 'title'}
    ax.plot(1, standard_dog(np.array([1]), **params_no_title)[0], 'ro', markersize=8)
    peak_idx = np.argmax(y[:50])
    ax.plot(x[peak_idx], y[peak_idx], 'go', markersize=8)
    
    # Labels and title
    ax.set_title(params['title'], fontsize=11)
    ax.set_xlabel('Relative Position')
    ax.set_ylabel('Bias Value')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([-0.8, 0.8])
    
    # Add parameter info
    info_text = f"Start: {standard_dog(np.array([1]), **params_no_title)[0]:.2f}\n"
    info_text += f"Peak: {y[peak_idx]:.2f} at x={x[peak_idx]:.0f}"
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, 
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('standard_dog_variations.pdf', dpi=300, bbox_inches='tight')
plt.savefig('standard_dog_variations.png', dpi=150, bbox_inches='tight')
# plt.show()  # Comment out for non-interactive execution

# Print the formula and parameter meanings
print("\n" + "="*60)
print("Standard DoG Formula:")
print("="*60)
print("\nDoG(x) = A1 * exp(-((x-c1)/s1)²) - A2 * exp(-((x-c2)/s2)²) + offset\n")
print("Parameters:")
print("  A1: Amplitude of positive Gaussian (peak height)")
print("  c1: Center of positive Gaussian (peak position)")
print("  s1: Width of positive Gaussian (peak sharpness)")
print("  A2: Amplitude of negative Gaussian (suppression strength)")
print("  c2: Center of negative Gaussian (suppression center)")
print("  s2: Width of negative Gaussian (suppression range)")
print("  offset: Vertical shift of entire curve")
print("\nTotal: 7 parameters (6 for DoG + 1 offset)")

# Example: Show how to use for position encoding
print("\n" + "="*60)
print("Example Usage for Position Encoding:")
print("="*60)
print("""
# Initialize position bias with DoG
def initialize_position_bias(max_len=1024):
    positions = np.arange(1, max_len + 1)
    
    # Parameters learned from your data
    bias = standard_dog(positions, 
                       A1=1.2, s1=4.0, c1=6,
                       A2=0.8, s2=15.0, c2=1, 
                       offset=0)  # Let model learn offset
    
    return bias

# Usage in model
position_bias = initialize_position_bias()
attention_weights = attention_weights + position_bias
""")