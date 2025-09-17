#!/usr/bin/env python3
"""Test different colormaps to find more eye-catching options"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font
mpl.rcParams['font.family'] = 'Times New Roman'

# Create sample data
data = np.random.rand(10, 10)
data = np.tril(data)  # Lower triangular

# List of eye-catching colormaps
colormaps = [
    'viridis',      # Current - purple to yellow
    'plasma',       # Purple-pink-yellow, more vibrant
    'inferno',      # Black-red-yellow, very dramatic
    'magma',        # Black-purple-white, high contrast
    'cividis',      # Blue-yellow, colorblind friendly
    'turbo',        # Rainbow, very colorful
    'hot',          # Black-red-yellow-white, classic heat
    'jet',          # Blue-cyan-yellow-red, very vibrant but not perceptually uniform
    'coolwarm',     # Blue-white-red, good for diverging data
    'RdYlBu_r',     # Red-yellow-blue reversed, eye-catching
    'Spectral_r',   # Red-yellow-green-blue reversed
    'seismic',      # Blue-white-red, symmetric
]

# Create figure with subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, cmap_name in enumerate(colormaps):
    ax = axes[idx]
    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad('white')
    
    # Apply power stretching like in your visualization
    stretched = np.power(data, 0.5)
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(stretched, dtype=bool), k=1)
    masked_data = np.ma.array(stretched, mask=mask)
    
    im = ax.imshow(masked_data, cmap=cmap, aspect='equal', vmin=0, vmax=1)
    ax.set_title(cmap_name, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('colormap_comparison.png', dpi=150, bbox_inches='tight')
print("Saved colormap comparison to colormap_comparison.png")
plt.show()