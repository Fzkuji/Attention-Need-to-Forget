import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# Your data
data = np.array([
    [73.26, 0.00, 0.00, 0.00, 0.01, 0.00, 0.01, 0.01, 0.00, 0.01],
    [0.00, 69.15, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.03, 0.00, 0.07, 0.00, 63.23, 0.00, 0.01, 0.00, 0.01, 0.01],
    [0.21, 0.00, 0.10, 0.00, 0.00, 0.01, 0.03, 0.01, 60.45, 0.01],
    [7.36, 0.16, 0.14, 0.00, 0.00, 0.00, 0.04, 0.00, 0.01, 0.00],
    [9.28, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00],
    [18.43, 0.50, 0.16, 0.00, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00]
])

fig, ax = plt.subplots(figsize=(8, 5))
h = sns.heatmap(data,
                annot=True,
                fmt='.1f',
                cmap='RdYlBu_r',
                # cbar_kws={'label': 'Sink Percentage (%)'},
                xticklabels=[f'{i}' for i in range(1, 11)],
                yticklabels=['[MASK] @ 1', '[MASK] @ 2', '[MASK] @ 5',
                            '[MASK] @ 9', '[MASK] @ 17', '[MASK] @ 33',
                            '[MASK] @ 65'],
                vmin=0,
                vmax=75,
                annot_kws={'fontsize': 12})

# Set colorbar label font size
cbar = h.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Sink Percentage (%)', fontsize=18, labelpad=10)

ax.set_xlabel('Token Position', fontsize=18)
ax.set_ylabel('Fixed Token Position', fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# ax.set_title('Attention Sink Migration with Fixed Token Placement')

# # Add a box to highlight the sink positions
# for i in range(4):
#     ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
#                               edgecolor='red', lw=2))

plt.tight_layout()
plt.savefig('attention_sink_heatmap.pdf', dpi=300)