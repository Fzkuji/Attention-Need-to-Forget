import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Set font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False

# Global stretching exponent - adjust this value to control stretching (0.2-0.5 works well)
# Smaller values = more stretching (emphasize small values more)
STRETCHING_EXPONENT = 0.5


def read_attention_matrix(filename, layer=None, head='mean'):
    """Read the attention matrix from text file for a specific layer and head
    
    Args:
        filename: Path to the attention data file
        layer: Layer number to extract (e.g., 0, 4, 8, 12, 16, 20). If None, reads first available.
        head: Head identifier to extract (default: 'mean')
    
    Returns:
        Numpy array containing the attention matrix
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find all headers
    headers = []
    header_indices = []
    for i, line in enumerate(lines):
        if line.startswith('====='):
            headers.append(line.strip())
            header_indices.append(i)
    
    # If no layer specified, use the first one
    if layer is None:
        target_header = f"===== Layer 0, Head {head} ====="
    else:
        target_header = f"===== Layer {layer}, Head {head} ====="
    
    # Find the target header
    start_idx = None
    end_idx = len(lines)
    
    for i, (header, idx) in enumerate(zip(headers, header_indices)):
        if header == target_header:
            start_idx = idx + 1
            # Find the next header to determine where this section ends
            if i + 1 < len(headers):
                end_idx = header_indices[i + 1]
            break
    
    if start_idx is None:
        available = "\n".join(headers)
        raise ValueError(f"Header '{target_header}' not found in file.\nAvailable headers:\n{available}")
    
    # Parse the matrix for the selected layer/head
    matrix = []
    for i in range(start_idx, end_idx):
        line = lines[i]
        if line.strip() and not line.startswith('====='):
            row = [float(x) for x in line.split()]
            matrix.append(row)
    
    return np.array(matrix)


def apply_custom_stretching(matrix):
    """Apply smooth non-linear stretching using power function
    This creates a smooth curve that emphasizes lower values
    """
    # Apply power function using the global exponent
    stretched = np.power(matrix, STRETCHING_EXPONENT)
    
    return stretched


def plot_lower_triangular_heatmap(matrix, output_file=None, title=None, max_tokens=None, font_size=48):
    """Create lower triangular heatmap with custom styling
    
    Args:
        matrix: The attention matrix to plot
        output_file: Path to save the output file
        title: Title for the plot
        max_tokens: Maximum number of tokens to display (crops the matrix)
        font_size: Uniform font size for all text elements
    """
    
    # Crop matrix if max_tokens is specified
    if max_tokens is not None and max_tokens < matrix.shape[0]:
        matrix = matrix[:max_tokens, :max_tokens]
    
    # Apply custom stretching to the matrix
    stretched_matrix = apply_custom_stretching(matrix)
    
    # Mask upper triangle (on stretched matrix)
    mask = np.triu(np.ones_like(stretched_matrix, dtype=bool), k=1)
    masked_matrix = np.ma.array(stretched_matrix, mask=mask)

    # Create figure with square aspect ratio
    fig_size = 10
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Use inferno colormap
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad('white')  # Set masked values to white

    # Plot heatmap with stretched values
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='equal', vmin=0, vmax=1)

    # Skip title - not adding even if provided
    # if title:
    #     ax.set_title(title, fontsize=font_size, pad=20)

    # # Customize axes with uniform font size
    # ax.set_xlabel('Position', fontsize=font_size)
    # ax.set_ylabel('Position', fontsize=font_size)

    # Set ticks
    tick_positions = np.arange(0, len(matrix), 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=font_size)
    ax.set_yticklabels(tick_positions, fontsize=font_size)

    # Add colorbar with custom scale labels (thinner width, same height)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.4)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)
    # Remove colorbar borders
    cbar.outline.set_visible(False)
    
    # Set colorbar ticks at evenly spaced positions
    # But the colors still follow the stretching
    even_positions = [0, 0.25, 0.5, 0.75, 1.0]
    
    # Calculate what actual values these positions represent after inverse stretching
    # For power function stretching: y = x^exponent, so x = y^(1/exponent)
    inverse_exponent = 1.0 / STRETCHING_EXPONENT
    
    # Calculate the actual values for each even position
    actual_values = [np.power(pos, inverse_exponent) if pos > 0 else 0 for pos in even_positions]
    value_labels = [f'{val:.2f}' for val in actual_values]
    
    cbar.set_ticks(even_positions)
    cbar.set_ticklabels(value_labels)

    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', format='pdf')
        print(f"Figure saved as {output_file}")
    else:
        plt.show()

    return fig, ax


# Main execution
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize attention matrices from text files')
    parser.add_argument('--file', type=str, default="gpt-oss-20b-repeat.txt",
                       help='Input file containing attention matrices')
    parser.add_argument('--layer', type=int, default=None, 
                       help='Layer number to visualize (e.g., 0, 4, 8, 12, 16, 20)')
    parser.add_argument('--head', type=str, default='mean', 
                       help='Head identifier to visualize (default: mean)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output file path for the visualization')
    parser.add_argument('--max-tokens', type=int, default=None,
                       help='Maximum number of tokens to display (crops the matrix)')
    args = parser.parse_args()
    
    # Construct output filename if not provided
    if args.output is None:
        layer_str = f"layer{args.layer}" if args.layer is not None else "layer0"
        head_str = f"head{args.head}"
        tokens_str = f"_tokens{args.max_tokens}" if args.max_tokens else ""
        args.output = f"attention_heatmap_{layer_str}_{head_str}{tokens_str}.pdf"

    try:
        # Read the attention matrix for the specified layer and head
        matrix = read_attention_matrix(args.file, layer=args.layer, head=args.head)
        
        layer_info = f"Layer {args.layer if args.layer is not None else 0}"
        print(f"Loaded {layer_info}, Head {args.head}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Value range: [{np.min(matrix):.6f}, {np.max(matrix):.6f}]")

        # Create title for the plot
        tokens_info = f" (First {args.max_tokens} tokens)" if args.max_tokens else ""
        title = f"Attention Matrix - {layer_info}, Head {args.head}{tokens_info}"

        # Create the plot (title will be ignored even if passed)
        fig, ax = plot_lower_triangular_heatmap(matrix, output_file=args.output, title=None, max_tokens=args.max_tokens, font_size=24)

    except FileNotFoundError:
        print(f"Error: Could not find file '{args.file}'")
        print("Please make sure the file exists and the path is correct.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")