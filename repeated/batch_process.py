#!/usr/bin/env python3
"""Batch process all text files in text/ folder and save visualizations to imgs/"""

import os
import sys
import glob
import matplotlib.pyplot as plt
from visualization import read_attention_matrix, plot_lower_triangular_heatmap

def process_all_files():
    # Directories
    text_dir = "text/"
    output_dir = "imgs/"
    
    # Get all txt files
    txt_files = glob.glob(os.path.join(text_dir, "*.txt"))
    
    if not txt_files:
        print(f"No text files found in {text_dir}")
        return
    
    print(f"Found {len(txt_files)} text files to process")
    
    # Process each file
    for txt_file in sorted(txt_files):
        filename = os.path.basename(txt_file)
        model_name = filename.replace('.txt', '')
        
        print(f"\nProcessing {filename}...")
        
        try:
            # Read available layers
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # Find all unique layers
            layers = set()
            for line in lines:
                if line.startswith('===== Layer'):
                    # Extract layer number
                    layer_num = int(line.split('Layer ')[1].split(',')[0])
                    layers.add(layer_num)
            
            layers = sorted(layers)
            print(f"  Found layers: {layers}")
            
            # Process each layer
            for layer in layers:
                try:
                    # Read the matrix
                    matrix = read_attention_matrix(txt_file, layer=layer, head='mean')
                    
                    # Create output filename
                    output_file = os.path.join(output_dir, f"{model_name}_layer{layer}.pdf")
                    
                    # Generate visualization (with max 64 tokens for clarity)
                    max_tokens = min(60, matrix.shape[0])
                    fig, ax = plot_lower_triangular_heatmap(
                        matrix, 
                        output_file=output_file,
                        title=None,
                        max_tokens=max_tokens,
                        font_size=36
                    )
                    
                    plt.close(fig)  # Close figure to free memory
                    print(f"  ✓ Layer {layer}: saved to {output_file}")
                    
                except Exception as e:
                    print(f"  ✗ Layer {layer}: Error - {e}")
                    
        except Exception as e:
            print(f"  ✗ Failed to process {filename}: {e}")
    
    print(f"\nProcessing complete! Check the {output_dir} folder for visualizations.")

if __name__ == "__main__":
    process_all_files()