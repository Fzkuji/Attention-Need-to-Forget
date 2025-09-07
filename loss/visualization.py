import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_loss_comparison():
    ours_data = load_jsonl('ours.jsonl')
    standard_data = load_jsonl('standard.jsonl')
    
    ours_steps = [d['current_steps'] for d in ours_data if d['loss'] is not None]
    ours_loss = [d['loss'] / 4 for d in ours_data if d['loss'] is not None]  # Divide by 4 for accumulation
    
    standard_steps = [d['current_steps'] for d in standard_data if d['loss'] is not None]
    standard_loss = [d['loss'] / 4 for d in standard_data if d['loss'] is not None]  # Divide by 4 for accumulation
    
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(standard_steps, standard_loss,
            label='Standard Attention',
            color='#0072B2',
            linewidth=2.5,
            alpha=0.9)
    ax.plot(ours_steps, ours_loss, 
            label='Ours',
            color='#D55E00',
            linewidth=2.5,
            alpha=0.9)

    ax.set_xlabel('Training Steps', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    
    ax.legend(loc='upper right', 
              frameon=True, 
              fontsize=13,
              framealpha=1.0)
    
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if len(ours_loss) > 0 and len(standard_loss) > 0:
        final_ours = ours_loss[-1]
        final_standard = standard_loss[-1]
        
        ax.text(ours_steps[-1], ours_loss[-1], f' {final_ours:.3f}', 
                fontsize=13, ha='left', va='center')
        ax.text(standard_steps[-1], standard_loss[-1], f' {final_standard:.3f}', 
                fontsize=13, ha='left', va='center')
    
    plt.tight_layout()
    
    plt.savefig('loss_comparison.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating loss comparison visualization...")
    create_loss_comparison()
    print("\nVisualization complete! Generated file: loss_comparison.pdf")