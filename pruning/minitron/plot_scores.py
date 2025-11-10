import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
import json    
import argparse
import matplotlib.pyplot as plt 

def plot_scores(data, save_path=None):
    assert "base_line" in data and "scores" in data, "Data must contain 'base_line' and 'scores' keys"
    
    plt.plot(data["scores"], marker='o', label='Pruned Model Scores')
    plt.axhline(y=data["base_line"], color='r', linestyle='--', label='Baseline Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.log_path, 'r') as f:
        data = json.load(f)
    # data = {
    #     "scores": [0.7, 0.4, 0.3, 0.8],
    #     "baseline": 1
    # }
    save_dir = args.log_path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, 
        os.path.basename(args.log_path).replace('.json', '.png')
    )

    plot_scores(data, save_path)
    print(f"Plot saved to {save_path}")