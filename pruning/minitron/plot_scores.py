import matplotlib.pyplot as plt 
def plot_scores(data):
    assert "base_line" in data and "scores" in data, "Data must contain 'base_line' and 'scores' keys"
    
    plt.plot(data["scores"], marker='o', label='Pruned Model Scores')
    plt.axhline(y=data["base_line"], color='r', linestyle='--', label='Baseline Score')
    plt.legend()
    
    plt.show()
    
if __name__ == '__main__':
    data = {
        "scores": [0.7, 0.4, 0.3, 0.8],
        "baseline": 1
    }

    plot_scores(data)