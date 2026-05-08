#evaluation
import numpy as np

def evaluate_method(matrix, name):
    n = matrix.shape[0]
    
    # Extract diagonal (True Positives/Matches)
    diagonal = np.diag(matrix)
    avg_signal = np.mean(diagonal)
    
    # Extract off-diagonal (Potential False Positives)
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]
    avg_noise = np.mean(off_diag)
    max_noise = np.max(off_diag)
    
    # Calculation
    separation_ratio = avg_signal / (avg_noise + 1e-6)
    margin = avg_signal - max_noise # How far is the worst 'fake' from the 'truth'
    
    print(f"--- Results for {name} ---")
    print(f"Avg Signal (Diagonal): {avg_signal:.3f}")
    print(f"Avg Noise (Off-Diag):  {avg_noise:.3f}")
    print(f"Max Noise (Worst Case): {max_noise:.3f}")
    print(f"Separation Ratio:      {separation_ratio:.2f}")
    print(f"Safety Margin:         {margin:.3f}")
    print("-" * 25)
    
    return separation_ratio