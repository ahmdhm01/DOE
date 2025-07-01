import numpy as np
import graphing
import pandas as pd

def generate_uniform_design(N, s, q):
    """
    Generates a Uniform Design (UD) table using modular arithmetic.
    
    Parameters:
    N : int  -> Number of experimental runs
    s : int  -> Number of factors
    q : int  -> Number of levels per factor

    Returns:
    UD_table : numpy array  -> Uniform Design Table
    """
    # Generate a sequence to distribute levels uniformly
    u_seq = np.arange(1, s + 1)  # A simple sequence like [1, 2, 3, ..., s]
    # Create the Uniform Design Table
    UD_table = np.zeros((N, s), dtype=int)
    for i in range(N):
        for j in range(s):
            UD_table[i, j] = ((i * u_seq[j]) % q) + 1  # Modulo mapping
    return UD_table



# Example: Generate a UD(9,3,4)
n, s, q = 50, 3, 15
sample = generate_uniform_design(n, s, q)

# Print the result
print("Uniform Design Table (UD(9,3,4)):")
print(sample)
df = pd.DataFrame(sample)
df.to_csv("Uniform_Design.csv", index=False)


graphing.graphing(df)

