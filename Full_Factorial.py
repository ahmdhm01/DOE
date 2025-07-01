import itertools  # For generating combinations
import pandas as pd  # For creating tables

def generate_full_factorial(num_factors, num_levels):
    """
    Generates a Full Factorial Design table.

    Parameters:
    num_factors : int -> Number of factors.
    num_levels : int -> Levels per factor (2 for [-1, 1] or 3 for [-1, 0, 1]).

    Returns:
    DataFrame -> Pandas DataFrame containing the full factorial design.
    """
    # Step 1: Define the levels based on the number of levels chosen
    if num_levels == 2:
        levels = [-1, 1]  # Two levels: Low (-1), High (1)
    elif num_levels == 3:
        levels = [-1, 0, 1]  # Three levels: Low (-1), Medium (0), High (1)
    else:
        raise ValueError("Only 2-level (-1,1) and 3-level (-1,0,1) designs are supported.")

    # Step 2: Generate all possible combinations using itertools.product
    full_factorial = list(itertools.product(levels, repeat=num_factors))

    # Step 3: Create column names dynamically (Factor 1, Factor 2, ...)
    column_names = [f"Factor {i+1}" for i in range(num_factors)]

    # Step 4: Convert to a Pandas DataFrame
    df = pd.DataFrame(full_factorial, columns=column_names)
    return df  # Return the factorial design table

# Example Usage: Modify these values
num_factors = 3  # Number of factors
num_levels = 2  # Choose 2 for [-1,1] or 3 for [-1,0,1]

# Generate Full Factorial Design
factorial_table = generate_full_factorial(num_factors, num_levels)

# Print the result
print("Full Factorial Design Table:")
print(factorial_table)

# Save to CSV (Optional)

factorial_table.to_csv("full_factorial_design.csv", index=False)
