import pyDOE2 
import pandas as pd

# Generate a Plackett-Burman Design for 7 factors
pb_matrix = pyDOE2.fullfact([2,5,7,8])  

# Convert to DataFrame
df = pd.DataFrame(pb_matrix, columns=[f"Factor {i+1}" for i in range(7)])
print(df)
