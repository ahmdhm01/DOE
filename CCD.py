'''
Created on Jun 17, 2025

@author: AHMDHM01
'''
import pyDOE3

import matplotlib.pyplot as plt

# Number of factors
num_factors = 3

# Generate the design matrix
design_matrix = pyDOE3.ccdesign(num_factors, center=(4, 4))
'''
# Define factor names
factor_names = ['T', 'P', 'C']
df = pd.DataFrame(design_matrix, columns=factor_names)
'''
# Display the design matrix
print("Central Composite Design Matrix:")
print(design_matrix)
'''
df.plot(kind = 'scatter')
plt.show()
'''

for j in range(1,3):
    plt.scatter(design_matrix[:,0],design_matrix[:,j])
          
    plt.title("Sample Distribution")
    plt.xlabel("Variable 1")
    plt.ylabel(f"Variable {j+1}")
    plt.grid()
    plt.show()
   
