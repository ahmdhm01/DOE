from scipy.stats import qmc
import matplotlib.pyplot as plt

# enter number of variables
n = int(input("No. of variables: "))

# enter no. of points
print("Recommended Minimum 10 * n")
p = int(input("No. of points: "))

# lower and upper bounds
lb=input("Enter lower bound seperated by space: ")
ub=input("Enter upper bound seperated by space: ")
lb = list(map(int, lb.split()))
ub = list(map(int, ub.split()))


sample = qmc.LatinHypercube(n, optimization="random-cd")
sample = sample.random(p)

sample = qmc.scale(sample, lb, ub)
print(sample)
if n < 5:
        
    plt.figure()
    
    for i in range(n-1):
        for j in range(i+1,n):
                plt.scatter(sample[:,i],sample[:,j])
     
        
                plt.title("Sample Distribution")
                plt.xlabel(f"Variable {i+1}")
                plt.ylabel(f"Variable {j+1}")
                plt.grid()
    
                plt.show()
else:
    
    for j in range(1,n):
                plt.scatter(sample[:,0],sample[:,j])
     
        
                plt.title("Sample Distribution")
                plt.xlabel("Variable 1")
                plt.ylabel(f"Variable {j+1}")
                plt.grid()
    
                plt.show()
