

import numpy as np
from geneticalgorithm import geneticalgorithm as ga

#from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class Optimizer:
    def __init__(self):
        self.num_vars = None
        self.degree = None
        self.x=None
        self.y=None
        self.degree=None
        self.model=None
        self.poly = None 
        self.No_indepd_variables = 1
        self.No_depd_variables = 1      
    # Load the data

    def load_data(self):
        data = np.loadtxt("C:\\Users\\ahmdhm01\\eclipse-workspace\\DOE\\Uniform_Design.csv", delimiter=',')
        self.x = data[:, :self.No_indepd_variables]
        self.y = data[:, self.No_indepd_variables:]
        #self.y=y_loaded.ravel()
        self.No_depd_variables = self.y.shape[1]
        return self.x, self.y
    
    def fitting(self, Type):
               
        if(Type == "Polynomial_Regression"):
            self.degree = int(input("degree of the desired curve: "))
            self.poly = PolynomialFeatures(self.degree, include_bias=False)
            poly_x = self.poly.fit_transform(self.x)
            self.model = LinearRegression()
            self.model.fit(poly_x, self.y)
        elif(Type == "Decision_Tree_Regression"):
            from sklearn.tree._classes import DecisionTreeRegressor


            self.model = DecisionTreeRegressor()
            self.model.fit(self.x, self.y)

        elif(Type == "Random_Forest_Regression"):
            from sklearn.ensemble._forest import RandomForestRegressor

            self.model = RandomForestRegressor()
            self.model.fit(self.x, self.y)
        
        return self.poly, self.model
    
    def fitness_function(self, x_input):
        x_input = np.array(x_input).reshape(1, -1)  # Reshape input
        x_poly = self.poly.transform(x_input)           # Polynomial transform
        prediction = self.model.predict(x_poly)
        return prediction[0][0] if prediction.ndim > 1 else prediction[0]
     

    def genetic(self, varbound):
        
        algorithm_param = {'max_num_iteration': None,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
        
        genetic_algor= ga(function=self.fitness_function,
             dimension=self.No_indepd_variables,
             variable_type='real',
             variable_boundaries=varbound,
             algorithm_parameters=algorithm_param)
        genetic_algor.run()
    
    
    def graphing(self, Type):
        import matplotlib.pyplot as plt
        if self.No_indepd_variables == 1:
            print(0)

            xp = np.linspace(min(self.x), max(self.x), 100).reshape(-1, 1)
            
            if Type == "Polynomial_Regression":
                xp_poly = self.poly.transform(xp)
                yp = self.model.predict(xp_poly)
            else:
                yp = self.model.predict(xp)

            plt.plot(self.x, self.y, 'o', label='Data')
            plt.plot(xp, yp, '-', label='Model')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("1D Polynomial Fit")
            plt.legend()
            plt.grid()
            plt.show()
            
            
        else:
            x1 = self.x[:, 0]
            for i in range(1,self.No_indepd_variables):
                x2 = self.x[:, i]
                x1_grid, x2_grid = np.meshgrid(
                    np.linspace(min(x1), max(x1), 100),
                    np.linspace(min(x2), max(x2), 100)
                    )
                print(x1_grid)
                x_base = np.mean(self.x, axis=0)
                x1_flat = x1_grid.ravel()
                x2_flat = x2_grid.ravel()
                X_grid_input = np.tile(x_base, (len(x1_flat), 1))
                X_grid_input[:, 0] = x1_flat
                X_grid_input[:, i] = x2_flat
                if Type == "Polynomial_Regression":
                    xp_poly = self.poly.transform(X_grid_input)
                    yp = self.model.predict(xp_poly)

                    #yp.reshape(yp.ravel(),(x1_grid.shape,self.No_depd_variables))
                else:
                    yp = self.model.predict(X_grid_input)
                
                #for j in range(No_depd_variables):
                for j in range(self.No_depd_variables):
                    z = yp[:,j].reshape(x1_grid.shape)
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(x1_grid, x2_grid, z, cmap='viridis', alpha=0.8)
                    ax.scatter(self.x[:, 0], self.x[:, i], self.y[:,j], color='r', label='Data Points')
                    ax.set_xlabel("Input 1")
                    ax.set_ylabel(f"Input {i+1}")
                    ax.set_zlabel(f"Output{j+1}")
                    plt.title(f"Surface Plot: x1 vs x{i+1}")
                    plt.legend()
                    plt.tight_layout()
                    plt.show() 
            
        
           
if __name__ == "__main__":
    
    print("=== Optimizing ===")
    
    Designer = Optimizer()
    
    #Filename = input("Please enter file name: ")
    # User input: number of independent variables
    Designer.No_indepd_variables = int(input("Please enter number of independent variables: "))
    
    print("Choose a fitting model:")
    print("\tPolynomial_Regression")
    print("\tDecision_Tree_Regression")
    print("\tRandom_Forest_Regression")
    Type = input("Please enter fitting type, copy and paste to ensure right writing: ")
    
    


    Designer.load_data()

    Designer.fitting(Type)

    Designer.graphing(Type)


    '''
    varbound = np.zeros((Designer.No_indepd_variables,2))
    for i in range(0, Designer.No_indepd_variables): 
        print("Please, minimum value for variable", i+1)
        varbound[i,0] = float(input(""))
        print("Please, maximum value for variable", i+1)
        varbound[i,1] = float(input(""))
        
    Designer.genetic(varbound)'''