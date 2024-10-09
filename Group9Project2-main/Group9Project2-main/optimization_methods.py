import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols
from sympy import hessian
import scipy.optimize as so
import pandas as pd
from multiprocessing import Pool

from chebyquad_problem import chebyquad
from chebyquad_problem import gradchebyquad

class optimizationProblem:
    '''
    General Optimization problem class. takes the function and optionally the gradient,
    calculating the gradient automatically if it is not given.
    The function and gradient need to be given as symbolic sympy equations.
    The calc methods take sympy equations and return sympy matrices, the call methods return numpy arrays with the answers.
    '''
    def __init__(self, func, func_gradient=None):
        self.func = func
        
        # get the symbolic function
        if isinstance(func, sp.Expr):
            # initialization of the used variables in the function
            self.variables = sorted(list(self.func.free_symbols), key=lambda x: str(x))

            # initialization of gradient if not given
            if func_gradient is None:
                self.func_gradient = self.calc_gradient()
            else:
                self.func_gradient = func_gradient
                    
            # initialization of the hessian
            self.hessian = self.calc_hessian()
        
        else:
            self.func_gradient = func_gradient
            self.hessian = None
        
    # changing symbolic function to make it callable
    def func_call(self, x):
        if isinstance(self.func, sp.Expr):
            return self.func.subs([self.variables[i], x[i]] for i in range(len(self.variables)))
        else:
            return self.func(x)
        
    # calling the gradient and changing it into numpy
    def gradient_call(self, x):
        if isinstance(self.func, sp.Expr):
            g = self.func_gradient.subs([self.variables[i], x[i]] for i in range(len(self.variables)))
            return np.array(g).astype(np.float64).flatten()
        else:
            if self.func_gradient == None:
                return self.finite_difference_gradient(self.func, x)
            else:
                return self.func_gradient(x)
        
    # changing symbolic hessian function to make it callable
    def hessian_call(self,x):
        if isinstance(self.func, sp.Expr):
            G = self.hessian.subs([self.variables[i], x[i]] for i in range(len(self.variables)))
            return np.array(G).astype(np.float64)
        else:
            return self.finite_difference_hessian(self.func, x)
        
    # calculate gradient by using symbolic differentiation with respect to all variables
    def calc_gradient(self):
        gradient = sp.Matrix([sp.diff(self.func, var) for var in self.variables])
        return gradient
    
    def finite_difference_gradient(self, f, x, h=1e-5):
        n = len(x)
        grad = np.zeros(n)  # Initialize gradient vector

        # finite difference approximation for the gradient 
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            grad[i] = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)
        return grad

    # calculate hessian with symbolic in-build function
    def calc_hessian(self):
        # Automatically calculate the hessian using the matrix object in sympy
        return hessian(self.func, self.variables)

    def finite_difference_hessian(self, f, x, h=1e-5):
        n = len(x)
        G = np.zeros((n, n))  # Initialize Hessian matrix

   	    # finite differences calculation
        for i in range(n):
            for j in range(n):
                e_i = np.zeros(n)
                e_j = np.zeros(n)
                e_i[i] = 1
                e_j[j] = 1
                G[i,j] = (f(x + h * e_i + h * e_j) - f(x + h * e_i - h * e_j) - 
                    f(x - h * e_i + h * e_j) + f(x - h * e_i - h * e_j)) / (4 * h**2)
        # symmetrizing step by taking average of G and G transposed
        return 0.5 * (G + G.T)


# optimization class (parent class)
class optimizationMethod():
    def __init__(self, optimization_problem, initial_x):
        self.optimization_problem = optimization_problem
        self.initial_x = initial_x
        self.path = [] # To store the path of the optimization
        
    def exact_line_search(self, x, p):
        # Introduce alpha, the step size and substitute x + alpha * p into the function
        alpha = symbols('alpha')
        new_x = [x[i] + alpha * p[i] for i in range(len(x))]

        # Create the 1D function for the exact line search
        func_alpha = self.optimization_problem.func.subs(
            [(self.optimization_problem.variables[i], new_x[i]) for i in range(len(x))])

        # Derive the function with respect to alpha and solve for the critical points
        func_alpha_deriv = sp.diff(func_alpha, alpha)
        alpha_opt = sp.solve(func_alpha_deriv, alpha)
        alpha_opt = [a.evalf() for a in alpha_opt if a.is_real and a > 0]

        if len(alpha_opt) > 0:
            return float(min(alpha_opt))  # return the smallest positive alpha
        return 1.0

class NewtonMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem,initial_x)

    # general newton method with updating done via the gradient and the hessian matrix
    def run(self):
        # fitting initial values for x to be the correct data type
        x = np.array(self.initial_x, dtype=float)
        # choosen stopping criterion
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x)) > 1e-5:
            self.path.append(x.copy())
            x += - np.dot(np.linalg.inv(self.optimization_problem.hessian_call(x)), self.optimization_problem.gradient_call(x))
            i += 1
        self.path.append(x.copy())
        return x, i
    
    def run_line_search(self):
        x = np.array(self.initial_x, dtype=float)
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x)) > 1e-5:
            self.path.append(x.copy())
            direction = - np.dot(np.linalg.inv(self.optimization_problem.hessian_call(x)),
                                 self.optimization_problem.gradient_call(x))
            alpha = self.exact_line_search(x, direction)  # Use exact line search for step size
            x += alpha * direction  # Update x with optimal alpha
            self.path.append(x.copy())
            i += 1
        return x, i
    
class Goldstein_Wolfe_method(optimizationMethod):
    # sigma : constant to check for armijo conditon
    # rho : constant to check for wolfe curvature condition
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
    def armijo_conditions(self, x, direction, alpha, sigma = 0.01, rho = 0.9):
        f_current_x = self.optimization_problem.func_call(x)
        gradient_current_x =self.optimization_problem.gradient_call(x)
        slope_current_x = np.dot(gradient_current_x, direction)
        f_alpha_i = self.optimization_problem.func_call(x + alpha * direction)
        return f_alpha_i <= f_current_x + sigma * alpha * slope_current_x
    
    def wolfe_condition(self, x, alpha, direction, sigma = 0.01, rho = 0.9):
        f_current_x = self.optimization_problem.func_call(x)
        gradient_current_x =self.optimization_problem.gradient_call(x)
        slope_current_x = np.dot(gradient_current_x, direction)
        gradient_alpha_i = self.optimization_problem.gradient_call(x + alpha * direction)
        return np.dot(gradient_alpha_i, direction) >= (rho * slope_current_x)
    
    def _compute_alpha(self, x, direction):
        alpha_minus = 1
        while not self.armijo_conditions(x, direction, alpha_minus, sigma = 0.01, rho = 0.9):
            alpha_minus = alpha_minus/2
    
        alpha_plus = alpha_minus
   
        while self.armijo_conditions(x, direction, alpha_plus, sigma = 0.01, rho = 0.9):
            alpha_plus = 2 * alpha_plus
          
        while not (self.wolfe_condition(x, alpha_minus, direction, sigma = 0.01, rho = 0.9)):
            alpha_0 = (alpha_plus + alpha_minus)/2 
            
            if self.armijo_conditions(x, direction, alpha_0, sigma = 0.01, rho = 0.9):
                alpha_minus = alpha_0
                
            else:
                alpha_plus = alpha_0        
        return alpha_minus
        
    def direction(self, x, gradient_current_x):
        Hessian_matrix = (self.optimization_problem.hessian_call(x))
        direction = - np.dot(np.linalg.inv(Hessian_matrix), gradient_current_x)
        return direction
    
    def run(self, max_iterations = 100):
        i = 0
        
        x = np.array(self.initial_x, dtype=float)
        
        while np.linalg.norm(self.optimization_problem.gradient_call(x)) > 1e-5 and (i < max_iterations):
            self.path.append(x.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x)
            direction = self.direction(x, gradient_current_x)
            alpha = self._compute_alpha(x, direction)
            
            x += alpha * direction
            
            i += 1
        self.path.append(x.copy())
        return x, i


class goodBroydenMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        self.wolfe_method = Goldstein_Wolfe_method(optimization_problem, initial_x)
        self.H_k = np.eye(len(self.initial_x))
        
        
    def direction(self, x, grad_k):
        direction = -np.dot(self.H_k, grad_k)
        return direction

    # update the hessian inverse so it doesn't need to be calculated each step 
    def update_hessian(self, x_k, grad_k):
        
        if self.x_k_prev is not None and self.gradient_prev_x is not None:
            delta_k = x_k - self.x_k_prev  
            gamma_k = grad_k - self.gradient_prev_x 
            denominator = np.dot(np.dot(delta_k.T, self.H_k), gamma_k)
            
            if denominator != 0:
                self.H_k = self.H_k + np.dot((delta_k - np.dot(self.H_k, gamma_k)), np.dot(delta_k.T, self.H_k)) / denominator
                
    def run(self, max_iterations= 100):
        self.x_k_prev = None
        self.gradient_prev_x = None
        x_k = np.array(self.initial_x, dtype=float)
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x_k)) > 1e-5 and (i < max_iterations):
            self.path.append(x_k.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x_k)
            direction = self.direction(x_k, gradient_current_x)
            alpha = self.wolfe_method._compute_alpha(x_k, direction)
            self.update_hessian(x_k, gradient_current_x)
            self.x_k_prev = x_k.copy()
            self.gradient_prev_x = gradient_current_x
            x_k += alpha * direction
            
            i +=1
        self.path.append(x_k.copy())
        return x_k, i
    
    
class badBroydenMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        self.H_k = np.eye(len(self.initial_x)) # Initialize an Identity matrix
        self.wolfe_method = Goldstein_Wolfe_method(optimization_problem, initial_x)
    
    def direction(self, x, grad_k):
        direction = -np.dot(self.H_k, grad_k)
        return direction

    def update_hessian(self, x_k, grad_k):
        if self.x_k_prev is not None and self.gradient_prev_x is not None:
            delta_k = x_k - self.x_k_prev  
            gamma_k = grad_k - self.gradient_prev_x 
            denominator = np.dot(gamma_k.T, gamma_k)
            
            if denominator != 0:
                self.H_k = self.H_k + np.dot((delta_k - np.dot(self.H_k, gamma_k)), gamma_k.T) / denominator
                                             
    def run(self, max_iterations = 100):
        self.x_k_prev = None
        self.gradient_prev_x = None
        x_k = np.array(self.initial_x, dtype=float)
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x_k)) > 1e-5 and (i < max_iterations):
            self.path.append(x_k.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x_k)
            direction = self.direction(x_k, gradient_current_x)
            alpha = self.wolfe_method._compute_alpha(x_k, direction)
            self.update_hessian(x_k, gradient_current_x)
            self.x_k_prev = x_k.copy()
            self.gradient_prev_x = gradient_current_x
            x_k += alpha * direction
            
            i +=1
        self.path.append(x_k.copy())
        return x_k, i


class symmetricBroydenMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        self.H_k = np.eye(len(self.initial_x)) # Initialize an Identity matrix
        self.wolfe_method = Goldstein_Wolfe_method(optimization_problem, initial_x)
        
    def direction(self, x, grad_k):
        direction = -np.dot(self.H_k, grad_k)
        return direction

    def update_hessian(self, x_k, grad_k):
        if self.x_k_prev is not None and self.gradient_prev_x is not None:
            delta_k = x_k - self.x_k_prev  
            gamma_k = grad_k - self.gradient_prev_x 
            u_k = delta_k - np.dot(self.H_k, gamma_k) 
        
            if u_k.all() != 0 and gamma_k.all() != 0:
                a_k = 1 / np.dot(u_k.T, gamma_k)
                self.H_k = self.H_k + a_k * np.dot(u_k, u_k.T)
                
    def run(self, max_iterations = 100):
        self.x_k_prev = None
        self.gradient_prev_x = None
        x_k = np.array(self.initial_x, dtype=float)
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x_k)) > 1e-5 and (i < max_iterations):
            self.path.append(x_k.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x_k)
            direction = self.direction(x_k, gradient_current_x)
            alpha = self.wolfe_method._compute_alpha(x_k, direction)
            self.update_hessian(x_k, gradient_current_x)
            self.x_k_prev = x_k.copy()
            self.gradient_prev_x = gradient_current_x
            x_k += alpha * direction
            
            i +=1
        self.path.append(x_k.copy())
        return x_k, i

class DFP(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        self.H_k = np.eye(len(self.initial_x)) # Initialize an Identity matrix
        self.wolfe_method = Goldstein_Wolfe_method(optimization_problem, initial_x)
        
    def direction(self, x, gradient_current_x):
        direction = -np.dot(self.H_k, gradient_current_x)
        return direction
    
    def _update_Hessian_matrix(self, x_k, gradient_current_x):
        if self.x_k_prev is not None and self.gradient_prev_x is not None:
        
            delta_k = (x_k - self.x_k_prev)
            gamma_k = (gradient_current_x - self.gradient_prev_x)
            denominator_a = np.dot(delta_k.T, gamma_k)
            z_k = np.dot(self.H_k, gamma_k)
            y_k = np.dot(gamma_k.T, self.H_k)
            denominator_b = np.dot(gamma_k.T, z_k)
            if denominator_a != 0 and denominator_b !=0:
                a = np.dot(delta_k, delta_k.T) / denominator_a
                b = np.dot(z_k, y_k) / denominator_b
                self.H_k += a - b 
             
    def run(self, max_iterations= 100):
        self.x_k_prev = None
        self.gradient_prev_x = None
        x_k = np.array(self.initial_x, dtype=float)
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x_k)) > 1e-5 and (i < max_iterations):
            self.path.append(x_k.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x_k)
            direction = self.direction(x_k, gradient_current_x)
            alpha = self.wolfe_method._compute_alpha(x_k, direction)
            self._update_Hessian_matrix(x_k, gradient_current_x)
            self.x_k_prev = x_k.copy()
            self.gradient_prev_x = gradient_current_x
            x_k += alpha * direction
            
            i +=1
        self.path.append(x_k.copy())
        return x_k, i
                
class BFGS(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        self.H_k = np.eye(len(self.initial_x)) # Initialize an Identity matrix
        self.wolfe_method = Goldstein_Wolfe_method(optimization_problem, initial_x)
        self.hessian_error = []
        
    def direction(self, x, gradient_current_x):
        direction = -np.dot(self.H_k, gradient_current_x)
        return direction
    
    def _update_hessian(self, x_k, gradient_current_x):
        if self.x_k_prev is not None and self.gradient_prev_x is not None:
            delta_k = (x_k - self.x_k_prev)
            gamma_k = (gradient_current_x - self.gradient_prev_x)
            
            denominator = np.dot(delta_k.T, gamma_k)
            
            a_k = np.dot(gamma_k.T, np.dot(self.H_k, gamma_k))
            b_k = np.dot(delta_k, delta_k.T)
            c_k = np.dot(delta_k, np.dot(gamma_k.T, self.H_k)) + np.dot(np.dot(self.H_k, gamma_k), delta_k.T)
            
            if denominator != 0:
                self.H_k = self.H_k + (1 + a_k / denominator) * b_k / denominator - c_k / denominator
                
    def _compute_hessian_error(self, x_k):
        # Compute the actual Hessian at x_k using finite differences 
        
        actual_hessian = self.optimization_problem.hessian_call(x_k)
        hessian_inv = np.linalg.inv(actual_hessian)
        
        # Compute the Frobenius norm of the difference between true inverse Hessian and H_k
        error = np.linalg.norm(hessian_inv - self.H_k, ord='fro')
        self.hessian_error.append(error)
        return self.hessian_error
            
    def run(self, max_iterations = 100):
        self.x_k_prev = None
        self.gradient_prev_x = None
        x_k = np.array(self.initial_x, dtype=float)
        i = 0
        while np.linalg.norm(self.optimization_problem.gradient_call(x_k)) > 1e-5 and (i < max_iterations):
            self.path.append(x_k.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x_k)
            direction = self.direction(x_k, gradient_current_x)
            alpha = self.wolfe_method._compute_alpha(x_k, direction)
            self._update_hessian(x_k, gradient_current_x)
            self._compute_hessian_error(x_k)
            self.x_k_prev = x_k.copy()
            self.gradient_prev_x = gradient_current_x
            x_k += alpha * direction
            
            i +=1
        self.path.append(x_k.copy())
        return x_k, i
            
def save_rosenbrock_plot(path, filename='rosenbrock_optimization_plot.png', cmap='viridis'):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    # Plot the optimization path
    if path is not None:
        path = np.array(path)  # Ensure path is a NumPy array
        ax.plot(path[:, 0], path[:, 1], 'k-', linewidth=2, label='Optimization Path')
        ax.plot(path[:, 0], path[:, 1], 'ro', markersize=5, label='Points')  # Points in red
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Optimization Plot')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def save_table_as_image(df, filename):
    plt.figure()
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    ax.set_frame_on(False)

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold')  # Set bold font for headers
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    

# %% Rosenbrock function implementation
def rosenbrock_function():
    x1, x2 = symbols('x1, x2')
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2
# global minimum at [1, 1]

def rosenbrock_function_gradient():
    x1, x2 = symbols('x1, x2')
    f1 = -400 *x1 * (x2 - x1**2) - 2 * (1-x1)
    f2 = 200 * (x2 - x1**2)
    return sp.Matrix([f1, f2])

def rosenbrock_steps(params):
    x, y = params
    rosenbrock_problem = optimizationProblem(rosenbrock_function())
    return NewtonMethod(rosenbrock_problem, [x, y]).run()[1]

# Rosenbrock plotting function
def plot_rosenbrock_steps(x_range, y_range, dimension):

    # Creating the array containing every combination required
    x_space = np.linspace(x_range[0], x_range[1], dimension)
    y_space = np.linspace(y_range[0], y_range[1], dimension)
    parameter_space = np.array(np.meshgrid(x_space, y_space)).T.reshape(-1, 2)

    with Pool() as pool:
        # Multithreading, the rosenbrock steps function is run simultaneously over several threads, with different parameters from the parameter space
        # For some reason the rosenbrock_steps function has to be outside the current function for multithreading to work
        results = np.array(pool.map(rosenbrock_steps, parameter_space))

    results = results.reshape(dimension, dimension) # Reshape to fit the dimensions of an image

    plt.figure(figsize=(8, 8), dpi=500)
    c = plt.imshow(results, cmap='Blues')
    plt.contour(results, [2, 3, 4, 5, 7], colors='black', linewidths=0.7).clabel(inline=True, fontsize=12)

    plt.title("Iterations to find minimum of Rosenbrock's function")
    plt.xlabel('x')
    plt.ylabel('y')

    # Define step size for labels (nth tick)
    n = 50
    plt.xticks(np.arange(0, len(x_space), n), np.round(x_space[::n], 2))
    plt.yticks(np.arange(0, len(y_space), n), np.round(y_space[::n], 2))
    plt.colorbar(c).set_label('Iterations to reach stopping criterion')
    plt.savefig('rosenbrock_steps.png')


# %% Testing function with known zero root
def test_function():
    x1, x2 = symbols('x1, x2')
    return (x1 + 1)**2 + x2**2
# should result in zero root: [-1, 0]

# %%
if __name__ == '__main__':
    # test of the code
    test_optimization = optimizationProblem(test_function())
    newton = NewtonMethod(test_optimization,[1.0, 2.0])
    xmin, it = newton.run_line_search()
    print(f'the zero root for the test function: {xmin}')
    
    # test of the code
    test_optimization = optimizationProblem(test_function())
    bfgs = BFGS(test_optimization,[1.0, 2.0])
    xmin, it = bfgs.run()
    print(f'the zero root for BFGS: {xmin} \n number of iterations: {it}')
    
    # rosenbrock optimization problem
    result = []
    
    optimization_rosenbrock = optimizationProblem(rosenbrock_function())
    rosenbrock = NewtonMethod(optimization_rosenbrock, [0.1, 0.5])
    xmin, it = rosenbrock.run_line_search()
    print(f'the zero root for newtonMethod: {xmin}')
    
    # trying to implement with already input gradient
    optimization_rosenbrock = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    rosenbrock = NewtonMethod(optimization_rosenbrock, [-0.4, 0.8])
    xmin, it = rosenbrock.run()
    print(f'the zero root for newtonMethod: {xmin}')
    save_rosenbrock_plot(rosenbrock.path, filename='newton_rosenbrock_gradient_plot.png')
    result.append(['Newton Method', xmin, it])
    
    #test_optimization for te rosenborkfunction
    Wolfe_rosenbrock_function = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    Wolfe = Goldstein_Wolfe_method(Wolfe_rosenbrock_function, [-0.4, 0.8])
    xmin, it = Wolfe.run()
    print(f'the zero root for goldstein_wolfe_method: {xmin}')
    save_rosenbrock_plot(Wolfe.path, filename='goldstein_wolfe_rosenbrock_plot.png')
    result.append(['Goldstein-Wolfe Method', xmin, it])
    
    goodBroydenMethod_rsoen = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    GB = goodBroydenMethod(goodBroydenMethod_rsoen, [0.0, 0.0 ])
    xmin, it = GB.run()
    print(f'the zero root for goodBroyden Method: {xmin}')
    save_rosenbrock_plot(GB.path, filename='goodBroyden_rosenbrock_plot.png')
    result.append(['Good Broyden', xmin, it])
      
    badBroydenMethod_rsoen = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    BB = badBroydenMethod(badBroydenMethod_rsoen, [0.5, 1.0 ])
    xmin, it = BB.run()
    print(f'the zero root for badBroyden Method: {xmin}')
    result.append(['Bad Broyden', xmin, it])

    symmetricBroydenMethod_rsoen = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    SB = symmetricBroydenMethod(symmetricBroydenMethod_rsoen, [0, 0 ])
    xmin, it = SB.run()
    print(f'the zero root for symmetricBroyden Method: {xmin}')
    result.append(['Symmetric Broyden', xmin, it])
    
    optimization_rosenbrock_for_DFP = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    DFP_op = DFP(optimization_rosenbrock_for_DFP, [0.0, 0.0])
    xmin, it = DFP_op.run()
    print(f'the zero root for DFP: {xmin}')
    save_rosenbrock_plot(DFP_op.path, filename='DFP_plot.png')
    result.append(['DFP', xmin, it])
    
    optimization_rosenbrock_for_BFGS = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    BFGS_op = BFGS(optimization_rosenbrock_for_BFGS, [0.2, 0.8])
    xmin, it = BFGS_op.run()
    print(f'the zero root for BFGS: {xmin} \n number of iterations: {it}')
    result.append(['BFGS', xmin, it])
    
    result.append(['Theoretical Value', [1.00,1.00]])
    df_result = pd.DataFrame(result, columns=['Method', 'Function Value', 'Iterations'])
    save_table_as_image(df_result, f'optimization_results_rosenbrock.png') 
    
    # Task 12 to print the BFGS approximation
    plt.plot(BFGS_op.hessian_error)
    plt.xlabel('Iterations')
    plt.ylabel('Error in Hessian approximation (Frobenius norm)')
    plt.title('BFGS Hessian Approximation Error')
    plt.savefig('hessian_error.png')

    
    # %% rosenbrock plotting
    plot_rosenbrock_steps([0.5, 1.5], [-1, 3], 50)
    # better results were obtained using dimension 500, but for the sake of this code not running eternally, it is exemplary shown with 50 
   
    # %% 
    def test_function_np(x):
        return (x[0] + 1)**2 + x[1]**2
    
    def test_function_np_grad(x):
        return np.array([2* (x[0] + 1), 2*x[1]])
    
    testNumpy = optimizationProblem(test_function_np)
    testNumpy_optimization = NewtonMethod(testNumpy, [0.1, 0.5])
    print(f'the zero root for the numpy test function is: {testNumpy_optimization.run()}')
    
    testNumpy = optimizationProblem(test_function_np, test_function_np_grad)   
    testNumpy_optimization = NewtonMethod(testNumpy, [0.1, 0.5])
    print(f'the zero root for the numpy test function is: {testNumpy_optimization.run()}')
    
    
    def chebyquad_optimization(n):
        results = []
        
        x = np.ones(n)
        chebyquad_problem = optimizationProblem(chebyquad, gradchebyquad)
        
        optimization_newton = NewtonMethod(chebyquad_problem, x)
        xmin, it = optimization_newton.run()
        results.append(['Newton Method', chebyquad(xmin), it])
        
        optimization_broyden = goodBroydenMethod(chebyquad_problem, x)
        xmin, it = optimization_broyden.run()
        results.append(['Good Broyden', chebyquad(xmin), it])
        
        optimization_broyden = badBroydenMethod(chebyquad_problem, x)
        xmin, it = optimization_broyden.run()
        results.append(['Bad Broyden', chebyquad(xmin), it])
        
        optimization_dfp = DFP(chebyquad_problem, x)
        xmin, it = optimization_dfp.run()
        results.append(['DFP', chebyquad(xmin), it])
        
        optimization_bfgs = BFGS(chebyquad_problem, x)
        xmin, it = optimization_bfgs.run()
        results.append(['BFGS', chebyquad(xmin), it])
        
        output = so.fmin_bfgs(chebyquad,np.ones(n),gradchebyquad, full_output=True, retall=True)
        xmin = output[0]
        fopt = output[1]
        it = len(output[7])
        results.append(['SciPy BFGS', fopt, it])
        
        df_results = pd.DataFrame(results, columns=['Method', 'Function Value', 'Iterations'])
        save_table_as_image(df_results, f'optimization_results_n{n}.png')
    
    
    chebyquad_optimization(4)
    chebyquad_optimization(8)
    chebyquad_optimization(11)

    
