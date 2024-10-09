import numpy as np
import matplotlib.pyplot as plt
import sympy
import sympy as sp
from sympy import symbols
from sympy import hessian

class optimizationProblem:
    '''
    General Optimization problem class. takes the function and optionally the gradient,
    calculating the gradient automatically if it is not given.
    The function and gradient need to be given as symbolic sympy equations.
    The calc methods take sympy equations and return sympy matrices, the call methods return numpy arrays with the answers.
    '''
    def __init__(self, func, func_gradient=None):
        # get the symbolic function
        self.func = func

        assert isinstance(self.func, sympy.Expr), 'The function must be sympy symbolic equation'

        # initialization of the used variables in the function
        self.variables = sorted(list(self.func.free_symbols), key=lambda x: str(x))

        # initialization of gradient if not given
        if func_gradient is None:
            self.func_gradient = self.calc_gradient()
        else:
            self.func_gradient = func_gradient

        assert isinstance(self.func_gradient, sp.Matrix), 'The gradient function needs to be a sympy matrix'

        # initialization of the hessian
        self.hessian = self.calc_hessian()


    # changing symbolic function to make it callable
    def func_call(self, x):
        return self.func.subs([self.variables[i], x[i]] for i in range(len(self.variables)))

    # calling the gradient and changing it into numpy
    def gradient_call(self, x):
        g = self.func_gradient.subs([self.variables[i], x[i]] for i in range(len(self.variables)))
        return np.array(g).astype(np.float64).flatten()

    # changing symbolic hessian function to make it callable
    def hessian_call(self,x):
        G = self.hessian.subs([self.variables[i], x[i]] for i in range(len(self.variables)))
        return np.array(G).astype(np.float64)

    # calculate gradient by using symbolic differentiation with respect to all variables
    def calc_gradient(self):
        gradient = sp.Matrix([sp.diff(self.func, var) for var in self.variables])
        return gradient

    # calculate hessian with symbolic in-build function
    def calc_hessian(self):
        # Automatically calculate the hessian using the matrix object in sympy
        return hessian(self.func, self.variables)

# optimization class (parent class)
class optimizationMethod():
    def __init__(self, optimization_problem, initial_x):
        self.optimization_problem = optimization_problem
        self.initial_x = initial_x
        self.path = [] # To store the path of the optimization
        
    def _compute_alpha(self, x, direction, sigma = 0.01, rho = 0.9, max_iterations = 20):
        alpha = 1
        
        f_current_x = self.optimization_problem.func_call(x)
        gradient_current_x =self.optimization_problem.gradient_call(x)
        slope_current_x = np.dot(gradient_current_x, direction)
        
        for _ in range (max_iterations):
            f_alpha_i = self.optimization_problem.func_call(x + alpha * direction)
            
            if f_alpha_i >= f_current_x + sigma * alpha * slope_current_x:
                alpha = alpha/2 
            else:
                gradient_alpha_i = self.optimization_problem.gradient_call(x + alpha * direction)
                if np.dot(gradient_alpha_i, direction) > (rho * slope_current_x):
                    alpha = alpha * 2 
                else:
                    return alpha                   
        return alpha


class NewtonMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem,initial_x)

    # general newton method with updating done via the gradient and the hessian matrix
    def run(self):
        # fitting initial values for x to be the correct data type
        x = np.array(self.initial_x, dtype=float)
        # choosen stopping criterion
        while np.linalg.norm(self.optimization_problem.gradient_call(x)) > 1e-3:
            self.path.append(x.copy())
            x += - np.dot(np.linalg.inv(self.optimization_problem.hessian_call(x)), self.optimization_problem.gradient_call(x))
        self.path.append(x.copy())
        return x
    
class Goldstein_Wolfe_method(optimizationMethod):
    # sigma : constant to check for armijo conditon
    # rho : constant to check for wolfe curvature condition
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
    
    def direction(self, x, gradient_current_x):
        Hessian_matrix = (self.optimization_problem.hessian_call(x))
        direction = - np.dot(np.linalg.inv(Hessian_matrix), gradient_current_x)
        return direction
    
    def run(self):
        
        x = np.array(self.initial_x, dtype=float)
        
        while np.linalg.norm(self.optimization_problem.gradient_call(x)) > 1e-3:
            self.path.append(x.copy())
            gradient_current_x = self.optimization_problem.gradient_call(x)
            direction = self.direction(x, gradient_current_x)
            alpha = self._compute_alpha(x, direction, sigma = 0.01, rho = 0.9, max_iterations = 20)
            
            x += alpha * direction
        self.path.append(x.copy())
        return x
    
    
class goodBroydenMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
        
    # update the hessian so it doesn't need to be calculated each step
    # call it B_k 
    
    # good broyden: x_k+1 = x_k - H_k * grad_k
    # grad_k is gradient evaluated at x_k
    # H_k needs to satisfy: H_k*delta_k = gamma_k
    # delta_k = x_k_next - x_k and gamma_k = grad_k_next - grad_k
    def run(self):
        H_k = np.identity(len(self.initial_x)) 
        x_k = np.array(self.initial_x, dtype=float)
        grad_k = self.optimization_problem.gradient_call(x_k)
        
        while np.linalg.norm(grad_k) > 1e-2:
            self.path.append(x_k.copy())
            
            direction = np.dot(H_k, grad_k)
            
            # udate x using inexact line method implemented with Goldstein Wolfe
            alpha = self._compute_alpha(x_k, direction)
            x_k_next = x_k -  alpha * direction
            
            grad_k_next = self.optimization_problem.gradient_call(x_k_next)
            
            delta_k = x_k_next - x_k 
            gamma_k = grad_k_next - grad_k
            
            H_k_next = H_k + np.outer((delta_k - np.dot(H_k, gamma_k)), np.dot(delta_k.T, H_k)) / np.dot(delta_k.T, np.dot(H_k, gamma_k))
            
            x_k = x_k_next
            grad_k = grad_k_next
            H_k = H_k_next
            
        self.path.append(x_k.copy())
        return x_k
    
    
class badBroydenMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
    def run(self):
        # H_k = np.identity(len(self.initial_x)) 
        H_k = np.ones((len(self.initial_x), len(self.initial_x)))
        x_k = np.array(self.initial_x, dtype=float)
        grad_k = self.optimization_problem.gradient_call(x_k)
        
        while np.linalg.norm(grad_k) > 1e-3:
            self.path.append(x_k.copy())
            
            direction = np.dot(H_k, grad_k)
            
            # udate x using inexact line method implemented with Goldstein Wolfe
            # alpha = self._compute_alpha(x_k, direction)
            # x_k_next = x_k -  alpha * direction
            x_k_next = x_k - direction
            
            grad_k_next = self.optimization_problem.gradient_call(x_k_next)
            
            delta_k = x_k_next - x_k 
            gamma_k = grad_k_next - grad_k
            
            H_k_next = np.outer((delta_k- np.dot(H_k, gamma_k)), np.dot(gamma_k.T, H_k)) / np.dot(gamma_k.T, np.dot(H_k, gamma_k))
            
            x_k = x_k_next
            grad_k = grad_k_next
            H_k = H_k_next
            print(f'{x_k} and {grad_k} and {H_k}')
            
        self.path.append(x_k.copy())
        return x_k
    
class symmetricBroydenMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
    def run(self):
        H_k = self.optimization_problem.hessian_call(self.initial_x)
        x_k = np.array(self.initial_x, dtype=float)
        grad_k = self.optimization_problem.gradient_call(x_k)
    
        while np.linalg.norm(grad_k) > 1e-3:
            self.path.append(x_k.copy())
            
            direction = np.dot(H_k, grad_k)
            
            # udate x using inexact line method implemented with Goldstein Wolfe
            alpha = self._compute_alpha(x_k, direction)
            x_k_next = x_k -  alpha * direction
            
            grad_k_next = self.optimization_problem.gradient_call(x_k_next)
            
            delta_k = x_k_next - x_k 
            gamma_k = grad_k_next - grad_k
            
            u = delta_k - np.dot(H_k, gamma_k)
            a = 1 / np.dot(u.T, gamma_k)
            
            H_k_next = H_k + np.dot(a, np.dot(u, u.T))
            
            x_k = x_k_next
            grad_k = grad_k_next
            H_k = H_k_next
            print(f'{x_k} and {grad_k} and {H_k}')

            
        self.path.append(x_k.copy())
        return x_k
        
class dfpMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
    def run(self):
        H_k = np.identity(len(self.initial_x)) 
        x_k = np.array(self.initial_x, dtype=float)
        grad_k = self.optimization_problem.gradient_call(x_k)
    
        return x_k
        
class bfgsMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)
        
    def run(self):
        H_k = np.identity(len(self.initial_x)) 
        x_k = np.array(self.initial_x, dtype=float)
        grad_k = self.optimization_problem.gradient_call(x_k)
    
        return x_k


def save_rosenbrock_plot(path, filename='rosenbrock_contour_plot.png', cmap='viridis'):
    # Create a grid of points to evaluate the Rosenbrock function
    #X, Y = np.meshgrid(np.linspace(-0.5, 2, 400), np.linspace(-0.5, 1.5, 400))
    #Z = 100 * (Y - X**2)**2 + (1 - X)**2  # Rosenbrock function computation
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Create contour lines
    #contour = ax.contour(X, Y, Z, levels=np.logspace(0, 5, 20), cmap=cmap)
    #ax.clabel(contour, inline=True, fontsize=10, fmt='%1.1f')  # Add labels to contours
    
    # Plot the optimization path
    if path is not None:
        path = np.array(path)  # Ensure path is a NumPy array
        ax.plot(path[:, 0], path[:, 1], 'k-', linewidth=2, label='Optimization Path')
        ax.plot(path[:, 0], path[:, 1], 'ro', markersize=5, label='Points')  # Points in red
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Rosenbrock Function Contour Plot')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300)
    plt.close(fig)  # Close the figure to free up memory


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
    print(f'the zero root: {newton.run()}')

    # rosenbrock optimization problem
    optimization_rosenbrock = optimizationProblem(rosenbrock_function())
    rosenbrock = NewtonMethod(optimization_rosenbrock, [0.1, 0.5])
    print(f'the zero root for newton method: {rosenbrock.run()}')

    # trying to implement with already input gradient
    optimization_rosenbrock = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    rosenbrock = NewtonMethod(optimization_rosenbrock, [-0.4, 0.8])
    print(f'the zero root for newton method with gradient: {rosenbrock.run()}')
    save_rosenbrock_plot(rosenbrock.path, filename='newton_rosenbrock_gradient_plot.png')
    
    
    #test_optimization for te rosenborkfunction
    Wolfe_rosenbrock_function = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    Wolfe = Goldstein_Wolfe_method(Wolfe_rosenbrock_function, [-0.4, 0.8])
    print(f'the zero root for goldstein_wolfe_method is: {Wolfe.run()}')
    save_rosenbrock_plot(Wolfe.path, filename='goldstein_wolfe_rosenbrock_plot.png')
    
    
    # test Quasi newton method 
    optimization_quasi_newton = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    '''
    good_broyden = goodBroydenMethod(optimization_quasi_newton, [-0.4, 0.2])
    print(f'the zero root for the good broyden method is: {good_broyden.run()}')
    
    bad_broyden = badBroydenMethod(optimization_quasi_newton, [-0.4, 0.8])
    print(f'the zero root for the bad broyden method is: {bad_broyden.run()}')
    '''
    symmetric_broyden = symmetricBroydenMethod(optimization_quasi_newton, [0.1,0.1])
    print(f'the zero root for the bad broyden is: {symmetric_broyden.run()}')