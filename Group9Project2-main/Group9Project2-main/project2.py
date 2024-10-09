import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols
from sympy import hessian
from multiprocessing import Pool

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

        assert isinstance(self.func, sp.Expr), "The function must be sympy symbolic equation, make sure you're not passing the creator function"

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

class NewtonMethod(optimizationMethod):
    def __init__(self, optimization_problem, initial_x):
        super().__init__(optimization_problem, initial_x)

    # general newton method with updating done via the gradient and the hessian matrix
    def run(self):
        # fitting initial values for x to be the correct data type
        x = np.array(self.initial_x, dtype=float)

        # counter for the number of steps (probably will be part of the stopping criterion in the final version)
        steps = 0
        # choosen stopping criterion
        while np.linalg.norm(self.optimization_problem.gradient_call(x)) > 1e-7:
            x += - np.dot(np.linalg.inv(self.optimization_problem.hessian_call(x)), self.optimization_problem.gradient_call(x))
            steps += 1
        return x, steps

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


if __name__ == '__main__':
    # test of the code
    # test_optimization = optimizationProblem(test_function())
    # newton = NewtonMethod(test_optimization,[1.0, 2.0])
    # print(f'the zero root: {newton.run()}')

    # rosenbrock optimization problem
    # optimization_rosenbrock = optimizationProblem(rosenbrock_function())
    # rosenbrock = NewtonMethod(optimization_rosenbrock, [0.1, 0.5])
    # print(f'the zero root: {rosenbrock.run()}')

    # # trying to implement with already input gradient
    # optimization_rosenbrock = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
    # rosenbrock = NewtonMethod(optimization_rosenbrock, [-0.4, 0.8])
    # print(f'the zero root: {rosenbrock.run()}')


    #%% rosenbrock plotting
    plot_rosenbrock_steps([0.5,1.5],[-1,3], 500)



